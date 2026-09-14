"""Gemini 215에서 RGB와 원본 깊이 영상을 함께 읽는다.
깊이는 미터로 변환하며 빔 인식·영상 정합·로봇 좌표 변환은 수행하지 않는다.
"""

from dataclasses import dataclass
import importlib
import math
import time
from typing import Self

import numpy as np
from numpy.typing import NDArray


class CameraError(RuntimeError):
    """카메라 연결·영상 수신·변환 실패를 나타낸다."""


@dataclass(frozen=True)
class RGBDFrame:
    """RGB와 원본 깊이, 각 영상의 장치 시각 및 호스트 수신 시각을 보관한다.

    rgb는 RGB 순서의 uint8 배열, depth_m은 미터 단위 float32 배열이다.
    깊이의 무효 픽셀은 NaN이며 두 영상은 픽셀 위치가 정합되지 않은 원본이다.
    장치 시각은 마이크로초이며 received_monotonic_s와 시계 기준이 다르다.
    """

    rgb: NDArray[np.uint8]
    depth_m: NDArray[np.float32]
    color_timestamp_us: int
    depth_timestamp_us: int
    received_monotonic_s: float


def _load_sdk():
    """카메라 사용 시에만 선택 의존성인 제조사 SDK를 불러온다."""
    try:
        return importlib.import_module("pyorbbecsdk")
    except ImportError as error:
        raise CameraError("카메라 의존성이 없습니다. uv sync --extra camera 를 실행하세요.") from error


def _decode_rgb(frame) -> NDArray[np.uint8]:
    """SDK 영상 버퍼를 독립적인 RGB 배열로 변환한다."""
    import cv2

    image_format = frame.get_format().name
    shape = (frame.get_height(), frame.get_width(), 3)
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if image_format == "MJPG":
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != shape:
            raise CameraError("RGB JPEG 영상을 해독하지 못했습니다.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if image_format == "RGB":
        return data.reshape(shape).copy()
    if image_format == "BGR":
        return data.reshape(shape)[:, :, ::-1].copy()
    raise CameraError(f"지원하지 않는 컬러 형식입니다: {image_format}")


def _decode_depth(frame) -> NDArray[np.float32]:
    r"""깊이 원시값을 미터로 변환하고 무효 픽셀을 NaN으로 표시한다.

    $$z_{\mathrm{m}}=d_{\mathrm{raw}}\,s_{\mathrm{mm}}/1000$$

    d_raw는 Y16 깊이 값이고 s_mm은 장치가 보고한 원시값당 밀리미터이다.
    """
    if frame.get_format().name != "Y16":
        raise CameraError("깊이 영상은 Y16 형식이어야 합니다.")
    scale_mm = frame.get_depth_scale()
    if not math.isfinite(scale_mm) or scale_mm <= 0:
        raise CameraError("장치가 보고한 깊이 단위가 올바르지 않습니다.")
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(frame.get_height(), frame.get_width())
    # 원시 깊이를 미터로 환산: $$z_{\mathrm{m}}=d_{\mathrm{raw}}s_{\mathrm{mm}}/1000$$
    depth_m = raw.astype(np.float32) * (scale_mm / 1000.0)
    depth_m[raw == 0] = np.nan
    return depth_m


def _profile_info(profile) -> dict:
    """현재 스트림의 해상도·속도·렌즈 파라미터를 장치에서 읽는다."""
    intrinsic = profile.get_intrinsic()
    distortion = profile.get_distortion()
    return {
        "width": profile.get_width(), "height": profile.get_height(),
        "fps": profile.get_fps(), "format": profile.get_format().name,
        "intrinsics": {name: getattr(intrinsic, name) for name in ("fx", "fy", "cx", "cy")},
        "distortion": {name: getattr(distortion, name) for name in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")},
        "distortion_model": distortion.model.name,
    }


class Gemini215Camera:
    """한 실행 흐름에서 카메라를 열고 RGB·깊이 프레임을 읽는 입력 통로다."""

    def __init__(self, serial: str | None = None, *, fps: int = 15) -> None:
        """장치 일련번호와 RGB·깊이의 공통 프레임 속도를 지정한다."""
        if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps는 양의 정수여야 합니다.")
        self.serial = serial
        self.fps = fps
        self.info: dict = {}
        self._context = None
        self._pipeline = None

    def open(self) -> Self:
        """Gemini 215 하나를 선택하고 원본 RGB·깊이 스트림을 시작한다."""
        if self._pipeline is not None:
            return self
        sdk = _load_sdk()
        sdk.Context.set_logger_to_console(sdk.OBLogLevel.ERROR)
        sdk.Context.set_logger_to_file(sdk.OBLogLevel.NONE, "")
        pipeline = None
        try:
            self._context = sdk.Context()
            devices = self._context.query_devices()
            matches = [i for i in range(devices.get_count())
                       if "Gemini 215" in devices.get_device_name_by_index(i)
                       and (self.serial is None or devices.get_device_serial_number_by_index(i) == self.serial)]
            if len(matches) != 1:
                raise CameraError(f"조건에 맞는 Gemini 215가 {len(matches)}개입니다. 연결 또는 --serial 값을 확인하세요.")
            device = devices.get_device_by_index(matches[0])
            device_info = device.get_device_info()
            pipeline = sdk.Pipeline(device)
            color = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR).get_video_stream_profile(
                1920, 1080, sdk.OBFormat.MJPG, self.fps)
            depth = pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR).get_video_stream_profile(
                1280, 800, sdk.OBFormat.Y16, self.fps)
            config = sdk.Config()
            config.enable_stream(color)
            config.enable_stream(depth)
            config.set_frame_aggregate_output_mode(sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            pipeline.enable_frame_sync()
            pipeline.start(config)
            extrinsic = depth.get_extrinsic_to(color)
            self.info = {
                "name": device_info.get_name(), "serial": device_info.get_serial_number(),
                "firmware": device_info.get_firmware_version(),
                "connection": device_info.get_connection_type(),
                "color": _profile_info(color), "depth": _profile_info(depth),
                "depth_to_color": {"rotation": np.asarray(extrinsic.rot).reshape(3, 3).tolist(),
                                   "translation_mm": np.asarray(extrinsic.transform).tolist()},
                "aligned": False, "depth_unit": "m", "invalid_depth": "NaN",
            }
            self._pipeline = pipeline
            return self
        except Exception as error:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            self._context = None
            self.info = {}
            if isinstance(error, CameraError):
                raise
            raise CameraError(f"카메라를 열지 못했습니다: {error}") from error

    def read(self, *, timeout_ms: int = 5000) -> RGBDFrame:
        """제한 시간 안에 RGB와 깊이가 모두 포함된 새 프레임을 읽는다."""
        if self._pipeline is None:
            raise CameraError("먼저 카메라를 열어 주세요.")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValueError("timeout_ms는 양의 정수여야 합니다.")
        deadline = time.monotonic() + timeout_ms / 1000.0
        try:
            while time.monotonic() < deadline:
                remaining_ms = max(1, math.ceil((deadline - time.monotonic()) * 1000))
                frames = self._pipeline.wait_for_frames(remaining_ms)
                if frames is None:
                    continue
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if color is None or depth is None:
                    continue
                received = time.monotonic()
                return RGBDFrame(_decode_rgb(color), _decode_depth(depth),
                                 color.get_timestamp_us(), depth.get_timestamp_us(), received)
        except Exception as error:
            raise CameraError(f"RGB·깊이 수신 실패: {error}") from error
        raise CameraError(f"{timeout_ms} ms 안에 RGB·깊이 영상을 받지 못했습니다.")

    def close(self) -> None:
        """영상 수신을 중단하고 장치 연결을 해제한다."""
        pipeline, self._pipeline = self._pipeline, None
        try:
            if pipeline is not None:
                pipeline.stop()
        finally:
            self._context = None

    def __enter__(self) -> Self:
        """문맥 관리자 진입 시 카메라를 연다."""
        return self.open()

    def __exit__(self, *args: object) -> None:
        """오류 여부와 관계없이 카메라를 닫는다."""
        self.close()
