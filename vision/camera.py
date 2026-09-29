"""Gemini 215의 컬러 스트림만 열어 데모용 BGR 영상을 읽는다.
깊이 스트림이나 로봇 제어 모듈은 사용하지 않는다.
"""

import time

import cv2
import numpy as np


class RGBCamera:
    """컬러 영상의 연결·수신·종료를 한 문맥에서 관리한다."""

    def __init__(self, serial=None):
        """장치를 열지 않고 선택할 일련번호를 보관한다."""
        self.serial = serial
        self.context = None
        self.pipeline = None

    def __enter__(self):
        """Gemini 215 한 대의 원본 컬러 스트림을 시작한다."""
        import pyorbbecsdk as sdk

        sdk.Context.set_logger_to_console(sdk.OBLogLevel.ERROR)
        sdk.Context.set_logger_to_file(sdk.OBLogLevel.NONE, "")
        try:
            self.context = sdk.Context()
            devices = self.context.query_devices()
            matches = [i for i in range(devices.get_count())
                       if "Gemini 215" in devices.get_device_name_by_index(i)
                       and (self.serial is None
                            or devices.get_device_serial_number_by_index(i) == self.serial)]
            if len(matches) != 1:
                raise RuntimeError(f"Gemini 215가 {len(matches)}대입니다. USB 연결 또는 --serial을 확인하세요.")
            device = devices.get_device_by_index(matches[0])
            self.pipeline = sdk.Pipeline(device)
            profiles = self.pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
            profile = profiles.get_video_stream_profile(1920, 1080, sdk.OBFormat.MJPG, 15)
            config = sdk.Config()
            config.enable_stream(profile)
            self.pipeline.start(config)
            for _ in range(15):
                self.read()
            print(f"RGB 연결: {device.get_device_info().get_name()} / 1920×1080", flush=True)
            return self
        except Exception as error:
            self.close()
            if "uvc_open" in str(error) and "-3" in str(error):
                raise RuntimeError(
                    "카메라 USB 접근 권한이 없습니다. vision 폴더에서 "
                    "sudo .venv/bin/python capture.py 또는 sudo .venv/bin/python live.py로 실행하세요."
                ) from error
            raise RuntimeError(
                f"카메라 연결 실패: {error}\n"
                "기존 자세 GUI의 카메라 표시나 다른 카메라 프로그램이 켜져 있다면 먼저 종료하세요."
            ) from error

    def read(self):
        """제한 시간 안에 받은 원본 컬러 프레임을 독립적인 BGR 배열로 반환한다."""
        if self.pipeline is None:
            raise RuntimeError("카메라가 열려 있지 않습니다.")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            frames = self.pipeline.wait_for_frames(500)
            frame = None if frames is None else frames.get_color_frame()
            if frame is None:
                continue
            data = np.frombuffer(frame.get_data(), dtype=np.uint8)
            image_format = frame.get_format().name
            if image_format == "MJPG":
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("카메라 JPEG를 해독하지 못했습니다.")
                return image
            image = data.reshape(frame.get_height(), frame.get_width(), 3)
            if image_format == "RGB":
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if image_format == "BGR":
                return image.copy()
            raise RuntimeError(f"지원하지 않는 컬러 형식: {image_format}")
        raise RuntimeError("5초 동안 RGB 영상을 받지 못했습니다. USB 연결을 확인하세요.")

    def close(self):
        """카메라 자원을 반환하며 원래 오류가 종료 오류에 가려지지 않게 한다."""
        pipeline, self.pipeline = self.pipeline, None
        try:
            if pipeline is not None:
                pipeline.stop()
        except Exception as error:
            print(f"카메라 종료 중 오류: {error}", flush=True)
        finally:
            self.context = None

    def __exit__(self, *args):
        """정상 종료와 예외 발생 모두에서 카메라를 닫는다."""
        self.close()
