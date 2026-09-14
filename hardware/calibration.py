"""모터의 현재 자세를 중점과 관절 영점으로 설정하고 캘리브레이션 파일에 저장한다.
관절의 평상시 읽기·이동은 담당하지 않으며 설정 도중 실패하면 미완료 상태를 남긴다.
"""

from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

from hardware.joint_control import DEFAULT_CALIBRATION_PATH, load_joint_calibration
from hardware.sts3215 import MIDPOINT_RAW, MotorError, STS3215Bus


class MotorCalibration:
    """기준 자세 설정과 영점 파일 저장 절차를 담당한다."""

    def __init__(
        self, bus: STS3215Bus, calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
    ) -> None:
        """열린 모터 버스와 저장할 설정 파일을 연결한다."""
        self._bus = bus
        self._path = Path(calibration_path)
        self._document, self._calibrations = load_joint_calibration(self._path)

    def _fresh_document(self) -> dict:
        """다른 프로그램이 설정 파일을 바꾸었으면 덮어쓰지 않도록 알린다."""
        document, _ = load_joint_calibration(self._path)
        if document != self._document:
            raise ValueError("관절 설정 파일이 변경되었습니다. 서버를 다시 열고 설정해 주세요.")
        return document

    def _save_document(self, document: dict) -> None:
        """설정 전체를 임시 파일로 기록한 뒤 기존 파일을 교체한다."""
        self._fresh_document()
        temporary_path = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self._path.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                yaml.safe_dump(document, temporary, allow_unicode=True, sort_keys=False, default_flow_style=None)
            temporary_path.replace(self._path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        self._document = deepcopy(document)
        _, self._calibrations = load_joint_calibration(self._path)

    def save_zero(self, joint: str) -> None:
        """정지한 현재 자세를 관절 영점으로 저장하며 모터 레지스터는 바꾸지 않는다.

        기존 home_single은 보존하고 home_raw에 새 기준을 저장한다.
        실행 후 관절 제어기는 reload_calibration으로 새 기준을 읽는다.
        """
        document = self._fresh_document()
        if document.get("midpoint_pending", False):
            raise MotorError("중점 설정이 미완료입니다. 기준 자세에서 전체 중점·영점 설정을 다시 실행해 주세요.")
        if joint not in self._calibrations:
            raise ValueError(f"알 수 없는 관절입니다: {joint}")
        calibration = self._calibrations[joint]
        position, speed = self._bus.read_position_speed(calibration.servo_id)
        calibration.raw_to_degrees(position)
        if speed != 0:
            raise ValueError("모터가 움직이고 있습니다. 기준 자세에서 멈춘 뒤 영점을 저장해 주세요.")
        document["joints"][joint]["home_raw"] = position
        self._save_document(document)

    def calibrate_all_zero(self) -> tuple[str, ...]:
        """등록된 모든 모터의 현재 자세를 중점으로 재정의하고 관절 영점으로 저장한다.

        모든 모터의 준비 상태와 파일 저장 가능 여부를 확인한 뒤 중점 명령을 보낸다.
        작업 도중 실패하면 미완료 표시를 유지해 재시작 후에도 이전 영점 사용을 막는다.
        실행 후 관절 제어기는 성공 여부와 관계없이 설정 파일을 다시 읽어야 한다.
        """
        document = self._fresh_document()
        for name, calibration in self._calibrations.items():
            if not calibration.lower_deg <= 0 <= calibration.upper_deg:
                raise ValueError(f"{name}의 관절 범위에 영점이 포함되지 않습니다.")
            self._bus.check_midpoint_setup(calibration.servo_id)
        document["midpoint_pending"] = True
        self._save_document(document)
        completed = []
        try:
            for name, calibration in self._calibrations.items():
                self._bus.calibrate_midpoint(calibration.servo_id)
                completed.append(name)
            document = self._fresh_document()
            for name in completed:
                document["joints"][name]["home_raw"] = MIDPOINT_RAW
            document.pop("midpoint_pending", None)
            self._save_document(document)
        except (MotorError, OSError, ValueError) as error:
            progress = ", ".join(completed) or "없음"
            raise MotorError(f"전체 중점·영점 설정 미완료. 중점 확인: {progress}. {error}") from error
        return tuple(completed)
