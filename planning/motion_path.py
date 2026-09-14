"""고정 팁과 시간 정보가 있는 관절 경로를 JSON으로 저장하고 불러온다.
여러 이동 구간을 같은 형식으로 묶고 재생 시각의 관절각을 보간한다.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kinematics.anchoring import TipAnchor
from kinematics.joints import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES
from kinematics.joint_limits import DEFAULT_CALIBRATION_PATH, load_joint_limits


@dataclass
class MotionSegment:
    r"""하나의 고정 팁을 유지하는 구간의 시간과 관절각을 담는다.

    time_s는 0부터 증가하는 초 단위 배열이다.
    q_rad는 J1부터 J7 순서의 \((N,7)\) 배열이다.
    gripper_q_rad는 G_L, G_R 순서의 \((N,2)\) 배열이며 None이면 XML 초기각을 사용한다.
    anchor는 해당 구간에서 고정할 팁의 월드 위치·방향이다.
    """

    name: str
    anchor: TipAnchor
    time_s: NDArray[np.float64]
    q_rad: NDArray[np.float64]
    gripper_q_rad: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """경로 배열을 복사하고 재생에 필요한 형상과 시간 순서를 검사한다."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("이동 구간의 이름이 필요합니다.")
        for name in ("time_s", "q_rad", "gripper_q_rad"):
            value = getattr(self, name)
            if value is None and name == "gripper_q_rad":
                continue
            if np.iscomplexobj(value):
                raise ValueError(f"{name}에는 실수만 사용할 수 있습니다.")
            array = np.array(value, dtype=float, copy=True)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name}에는 유한한 값만 사용할 수 있습니다.")
            setattr(self, name, array)
        if (
            self.time_s.ndim != 1 or len(self.time_s) < 2
            or self.time_s[0] != 0 or np.any(np.diff(self.time_s) <= 0)
        ):
            raise ValueError("구간 시간은 0부터 엄격히 증가하는 두 개 이상의 값이어야 합니다.")
        if self.q_rad.shape != (len(self.time_s), len(ARM_JOINT_NAMES)):
            raise ValueError("팔 관절 경로의 형상은 (시간 지점 수, 7)이어야 합니다.")
        if self.gripper_q_rad is not None and self.gripper_q_rad.shape != (len(self.time_s), 2):
            raise ValueError("손가락 관절 경로의 형상은 (시간 지점 수, 2)여야 합니다.")

    @property
    def duration_s(self) -> float:
        """해당 구간의 재생 시간을 초 단위로 반환한다."""
        return float(self.time_s[-1])


@dataclass
class MotionPath:
    """관절각이 연속으로 연결되는 이동 구간들을 재생 순서대로 담는다."""

    segments: tuple[MotionSegment, ...]

    def __post_init__(self) -> None:
        """구간 목록과 연결 지점의 팔 관절각이 일치하는지 검사한다."""
        self.segments = tuple(self.segments)
        if not self.segments:
            raise ValueError("경로에는 이동 구간이 하나 이상 필요합니다.")
        for before, after in zip(self.segments, self.segments[1:]):
            if not np.allclose(before.q_rad[-1], after.q_rad[0], atol=1e-8, rtol=0):
                raise ValueError("연결할 구간의 마지막·첫 팔 관절각이 일치해야 합니다.")

    @property
    def duration_s(self) -> float:
        """전체 구간의 재생 시간을 초 단위로 반환한다."""
        return sum(segment.duration_s for segment in self.segments)

    def validate_limits(self, calibration_path: str | Path = DEFAULT_CALIBRATION_PATH) -> None:
        """전체 경로 지점을 현재 팔·손가락 제한과 비교하고 첫 위반 위치를 알린다.

        각 관절의 고정된 구간 제한은 양 끝점이 만족하면 선형 보간 중에도 만족한다.
        이 검사는 속도·충돌 및 실물 추종 오차를 평가하지 않는다.
        """
        limits = load_joint_limits((*ARM_JOINT_NAMES, *GRIPPER_JOINT_NAMES), calibration_path)
        for segment in self.segments:
            grippers = segment.gripper_q_rad
            if grippers is None:
                grippers = np.zeros((len(segment.time_s), len(GRIPPER_JOINT_NAMES)))
            for names, values in ((ARM_JOINT_NAMES, segment.q_rad), (GRIPPER_JOINT_NAMES, grippers)):
                for index, name in enumerate(names):
                    lower, upper = limits[name]
                    invalid = np.flatnonzero((values[:, index] < lower) | (values[:, index] > upper))
                    if invalid.size:
                        point = int(invalid[0])
                        raise ValueError(
                            f"{segment.name}의 {point + 1}번째 지점: {name} 각도 "
                            f"{np.rad2deg(values[point, index]):.3f}°가 현재 제한 "
                            f"{np.rad2deg(lower):.3f}°부터 {np.rad2deg(upper):.3f}°를 벗어났습니다."
                        )

    def sample(self, elapsed_s: float) -> tuple[int, NDArray[np.float64], NDArray[np.float64] | None]:
        r"""재생 시각에 해당하는 구간 번호와 팔·손가락 관절각을 반환한다.

        $$
        q(t)=(1-u)q_i+u q_{i+1},\qquad u=\frac{t-t_i}{t_{i+1}-t_i}
        $$

        \(t\)는 선택된 구간 안에서의 시간이다. 범위 밖 시각은 시작 또는 끝에 맞춘다.
        이 시간 정보는 재생용이며 모터 속도 제한이나 동역학 검증을 뜻하지 않는다.
        """
        if not np.isfinite(elapsed_s):
            raise ValueError("재생 시각은 유한한 값이어야 합니다.")
        local_time = float(np.clip(elapsed_s, 0, self.duration_s))
        for segment_index, segment in enumerate(self.segments):
            if local_time <= segment.duration_s or segment_index == len(self.segments) - 1:
                break
            # 지나온 구간의 시간을 제외: $$t_{\mathrm{local}}=t_{\mathrm{local}}-T_{\mathrm{segment}}$$
            local_time -= segment.duration_s
        index = int(np.searchsorted(segment.time_s, local_time, side="right") - 1)
        index = min(max(index, 0), len(segment.time_s) - 2)
        # 두 시간 지점 사이의 진행 비율: $$u=(t-t_i)/(t_{i+1}-t_i)$$
        fraction = (local_time - segment.time_s[index]) / (segment.time_s[index + 1] - segment.time_s[index])
        # 팔 관절각의 선형 보간: $$q(t)=(1-u)q_i+u q_{i+1}$$
        q_rad = (1 - fraction) * segment.q_rad[index] + fraction * segment.q_rad[index + 1]
        gripper_q_rad = None
        if segment.gripper_q_rad is not None:
            # 손가락 관절각의 선형 보간: $$g(t)=(1-u)g_i+u g_{i+1}$$
            gripper_q_rad = (1 - fraction) * segment.gripper_q_rad[index] + fraction * segment.gripper_q_rad[index + 1]
        return segment_index, q_rad, gripper_q_rad


def save_path(
    motion_path: MotionPath, file_path: str | Path,
    *, calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
) -> Path:
    """현재 제한을 검사한 뒤 관절 순서·단위와 고정 자세를 JSON 파일로 저장한다."""
    motion_path.validate_limits(calibration_path)
    payload = {
        "format": "xs.motion_path.v1",
        "angle_unit": "rad",
        "length_unit": "m",
        "time_unit": "s",
        "joint_names": list(ARM_JOINT_NAMES),
        "gripper_joint_names": list(GRIPPER_JOINT_NAMES),
        "segments": [
            {
                "name": segment.name,
                "fixed_tip": segment.anchor.fixed_tip,
                "T_world_fixed_tip": segment.anchor.T_world_fixed_tip.tolist(),
                "time_s": segment.time_s.tolist(),
                "q_rad": segment.q_rad.tolist(),
                "gripper_q_rad": None if segment.gripper_q_rad is None else segment.gripper_q_rad.tolist(),
            }
            for segment in motion_path.segments
        ],
    }
    destination = Path(file_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return destination


def load_path(
    file_path: str | Path, *, calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
) -> MotionPath:
    """저장된 JSON의 형식과 단위를 확인하고 현재 관절 제한으로 다시 검사한다."""
    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != "xs.motion_path.v1":
        raise ValueError("지원하지 않는 경로 파일 형식입니다.")
    if (
        payload.get("joint_names") != list(ARM_JOINT_NAMES)
        or payload.get("gripper_joint_names") != list(GRIPPER_JOINT_NAMES)
        or payload.get("angle_unit") != "rad"
        or payload.get("length_unit") != "m"
        or payload.get("time_unit") != "s"
    ):
        raise ValueError("경로 파일의 관절 순서 또는 단위가 일치하지 않습니다.")
    try:
        motion_path = MotionPath(tuple(
            MotionSegment(
                name=item["name"], anchor=TipAnchor(item["fixed_tip"], item["T_world_fixed_tip"]),
                time_s=item["time_s"], q_rad=item["q_rad"], gripper_q_rad=item.get("gripper_q_rad"),
            )
            for item in payload["segments"]
        ))
    except (KeyError, TypeError) as error:
        raise ValueError("경로 파일의 구간 항목을 확인해 주세요.") from error
    motion_path.validate_limits(calibration_path)
    return motion_path
