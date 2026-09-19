"""4단계 출발 위치를 횡방향 삽입의 독립된 목표 기준으로 보관한다.
높이나 관절각을 되돌리지 않고 빔 폭 방향의 남은 복귀 거리만 계산한다.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from kinematics.joints import ARM_JOINT_NAMES


DEFAULT_RETURN_REFERENCE_PATH = Path(__file__).resolve().parents[1] / "outputs/stage4_return_reference.json"


@dataclass(frozen=True)
class LateralReference:
    """월드 기준 출발 팁 위치와 빔 바깥 방향 및 원본 실행 식별자를 담는다."""

    point_world_m: tuple[float, float, float]
    outward_world: tuple[float, float, float]
    source_run_id: str

    def __post_init__(self):
        r"""유효한 위치와 실행 식별자를 확인하고 횡방향을 단위 벡터로 보관한다.

        $$u_W=\tilde u_W/\|\tilde u_W\|$$
        """
        point = np.asarray(self.point_world_m, dtype=float)
        outward = np.asarray(self.outward_world, dtype=float)
        if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (point, outward)):
            raise ValueError("4단계 출발 위치와 빔 횡방향이 유효하지 않습니다.")
        if np.linalg.norm(outward) < 1e-8 or not isinstance(self.source_run_id, str) or not self.source_run_id:
            raise ValueError("4단계 복귀 기준의 방향과 실행 기록을 확인해 주세요.")
        # 원본 빔 바깥 방향의 정규화: $$u_W=\tilde u_W/\|\tilde u_W\|$$
        outward = outward / np.linalg.norm(outward)
        object.__setattr__(self, "point_world_m", tuple(point.tolist()))
        object.__setattr__(self, "outward_world", tuple(outward.tolist()))

    def remaining(self, tip_world_m):
        r"""현재 팁에서 출발 횡위치까지의 부호 있는 복귀 거리를 계산한다.

        $$e_y=u_W^T(p_{tip,W}-p_{start,W})$$

        양수이면 빔 안쪽으로 이동하고 음수이면 출발 횡위치를 지나친 상태이다.
        """
        point = np.asarray(tip_world_m, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("현재 팁 위치가 유효하지 않습니다.")
        # 출발 위치에 대한 현재 팁 변위: $$\Delta p_W=p_{tip,W}-p_{start,W}$$
        displacement = point - np.asarray(self.point_world_m)
        # 높이와 길이 방향을 제외한 횡방향 오차: $$e_y=u_W^T\Delta p_W$$
        return float(np.dot(self.outward_world, displacement))

    def as_dict(self):
        """상태 표시와 저장에 사용할 복귀 기준의 복사본을 반환한다."""
        return {"schema_version": 1, "kind": "stage4_start_lateral", **asdict(self)}


def reference_from_stage4(status, fk):
    r"""완료된 4단계의 출발 위치와 도착 시 관측한 빔 방향으로 복귀 기준을 만든다.

    $$u_W=R_{WC}\frac{u_C-(u_C^Tn_C)n_C}{\|u_C-(u_C^Tn_C)n_C\|}$$
    """
    if status.get("state") != "REACHED" or not status.get("arrival"):
        raise ValueError("완료된 4단계 출발 기준이 없습니다. 먼저 4단계를 완료해 주세요.")
    try:
        arrival, observed = status["arrival"], status["observation"]
        # 관측 당시 실측 관절각의 라디안 변환: $$q_{rad}=q_{deg}\pi/180$$
        q = np.deg2rad([arrival["positions_deg"][name] for name in ARM_JOINT_NAMES])
        camera = fk.depth_camera_pose(q)
        normal = np.asarray(observed["normal"], dtype=float)
        outward = np.asarray(observed["outward"], dtype=float)
        if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (normal, outward)) or np.linalg.norm(normal) < 1e-8:
            raise ValueError("4단계 관측의 빔 방향이 유효하지 않습니다.")
        # 관측 평면 법선의 정규화: $$n_C=\tilde n_C/\|\tilde n_C\|$$
        normal = normal / np.linalg.norm(normal)
        # 빔 평면에 투영한 횡방향: $$u'_C=u_C-(u_C^Tn_C)n_C$$
        outward = outward - float(outward @ normal) * normal
        # 카메라 횡방향을 월드로 회전: $$u'_W=R_{WC}u'_C$$
        outward_world = camera[:3, :3] @ outward
        return LateralReference(arrival["start_tip_world_m"], outward_world, status["run_id"])
    except (KeyError, TypeError) as error:
        raise ValueError("4단계 기록에 출발 위치와 빔 방향이 필요합니다.") from error


def save_lateral_reference(reference, path):
    """재시작 후 같은 복귀 목표를 사용하도록 기준을 원자적으로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(reference.as_dict(), ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_lateral_reference(path):
    """서버 재시작 전에 저장한 4단계 복귀 기준을 읽는다."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("schema_version") != 1 or document.get("kind") != "stage4_start_lateral":
            raise ValueError("저장된 4단계 복귀 기준의 형식을 확인해 주세요.")
        return LateralReference(document["point_world_m"], document["outward_world"], document["source_run_id"])
    except FileNotFoundError as error:
        raise ValueError("4단계 출발 기준이 없습니다. 먼저 4단계를 완료해 주세요.") from error
    except (KeyError, TypeError) as error:
        raise ValueError("저장된 4단계 복귀 기준의 위치와 방향을 확인해 주세요.") from error
