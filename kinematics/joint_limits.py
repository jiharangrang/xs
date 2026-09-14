"""캘리브레이션 파일의 관절 범위를 실물 제어와 경로 계산에 공통으로 제공한다.
파일의 라디안 값을 유지하며 모터 통신이나 모델 로드는 수행하지 않는다.
"""

import math
from pathlib import Path
from typing import Sequence

import yaml


DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "models/xs/calibration.yaml"


def limits_from_document(document: dict, names: Sequence[str]) -> dict[str, tuple[float, float]]:
    """이미 읽은 설정에서 요청한 관절의 유효한 라디안 하한과 상한을 반환한다."""
    limits = {}
    try:
        for name in names:
            entry = document["joints"][name]
            values = (entry["lower_rad"], entry["upper_rad"])
            if any(isinstance(value, bool) for value in values):
                raise ValueError(f"{name}의 관절 제한에는 숫자를 사용해야 합니다.")
            lower, upper = map(float, values)
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(f"{name}의 관절 하한과 상한이 올바르지 않습니다.")
            limits[name] = (lower, upper)
    except (KeyError, TypeError) as error:
        raise ValueError("캘리브레이션 파일의 관절 이름과 제한 항목을 확인해 주세요.") from error
    return limits


def load_joint_limits(
    names: Sequence[str], calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
) -> dict[str, tuple[float, float]]:
    """지정한 캘리브레이션 파일에서 관절 제한을 읽으며 XML 범위로 대체하지 않는다."""
    try:
        document = yaml.safe_load(Path(calibration_path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"관절 설정 파일 형식이 올바르지 않습니다: {calibration_path}") from error
    return limits_from_document(document, names)
