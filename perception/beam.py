"""깊이에서 직선 빔의 평면·긴 모서리·폭 중심선을 추정한다.
카메라 기준 미터 좌표를 반환하며 CAD 정답이나 모터 명령을 사용하지 않는다.
"""

from dataclasses import dataclass, fields

import numpy as np

from perception.depth import depth_points, plane_points, rectify_image


class BeamDetectionError(ValueError):
    """평면을 추정할 만큼 유효한 관측이 없음을 나타낸다."""


@dataclass(frozen=True)
class BeamEstimate:
    """관측된 빔 정보와 품질을 담으며 확인하지 못한 항목은 None으로 둔다.

    normal은 카메라를 향하는 단위 법선이다. centerline_point_m은 빔 전체의
    중앙이 아니라 무한 중심선 위에서 카메라에 가장 가까운 점이다.
    width_m은 관측한 깊이 경계 사이의 폭이며, 실제 모서리인지 영상 확인이 필요하다.
    """

    normal: np.ndarray
    plane_offset_m: float
    plane_rms_m: float
    inlier_fraction: float
    plane_mask: np.ndarray
    status: str = "plane_only"
    axis: np.ndarray | None = None
    width_axis: np.ndarray | None = None
    width_m: float | None = None
    centerline_point_m: np.ndarray | None = None
    edge_lines_m: np.ndarray | None = None

    def as_dict(self) -> dict:
        """큰 영상 마스크를 제외한 결과를 JSON으로 저장할 수 있게 바꾼다."""
        result = {}
        for field in fields(self):
            if field.name == "plane_mask":
                continue
            value = getattr(self, field.name)
            result[field.name] = value.tolist() if isinstance(value, np.ndarray) else value
        return {"coordinate_frame": "depth_camera_optical", "length_unit": "m", **result}


def _least_squares_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    r"""점들의 수직 거리 제곱합이 가장 작은 평면을 구한다.

    $$n=\operatorname{eigmin}(Q^TQ),\quad d=-n^T\bar p,\quad Q_i=p_i-\bar p$$

    n은 단위 법선, d는 평면 상수이며 법선은 카메라 쪽을 향하게 한다.
    """
    center = points.mean(axis=0)
    # 점들의 중심을 제거: $$Q_i=p_i-\bar p$$
    centered = points - center
    # 평면의 법선은 산포 행렬의 최소 고유벡터: $$n=\operatorname{eigmin}(Q^TQ)$$
    values, vectors = np.linalg.eigh(centered.T @ centered)
    if values[1] <= 1e-10:
        raise BeamDetectionError("점들이 선에 몰려 있어 평면 방향을 구할 수 없습니다.")
    normal = vectors[:, 0]
    if normal @ center > 0:
        normal = -normal
    # 평면의 상수항: $$d=-n^T\bar p$$
    offset = -float(normal @ center)
    return normal, offset


def _ransac_plane(points: np.ndarray, threshold_m: float, seed: int) -> tuple[np.ndarray, float]:
    r"""튀는 깊이에 덜 민감하도록 무작위 평면 후보 중 지지가 큰 것을 고른다.

    $$\mathcal I=\{i:|n^Tp_i+d|\leq\tau\}$$

    I는 평면에 가까운 점의 집합, tau는 허용 수직 거리다.
    """
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(len(points), 6000), replace=False)]
    best = np.zeros(len(sample), dtype=bool)
    for _ in range(128):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        # 세 점이 정의하는 평면의 법선: $$n'=(b-a)\times(c-a)$$
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        # 법선을 단위 길이로 정규화: $$n=n'/\|n'\|$$
        normal = normal / length
        # 후보 평면까지의 수직 거리 판정: $$i\in\mathcal I\iff|n^T(p_i-a)|\leq\tau$$
        inliers = np.abs((sample - a) @ normal) <= threshold_m
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 100:
        raise BeamDetectionError("충분한 점이 함께 놓이는 평면을 찾지 못했습니다.")
    normal, offset = _least_squares_plane(sample[best])
    for _ in range(2):
        # 전체 관측에서 평면을 지지하는 점을 다시 선택: $$\mathcal I=\{i:|n^Tp_i+d|\leq\tau\}$$
        inliers = np.abs(points @ normal + offset) <= threshold_m
        if inliers.sum() < 100:
            raise BeamDetectionError("평면을 다시 맞추는 동안 유효 점이 부족해졌습니다.")
        normal, offset = _least_squares_plane(points[inliers])
    return normal, offset


def _boundary_lines(mask: np.ndarray, profile: dict, normal: np.ndarray, offset: float,
                    bounds: tuple[int, int, int, int]) -> list[np.ndarray]:
    r"""바깥 윤곽에 강건한 직선을 맞추고 관측 구간을 3차원 선분으로 반환한다.

    $$h_i=\|(p_i-a)\times t\|,\quad \ell(s)=\bar p+st$$

    h는 직선까지 거리, t는 단위 방향이다. 영상·ROI 경계와 내부 구멍은 제외한다.
    """
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pixels = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    x0, y0, x1, y1 = bounds
    margin = max(3, round(min(mask.shape) * 0.01))
    interior = ((pixels[:, 0] >= x0 + margin) & (pixels[:, 0] < x1 - margin)
                & (pixels[:, 1] >= y0 + margin) & (pixels[:, 1] < y1 - margin))
    points = plane_points(pixels[interior], profile, normal, offset)
    points = points[np.all(np.isfinite(points), axis=1)]
    rng = np.random.default_rng(0)
    if len(points) > 4000:
        points = points[rng.choice(len(points), 4000, replace=False)]
    lines = []
    for _ in range(6):
        if len(points) < 50:
            break
        best = np.zeros(len(points), dtype=bool)
        for _ in range(128):
            a, b = points[rng.choice(len(points), 2, replace=False)]
            # 직선 후보의 방향: $$v=b-a$$
            vector = b - a
            length = np.linalg.norm(vector)
            if length < 0.025:
                continue
            # 후보의 단위 방향: $$t=v/\|v\|$$
            axis = vector / length
            # 경계점과 후보 직선 사이의 거리: $$h_i=\|(p_i-a)\times t\|$$
            distances = np.linalg.norm(np.cross(points - a, axis), axis=1)
            inliers = distances < 0.001
            if inliers.sum() > best.sum():
                best = inliers
        if best.sum() < 50:
            break
        selected, points = points[best], points[~best]
        center = selected.mean(axis=0)
        # 경계점의 중심을 제거: $$Q_i=p_i-\bar p$$
        centered = selected - center
        # 지지하는 경계점의 주방향: $$t=\operatorname{eigmax}(Q^TQ)$$
        _, vectors = np.linalg.eigh(centered.T @ centered)
        axis = vectors[:, -1]
        # 이상점의 영향을 줄인 관측 구간: $$(s_0,s_1)=\operatorname{quantile}(Q_i^Tt;0.01,0.99)$$
        limits = np.quantile(centered @ axis, [0.01, 0.99])
        if limits[1] - limits[0] < 0.025:
            continue
        # 관측 선분의 양 끝을 공간 좌표로 복원: $$p_j=\bar p+s_jt$$
        lines.append(center + limits[:, None] * axis)
    return sorted(lines, key=lambda line: np.linalg.norm(line[1] - line[0]), reverse=True)


def _edge_geometry(lines: list[np.ndarray], normal: np.ndarray,
                   axis_hint: np.ndarray | None) -> dict:
    r"""평행하고 관측 구간이 겹치는 두 긴 경계로 폭과 중심선을 계산한다.

    $$b=\frac{n\times t}{\|n\times t\|},\quad
    w=|b^T(p_2-p_1)|,\quad c=(I-tt^T)(p_1+p_2)/2$$

    t와 b는 길이·폭 방향, p는 경계의 중점, c는 카메라에 가장 가까운 중심선 점이다.
    """
    if not lines:
        return {}
    candidates = []
    for i, first in enumerate(lines):
        for second in lines[i + 1:]:
            directions = np.array([first[1] - first[0], second[1] - second[0]])
            # 두 경계의 단위 방향: $$t_i=v_i/\|v_i\|$$
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            if abs(directions[0] @ directions[1]) < np.cos(np.deg2rad(8)):
                continue
            if directions[0] @ directions[1] < 0:
                directions[1] *= -1
            # 두 경계의 공통 길이 방향: $$t' = t_1+t_2$$
            axis = directions.sum(axis=0)
            # 공통 방향을 단위 길이로 정규화: $$t=t'/\|t'\|$$
            axis /= np.linalg.norm(axis)
            intervals = np.sort(np.array([first @ axis, second @ axis]), axis=1)
            overlap = min(intervals[:, 1]) - max(intervals[:, 0])
            if overlap < 0.025:
                continue
            # 법선과 길이 방향에 수직인 폭 방향: $$b'=n\times t$$
            width_axis = np.cross(normal, axis)
            # 폭 방향의 단위 벡터: $$b=b'/\|b'\|$$
            width_axis /= np.linalg.norm(width_axis)
            centers = np.array([first.mean(axis=0), second.mean(axis=0)])
            # 두 경계의 폭 방향 간격: $$w=|b^T(p_2-p_1)|$$
            width = abs(float((centers[1] - centers[0]) @ width_axis))
            if width < 0.005:
                continue
            score = overlap
            if axis_hint is not None:
                score *= 0.5 + 0.5 * abs(axis @ axis_hint)
            candidates.append((score, axis, width_axis, width, centers.mean(axis=0), np.array([first, second])))
    if not candidates:
        # 한 모서리에서 확인한 직선 방향: $$t'=p_1-p_0$$
        axis = lines[0][1] - lines[0][0]
        # 직선 방향을 단위 길이로 정규화: $$t=t'/\|t'\|$$
        axis /= np.linalg.norm(axis)
        reference = axis_hint if axis_hint is not None else np.eye(3)[np.argmax(np.abs(axis))]
        if axis @ reference < 0:
            axis = -axis
        # 한 모서리만 보여도 법선과 길이 방향으로 폭 방향은 정의된다: $$b=n\times t$$
        width_axis = np.cross(normal, axis)
        return {"status": "axis_only", "axis": axis, "width_axis": width_axis,
                "edge_lines_m": np.array([lines[0]])}
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, axis, width_axis, width, midpoint, edges = candidates[0]
    if axis_hint is None and any(item[0] > score * 0.8 and abs(item[1] @ axis) < 0.8 for item in candidates[1:]):
        return {"status": "ambiguous_edges"}
    reference = axis_hint if axis_hint is not None else np.eye(3)[np.argmax(np.abs(axis))]
    if axis @ reference < 0:
        axis, width_axis = -axis, -width_axis
    # 길이 방향의 임의 위치 성분을 제거: $$c=p_{mid}-t(t^Tp_{mid})$$
    centerline_point = midpoint - axis * (axis @ midpoint)
    return {"status": "two_edges", "axis": axis, "width_axis": width_axis,
            "width_m": width, "centerline_point_m": centerline_point, "edge_lines_m": edges}


def estimate_beam(depth_m: np.ndarray, profile: dict, *,
                  roi: tuple[int, int, int, int] | None = None,
                  depth_range_m: tuple[float, float] = (0.10, 0.70),
                  plane_threshold_m: float = 0.002, axis_hint: np.ndarray | None = None,
                  seed: int = 0) -> BeamEstimate:
    r"""원본 깊이의 가장 큰 평면 후보에서 빔 방향·폭·중심선을 추정한다.

    $$e_{RMS}=\sqrt{\frac{1}{N}\sum_i(n^Tp_i+d)^2}$$

    roi는 픽셀 (왼쪽, 위, 오른쪽, 아래)이며 오른쪽·아래는 포함하지 않는다.
    axis_hint는 후보 선택과 부호에만 쓰고 실제 경계를 그 방향으로 고정하지 않는다.
    """
    import cv2

    if not np.isfinite(plane_threshold_m) or plane_threshold_m <= 0:
        raise ValueError("평면 허용 거리는 유한한 양수여야 합니다.")
    if not np.all(np.isfinite(depth_range_m)) or not 0 < depth_range_m[0] < depth_range_m[1]:
        raise ValueError("깊이 범위는 양수이며 가까운 거리부터 지정해야 합니다.")
    if axis_hint is not None:
        axis_hint = np.asarray(axis_hint, dtype=float)
        if axis_hint.shape != (3,) or not np.all(np.isfinite(axis_hint)) or np.linalg.norm(axis_hint) < 1e-8:
            raise ValueError("방향 참고값은 길이가 있는 3차원 벡터여야 합니다.")
        # 참고 방향의 단위 벡터: $$t_h=t'_h/\|t'_h\|$$
        axis_hint = axis_hint / np.linalg.norm(axis_hint)
    depth = rectify_image(np.asarray(depth_m, dtype=np.float32), profile, depth=True)
    height, width = depth.shape
    bounds = (0, 0, width, height) if roi is None else roi
    x0, y0, x1, y1 = bounds
    if not all(isinstance(value, (int, np.integer)) for value in bounds) or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("ROI는 영상 안의 정수 픽셀 범위여야 합니다.")
    region = np.zeros(depth.shape, dtype=bool)
    region[y0:y1, x0:x1] = True
    valid = region & np.isfinite(depth) & (depth >= depth_range_m[0]) & (depth <= depth_range_m[1])
    if valid.sum() < 200:
        raise BeamDetectionError("평면 추정에 필요한 유효 깊이가 200픽셀보다 적습니다.")
    cloud = depth_points(depth, profile)
    normal, offset = _ransac_plane(cloud[valid], plane_threshold_m, seed)
    # 모든 영상점의 평면 잔차: $$e(u,v)=n^Tp(u,v)+d$$
    residual = cloud @ normal + offset
    mask = valid & (np.abs(residual) <= plane_threshold_m)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = labels == largest
    if mask.sum() < 200:
        raise BeamDetectionError("연속된 평면 영역이 너무 작습니다.")
    normal, offset = _least_squares_plane(cloud[mask])
    # 최종 평면의 수직 거리 잔차: $$e_i=n^Tp_i+d$$
    residual = cloud[mask] @ normal + offset
    # 관측 점들의 평면 적합 오차: $$e_{RMS}=\sqrt{\operatorname{mean}(e_i^2)}$$
    rms = float(np.sqrt(np.mean(residual**2)))
    lines = _boundary_lines(mask, profile, normal, offset, bounds)
    geometry = _edge_geometry(lines, normal, axis_hint)
    return BeamEstimate(normal, offset, rms, float(mask.sum() / valid.sum()), mask, **geometry)
