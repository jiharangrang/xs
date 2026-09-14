"""카메라 영상의 왜곡을 보정하고 픽셀과 카메라 기준 3차원 좌표를 변환한다.
입출력 길이는 미터이며 촬영·모터 통신·파일 저장에는 관여하지 않는다.
"""

import numpy as np


def pixel_rays(pixels: np.ndarray, profile: dict) -> np.ndarray:
    r"""왜곡 없는 픽셀에서 광학 깊이가 1인 카메라 광선을 만든다.

    $$r(u,v)=((u-c_x)/f_x,\ (v-c_y)/f_y,\ 1)$$

    u와 v는 픽셀 좌표, f와 c는 초점거리와 주점이다.
    """
    pixels = np.asarray(pixels, dtype=float)
    lens = profile["intrinsics"]
    values = [lens[key] for key in ("fx", "fy", "cx", "cy")]
    if pixels.shape[-1] != 2 or not np.all(np.isfinite(values)) or min(values[:2]) <= 0:
        raise ValueError("픽셀 좌표와 렌즈 설정을 확인하세요.")
    rays = np.ones((*pixels.shape[:-1], 3))
    # 가로 광선 성분: $$r_x=(u-c_x)/f_x$$
    rays[..., 0] = (pixels[..., 0] - lens["cx"]) / lens["fx"]
    # 세로 광선 성분: $$r_y=(v-c_y)/f_y$$
    rays[..., 1] = (pixels[..., 1] - lens["cy"]) / lens["fy"]
    return rays


def depth_points(depth_m: np.ndarray, profile: dict) -> np.ndarray:
    r"""왜곡 없는 깊이를 영상과 같은 배열 구조의 3차원 점으로 바꾼다.

    $$p(u,v)=z(u,v)r(u,v)$$

    z는 광학축 방향 거리이며 무효 깊이는 세 좌표 모두 NaN으로 만든다.
    """
    depth = np.asarray(depth_m, dtype=float)
    if depth.shape != (profile["height"], profile["width"]):
        raise ValueError("깊이와 카메라 프로필의 해상도가 다릅니다.")
    rows, columns = np.indices(depth.shape)
    rays = pixel_rays(np.stack((columns, rows), axis=-1), profile)
    # 픽셀 광선에 광학 깊이를 곱한다: $$p=zr$$
    points = depth[..., None] * rays
    points[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return points


def plane_points(pixels: np.ndarray, profile: dict, normal: np.ndarray, offset_m: float) -> np.ndarray:
    r"""픽셀 광선과 추정 평면의 교점을 구한다.

    $$p=-\frac{d}{n^Tr}r$$

    n과 d는 평면 n^T p+d=0의 단위 법선과 상수이며, 카메라 뒤의 교점은 무효다.
    """
    rays = pixel_rays(pixels, profile)
    # 광선의 법선 방향 성분: $$s=n^Tr$$
    denominator = rays @ normal
    depth = np.full_like(denominator, np.nan)
    np.divide(-offset_m, denominator, out=depth, where=np.abs(denominator) > 1e-8)
    depth[depth <= 0] = np.nan
    # 교점의 카메라 좌표: $$p=zr$$
    return rays * depth[..., None]


def project_points(points: np.ndarray, profile: dict) -> np.ndarray:
    r"""카메라 앞의 3차원 점을 왜곡 없는 영상에 투영한다.

    $$(u,v)=(f_x x/z+c_x,\ f_y y/z+c_y)$$

    카메라 뒤나 광학 중심의 점은 NaN으로 표시한다.
    """
    points = np.asarray(points, dtype=float)
    lens = profile["intrinsics"]
    output = np.full((*points.shape[:-1], 2), np.nan)
    valid = np.all(np.isfinite(points), axis=-1) & (points[..., 2] > 1e-8)
    # 광학 깊이로 정규화한 영상 좌표: $$(x_n,y_n)=(x/z,y/z)$$
    normalized = points[valid, :2] / points[valid, 2, None]
    # 픽셀 초점거리를 적용: $$(u_0,v_0)=(f_xx_n,f_yy_n)$$
    centered = normalized * [lens["fx"], lens["fy"]]
    # 영상의 주점을 더한다: $$(u,v)=(u_0+c_x,v_0+c_y)$$
    output[valid] = centered + [lens["cx"], lens["cy"]]
    return output


def rectify_image(image: np.ndarray, profile: dict, *, depth: bool = False) -> np.ndarray:
    """실제 RGB 또는 깊이를 같은 렌즈 행렬의 왜곡 없는 영상으로 바꾼다."""
    import cv2

    if image.shape[:2] != (profile["height"], profile["width"]):
        raise ValueError("실제 영상과 카메라 프로필의 해상도가 다릅니다.")
    distortion = profile.get("distortion", {})
    coefficients = np.array([distortion.get(key, 0.0) for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")])
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("렌즈 왜곡값은 유한한 값이어야 합니다.")
    if not np.any(coefficients):
        return image.copy()
    if "BROWN_CONRADY" not in profile.get("distortion_model", ""):
        raise ValueError("이 비교는 Brown-Conrady 렌즈 왜곡만 지원합니다.")
    lens = profile["intrinsics"]
    matrix = np.array([[lens["fx"], 0, lens["cx"]], [0, lens["fy"], lens["cy"]], [0, 0, 1]])
    maps = cv2.initUndistortRectifyMap(matrix, coefficients, None, matrix,
                                     (profile["width"], profile["height"]), cv2.CV_32FC1)
    return cv2.remap(image, *maps, interpolation=cv2.INTER_NEAREST if depth else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan") if depth else 0)
