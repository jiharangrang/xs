"""삼각형 메시를 반공간으로 잘라 경계와 교차하는 면의 형상을 보존한다.
높이와 빔 폭 안의 간격을 계산할 때 꼭짓점만 검사해서 빠지는 영역을 방지한다.
"""

import numpy as np


def clip_triangles(triangles, normal, limit):
    r"""평면의 음의 반공간에 남는 삼각형과 교차점을 반환한다.

    $$s(p)=n^Tp-b,\quad p_*=p_a+\frac{s_a}{s_a-s_b}(p_b-p_a)$$
    """
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    if not len(triangles):
        return triangles.copy()
    # 절단 평면에 대한 꼭짓점의 부호 있는 위치: $$s_i=n^Tp_i-b$$
    distances = triangles @ normal - limit
    inside = distances <= 0
    output = [triangles[inside.all(axis=1)]]
    for triangle, values in zip(triangles[inside.any(axis=1) & ~inside.all(axis=1)],
                                distances[inside.any(axis=1) & ~inside.all(axis=1)], strict=True):
        polygon = []
        for index in range(3):
            previous = (index - 1) % 3
            if (values[previous] <= 0) != (values[index] <= 0):
                # 모서리와 절단 평면의 교차 비율: $$t=s_a/(s_a-s_b)$$
                fraction = values[previous] / (values[previous] - values[index])
                # 실제 삼각형 모서리 위 교차점: $$p_*=p_a+t(p_b-p_a)$$
                crossing = triangle[previous] + fraction * (triangle[index] - triangle[previous])
                polygon.append(crossing)
            if values[index] <= 0:
                polygon.append(triangle[index])
        output.extend(np.array([[polygon[0], polygon[i], polygon[i + 1]]]) for i in range(1, len(polygon) - 1))
    return np.concatenate(output, axis=0)


def strip_vertices(triangles, axis, lower, upper):
    """두 평행 경계 사이에 겹치는 메시의 꼭짓점과 교차점을 반환한다."""
    clipped = clip_triangles(triangles, axis, upper)
    clipped = clip_triangles(clipped, -np.asarray(axis), -lower)
    return clipped.reshape(-1, 3)
