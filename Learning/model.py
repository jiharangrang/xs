"""논문의 링크 SE(3) 입력과 네 개의 128차원 은닉층으로 충돌거리를 학습한다.
배치 순기구학은 PyTorch로 계산해 관절각에 대한 미분도 유지한다.
"""

import mujoco
import numpy as np
import torch
from torch import nn

from common import BODIES, JOINTS


def pose_matrix(position, quaternion):
    r"""MuJoCo 위치와 쿼터니언을 동차변환으로 변환한다.

    $$T=\begin{bmatrix}R&p\\0&1\end{bmatrix}$$
    """
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    transform = np.eye(4)
    transform[:3, :3] = rotation.reshape(3, 3)
    transform[:3, 3] = position
    return transform


class LinkFeatures(nn.Module):
    """XML의 몸체·회전축·회전 중심을 보존하는 배치 순기구학이다."""

    def __init__(self, model):
        """MuJoCo에서 고정 변환과 회전관절 정보를 읽어 미분 가능한 상수로 보관한다."""
        super().__init__()
        local = [pose_matrix(model.body_pos[i], model.body_quat[i]) for i in range(model.nbody)]
        self.parents = model.body_parentid.tolist()
        self.body_ids = [model.body(name).id for name in BODIES]
        self.moving = {}
        self.register_buffer("local", torch.tensor(np.stack(local), dtype=torch.float32))
        beam = pose_matrix(model.body("ibeam").pos, model.body("ibeam").quat)
        self.register_buffer("beam_inverse", torch.tensor(np.linalg.inv(beam), dtype=torch.float32))
        for index, name in enumerate(JOINTS):
            joint = model.joint(name).id
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("이 입력 표현은 현재 XS의 회전관절만 지원합니다.")
            body = int(model.jnt_bodyid[joint])
            if model.body_jntnum[body] != 1:
                raise ValueError("한 몸체의 다중 관절은 지원하지 않습니다.")
            x, y, z = model.jnt_axis[joint]
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            self.register_buffer(f"skew_{index}", torch.tensor(skew, dtype=torch.float32))
            self.register_buffer(f"pivot_{index}", torch.tensor(model.jnt_pos[joint], dtype=torch.float32))
            self.moving[body] = (index, float(model.qpos0[model.jnt_qposadr[joint]]))
        self.output_dim = 12 * len(self.body_ids)

    def forward(self, q):
        r"""관절각을 빔 기준 링크 변환의 행 우선 벡터로 바꾼다.

        $$g(q)=\operatorname{concat}_{i}\operatorname{vec}_{row}(T_{Bi}(q)_{1:3,1:4})$$

        각 링크의 회전행렬 아홉 값과 위치 세 값을 논문 식 (8)의 순서로 사용한다.
        """
        if q.ndim != 2 or q.shape[1] != len(JOINTS):
            raise ValueError("입력 배열은 표본 수 × 관절 여덟 개여야 합니다.")
        batch = len(q)
        identity = torch.eye(3, dtype=q.dtype, device=q.device)
        transforms = [torch.eye(4, dtype=q.dtype, device=q.device).expand(batch, -1, -1)]
        for body in range(1, len(self.parents)):
            local = self.local[body].expand(batch, -1, -1)
            if body in self.moving:
                index, reference = self.moving[body]
                # 기준각으로부터의 회전량: $$\alpha=q_j-q_{j,ref}$$
                angle = q[:, index] - reference
                skew = getattr(self, f"skew_{index}")
                pivot = getattr(self, f"pivot_{index}")
                # 회전축의 Rodrigues 공식: $$R=I+\sin(\alpha)K+(1-\cos(\alpha))K^2$$
                rotation = identity + angle.sin()[:, None, None] * skew + (1 - angle.cos())[:, None, None] * (skew @ skew)
                # 회전 중심을 고정하는 병진: $$t=c-Rc$$
                translation = pivot - rotation @ pivot
                upper = torch.cat((rotation, translation[:, :, None]), dim=2)
                bottom = q.new_tensor([0, 0, 0, 1]).expand(batch, 1, 4)
                motion = torch.cat((upper, bottom), dim=1)
                # 몸체의 기준 배치에 관절 회전을 적용: $$T_{local}=T_{rest}T_{joint}$$
                local = local @ motion
            # 부모에서 현재 몸체로 이어지는 순기구학: $$T_{Wi}=T_{Wp}T_{pi}$$
            transforms.append(transforms[self.parents[body]] @ local)
        frames = torch.stack([transforms[i] for i in self.body_ids], dim=1)
        # 모든 링크를 빔 기준으로 표현: $$T_{Bi}=T_{BW}T_{Wi}$$
        relative = self.beam_inverse @ frames
        return relative[:, :, :3, :].reshape(batch, self.output_dim)


class SE3NN(nn.Module):
    """논문 §3.2의 은닉층 네 개와 선형 거리 출력 한 개를 사용한다."""

    def __init__(self, model):
        """학습 장면의 순기구학과 128차원 ReLU 은닉층을 구성한다."""
        super().__init__()
        self.features = LinkFeatures(model)
        layers = []
        width = self.features.output_dim
        for _ in range(4):
            layers.extend((nn.Linear(width, 128), nn.ReLU()))
            width = 128
        layers.append(nn.Linear(128, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, q):
        r"""관절각에서 링크 표현을 거쳐 부호 있는 충돌거리를 예측한다.

        $$\hat d_\theta(q)=NN_\theta(g(q))$$
        """
        return self.network(self.features(q)).squeeze(-1)


def load_checkpoint(path, scene):
    """같은 장면에서 학습된 가중치만 CPU로 읽어 평가 모드로 반환한다."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["scene_id"] != scene.scene_id or checkpoint["joint_names"] != list(JOINTS):
        raise ValueError("학습 장면 또는 관절 순서가 체크포인트와 다릅니다.")
    model = SE3NN(scene.model)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
