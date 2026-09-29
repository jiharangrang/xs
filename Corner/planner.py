"""고정턱의 옆 이탈과 재삽입을 이용해 양쪽 그리퍼가 ㄱ자 빔을 건너게 한다.
고정단을 교대하며 기존 FK·IK와 관절 제한만 읽어 사용한다.
"""

from dataclasses import dataclass
import json

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

from kinematics.anchoring import TipAnchor
from kinematics.ik import IKSettings, InverseKinematics
from kinematics.poses import pose_error

from Corner.scene import OUTPUT, Scene, site_pose


@dataclass(frozen=True)
class Settings:
    """모서리 위치와 옆 이탈·하강 거리 및 기구학 영상의 속도를 지정한다."""

    corner_x: float = .16
    initial_span: float = .10
    front_y: float = .19
    rear_y: float = .08
    sideways: float = .035
    rear_exit_x: float = .045
    drop: float = .04
    grasp_z: float = .241830662
    open_deg: float = -120.
    translation_step: float = .003
    rotation_step_deg: float = 2.
    speed: float = .55


def interpolate_pose(start, end, fraction):
    r"""두 팁 자세 사이의 위치와 최단 회전을 보간한다.

    $$p(s)=(1-s)p_0+sp_1,\quad R(s)=\operatorname{Slerp}(R_0,R_1;s)$$
    """
    pose = np.eye(4)
    # 위치를 구간 비율로 보간: $$p(s)=(1-s)p_0+sp_1$$
    pose[:3, 3] = (1 - fraction) * start[:3, 3] + fraction * end[:3, 3]
    # 최단 회전으로 방향 보간: $$R(s)=\operatorname{Slerp}(R_0,R_1;s)$$
    pose[:3, :3] = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], end[:3, :3]]))([fraction]).as_matrix()[0]
    return pose


def pose_at(x, y, z, yaw):
    """수평 그리퍼의 위치와 수직축 회전으로 목표 자세를 구성한다."""
    pose = np.eye(4)
    pose[:3, 3] = [x, y, z]
    pose[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
    return pose


class Planner:
    """한쪽 팁을 고정한 연속 IK 경로와 그리퍼 개폐 순서를 생성한다."""

    def __init__(self, scene: Scene, settings=Settings()):
        """영상 경로 계산에 필요한 모델과 작은 이동용 솔버를 준비한다."""
        self.scene = scene
        self.settings = settings
        self.solver = InverseKinematics(settings=IKSettings(starts=1, max_iterations=150))
        self.q = np.deg2rad([-37.3, -22.3, 0, 135.4, 0, 22.3, 37.3])
        self.grippers = np.zeros(2)
        self.anchor = TipAnchor("tip_L", pose_at(0, 0, settings.grasp_z, 180))
        self.samples = []
        self.phases = []

    def append(self, phase, *, hold=.0):
        r"""현재 상태를 저장하고 최대 관절 속도에 맞춰 재생 시각을 증가시킨다.

        $$\Delta t=\max(\Delta t_{hold},1.5\|q_k-q_{k-1}\|_\infty/v_{max})$$

        개폐 관절도 속도 계산에 포함하며 보간의 최대 속도를 고려한다.
        """
        self.scene.set(self.q, self.grippers, self.anchor)
        time = 0.
        if self.samples:
            previous = self.samples[-1]
            # 몸통과 개폐 관절의 최대 변화량: $$\delta=\|q_k-q_{k-1}\|_\infty$$
            delta = np.max(np.abs(np.r_[self.q, self.grippers] - np.r_[previous["q"], previous["grippers"]]))
            # 매끄러운 보간의 속도 상한을 고려한 구간 시간: $$\Delta t=\max(\Delta t_{hold},1.5\delta/v_{max})$$
            duration = max(hold, 1.5 * delta / self.settings.speed, .035)
            time = previous["time"] + duration
        self.samples.append({"time": time, "q": self.q.copy(), "grippers": self.grippers.copy(),
                             "fixed_tip": self.anchor.fixed_tip, "anchor": self.anchor.T_world_fixed_tip.copy(),
                             "phase": phase, "L": site_pose(self.scene.data, "tip_L"),
                             "R": site_pose(self.scene.data, "tip_R")})

    def solve(self, target):
        r"""직전 관절각과 가까운 IK 해를 구하며 두 비틀림 관절의 누적 회전을 줄인다.

        $$\min_q\frac12\|q-q_{prev}\|^2+\frac{0.01}{2}(q_3^2+q_5^2),\quad e(q)=0$$

        위치·방향과 기존 관절 제한을 만족하는 국소 해를 사용한다.
        """
        previous = self.q.copy()
        weights = np.array([0., 0., .01, 0., .01, 0., 0.])

        def objective(q):
            r"""관절 연속성과 비틀림을 함께 평가한다.

            $$C=\tfrac12\|q-q_{prev}\|^2+\tfrac12 q^TWq$$
            """
            # 직전 관절각과의 차이: $$\delta=q-q_{prev}$$
            delta = q - previous
            # 이동량과 비틀림의 가중 비용: $$C=\tfrac12\delta^T\delta+\tfrac12 q^TWq$$
            return .5 * np.sum(delta * delta) + .5 * np.sum(weights * q * q)

        def gradient(q):
            r"""연속성 비용의 관절 기울기를 반환한다.

            $$\nabla C=q-q_{prev}+Wq$$
            """
            # 관절별 비용의 미분: $$\nabla C=q-q_{prev}+Wq$$
            return q - previous + weights * q

        def constraint(q):
            r"""목표 상대 자세의 위치·방향 잔차를 계산한다.

            $$e=[(p-p_d)/0.1;\log(R_d^TR)^\vee]$$
            """
            error = pose_error(self.solver.fk.forward(q).T_tip_L_tip_R, target)
            # 위치 잔차의 길이 척도 정규화: $$e_p=(p-p_d)/0.1$$
            error[:3] /= .1
            return error

        result = minimize(objective, previous, jac=gradient, method="SLSQP",
                          bounds=self.solver.fk.joint_limits,
                          constraints={"type": "eq", "fun": constraint},
                          options={"maxiter": 150, "ftol": 1e-10})
        if result.success and np.linalg.norm(constraint(result.x)) >= 1e-3:
            result.success = False
        return result

    def move(self, goal, label):
        r"""이전 해에서 이어지는 작은 팁 이동들을 계산해 연속 관절 경로에 추가한다.

        $$N=\left\lceil\max(\|p_1-p_0\|/h_p,\theta/h_R)\right\rceil$$
        """
        moving = "tip_R" if self.anchor.fixed_tip == "tip_L" else "tip_L"
        start = site_pose(self.scene.data, moving)
        # 목표까지의 이동 길이: $$d=\|p_1-p_0\|$$
        distance = np.linalg.norm(goal[:3, 3] - start[:3, 3])
        # 목표까지의 회전각: $$\theta=\|\log(R_0^TR_1)^\vee\|$$
        angle = Rotation.from_matrix(start[:3, :3].T @ goal[:3, :3]).magnitude()
        # 위치·회전 간격을 모두 만족하는 표본 수: $$N=\lceil\max(d/h_p,\theta/h_R)\rceil$$
        count = max(1, int(np.ceil(max(distance / self.settings.translation_step,
                                     angle / np.deg2rad(self.settings.rotation_step_deg)))))
        phase = len(self.phases)
        self.phases.append(label)
        for index in range(1, count + 1):
            target = interpolate_pose(start, goal, index / count)
            result = self.solve(self.anchor.to_relative_target(target))
            if not result.success:
                raise RuntimeError(f"{label}: {index}/{count} 위치에서 연속 IK 실패")
            if np.max(np.abs(result.x - self.q)) > .2:
                raise RuntimeError(f"{label}: {index}/{count} 위치에서 연속 IK 관절 변화가 너무 큽니다.")
            self.q = result.x.copy()
            self.append(phase)
        print(f"경로: {label} / {count}개", flush=True)

    def grip(self, side, opened, label):
        """몸통을 유지한 채 지정한 개폐 관절을 부드럽게 움직인다."""
        index = 0 if side == "L" else 1
        goal = np.deg2rad(self.settings.open_deg) if opened else 0.
        phase = len(self.phases)
        self.phases.append(label)
        for value in np.linspace(self.grippers[index], goal, 36)[1:]:
            self.grippers[index] = value
            self.append(phase)

    def plan(self):
        """앞그리퍼 이동·고정단 교대·뒤그리퍼 이동으로 전체 로봇의 전환을 생성한다."""
        s = self.settings
        start = pose_at(s.initial_span, 0, s.grasp_z, 0)
        result = self.solve(self.anchor.to_relative_target(start))
        if not result.success:
            raise RuntimeError("시작 양쪽 파지 자세를 구하지 못했습니다.")
        self.q = result.x.copy()
        self.phases.append("시작 · A 빔 양쪽 파지")
        self.append(0)
        self.append(0, hold=1.5)
        self.grip("R", True, "앞그리퍼 열기 · 뒤 고정")
        self.move(pose_at(s.initial_span, s.sideways, s.grasp_z, 0), "앞 고정턱 옆 이탈")
        self.move(pose_at(s.initial_span, s.sideways, s.grasp_z - s.drop, 0), "앞그리퍼 하강")
        self.move(pose_at(s.corner_x - s.sideways, s.front_y, s.grasp_z - s.drop, 90), "빔 아래 우회 · 앞 방향 전환")
        self.move(pose_at(s.corner_x - s.sideways, s.front_y, s.grasp_z, 90), "B 빔 옆에서 앞 상승")
        self.move(pose_at(s.corner_x, s.front_y, s.grasp_z, 90), "앞그리퍼 B 빔 횡삽입")
        self.grip("R", False, "앞그리퍼 B 빔 파지")
        self.anchor = TipAnchor("tip_R", site_pose(self.scene.data, "tip_R"))
        self.append(len(self.phases) - 1, hold=1.)
        self.grip("L", True, "고정단 교대 · 뒤그리퍼 열기")
        self.move(pose_at(s.rear_exit_x, -s.sideways, s.grasp_z, 180), "뒤 고정턱 대각선 옆 이탈")
        self.move(pose_at(s.rear_exit_x, -s.sideways, s.grasp_z - s.drop, 180), "뒤그리퍼 하강")
        self.move(pose_at(s.corner_x, -s.sideways, s.grasp_z - s.drop, 180), "뒤그리퍼 빔 아래 전진")
        self.move(pose_at(s.corner_x + s.sideways, s.rear_y, s.grasp_z - s.drop, 270), "빔 아래 우회 · 뒤 방향 전환")
        self.move(pose_at(s.corner_x + s.sideways, s.rear_y, s.grasp_z, 270), "B 빔 옆에서 뒤 상승")
        self.move(pose_at(s.corner_x, s.rear_y, s.grasp_z, 270), "뒤그리퍼 B 빔 횡삽입")
        self.grip("L", False, "완료 · B 빔 양쪽 파지")
        self.append(len(self.phases) - 1, hold=2.)
        return self.samples

    def save(self, directory=OUTPUT):
        """기구학 경로와 단계 이름·설정을 폴더 내부에 저장한다."""
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {name: np.asarray([sample[name] for sample in self.samples])
                  for name in self.samples[0]}
        np.savez_compressed(directory / "trajectory.npz", **arrays)
        (directory / "trajectory.json").write_text(json.dumps(
            {"format": "xs.corner.v1", "scope": "kinematic_simulation_only", "phases": self.phases,
             "settings": vars(self.settings), "samples": len(self.samples),
             "duration_s": self.samples[-1]["time"]}, ensure_ascii=False, indent=2), encoding="utf-8")
