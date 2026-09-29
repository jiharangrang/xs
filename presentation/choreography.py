"""실측 관절각 대신 이상적인 팁 경로와 설명용 처짐으로 발표 자세를 만든다.
보정의 이동·대기 시각과 원본 단계 결과는 녹화 기록을 따른다.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from kinematics.anchoring import TipAnchor
from kinematics.ik import IKSettings, InverseKinematics
from simulation.model import apply_anchor, set_arm_angles


@dataclass(frozen=True)
class PresentationSettings:
    """실측값이 아닌 발표 연출의 크기를 지정한다."""

    sag_mm: float = 15.0
    tilt_deg: float = 8.0
    side_clearance_mm: float = 40.0
    rear_clearance_mm: float = 25.0

    def __post_init__(self):
        """표현 가능한 범위의 유한한 설정만 허용한다."""
        bounds = {'sag_mm': (0, 40), 'tilt_deg': (0, 15), 'side_clearance_mm': (25, 60), 'rear_clearance_mm': (10, 40)}
        for name, (lower, upper) in bounds.items():
            if not np.isfinite(getattr(self, name)) or not lower <= getattr(self, name) <= upper:
                raise ValueError(f'{name}은 {lower} 이상 {upper} 이하이어야 합니다.')


def blend(a, b, fraction):
    r"""두 값 사이를 일정 비율로 보간한다.

    $$x=(1-u)a+ub$$
    """
    # 두 끝값 사이의 보간: $$x=(1-u)a+ub$$
    return (1 - fraction) * a + fraction * b


def pose_between(a, b, fraction):
    r"""팁 위치는 직선으로, 방향은 회전 구면상에서 보간한다.

    $$T(u)=\begin{bmatrix}(\operatorname{Slerp}(R_a,R_b,u))&(1-u)p_a+up_b\\0&1\end{bmatrix}$$
    """
    result = np.eye(4)
    result[:3, 3] = blend(a[:3, 3], b[:3, 3], fraction)
    result[:3, :3] = Slerp([0, 1], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))([fraction]).as_matrix()[0]
    return result


@dataclass
class ArmMotion:
    """실제 시간 구간과 이에 대응하는 표시용 관절 경로를 보관한다."""

    start: float
    end: float
    anchor: TipAnchor
    q: np.ndarray

    def sample(self, seconds):
        r"""시간에 해당하는 두 경로 표본 사이를 보간한다.

        $$v=\operatorname{clip}((t-t_0)/(t_1-t_0),0,1)(N-1)$$
        """
        # 경로 표본의 연속 인덱스: $$v=\operatorname{clip}((t-t_0)/(t_1-t_0),0,1)(N-1)$$
        value = np.clip((seconds - self.start) / (self.end - self.start), 0, 1) * (len(self.q) - 1)
        lower = min(int(value), len(self.q) - 2)
        return blend(self.q[lower], self.q[lower + 1], value - lower)


class Choreography:
    """고정 발의 월드 위치를 유지하며 단계마다 이상적인 도착 자세를 생성한다."""

    def __init__(self, recording, settings=None):
        """당시 모델·이론 자세와 이벤트 시간으로 전체 표시 경로를 미리 계산한다."""
        self.recording = recording
        self.settings = settings or PresentationSettings()
        self.solver = InverseKinematics(recording.model_path, IKSettings(starts=1), calibration_path=recording.calibration_path)
        self.q = np.asarray(recording.observation_pose['q_grasp_rad'])
        grasp = self.solver.fk.forward(self.q)
        self.left = grasp.T_world_tip_L.copy()
        self.right = grasp.T_world_tip_R.copy()
        self.grasp_right = self.right.copy()
        self.motions = []
        self.max_ik_error_m = 0.0
        self._prepare()
        self._walking()
        self.motions.sort(key=lambda motion: motion.start)
        self.starts = np.array([motion.start for motion in self.motions])
        self.grips = self._gripper_tracks()

    def _move(self, start, end, fixed, target):
        r"""이상적인 팁의 직선 경로를 작은 간격으로 나눠 역기구학을 계산한다.

        $$q_k=\operatorname{IK}(T_{F}^{-1}T_k,q_{k-1})$$
        """
        if end <= start:
            raise ValueError('이동 구간의 종료 시각이 시작보다 늦어야 합니다.')
        current = self.right if fixed == 'tip_L' else self.left
        anchor = TipAnchor(fixed, self.left if fixed == 'tip_L' else self.right)
        # 이동 거리: $$d=\|p_1-p_0\|$$
        distance = np.linalg.norm(target[:3, 3] - current[:3, 3])
        # 회전 변화량: $$\theta=\|\operatorname{Log}(R_1R_0^T)^\vee\|$$
        angle = np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec())
        count = max(2, int(np.ceil(distance / .002)), int(np.ceil(angle / np.deg2rad(1))))
        path = [self.q.copy()]
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            goal = pose_between(current, target, fraction)
            result = self.solver.solve(anchor.to_relative_target(goal), self.q)
            if not result.success:
                raise ValueError(f'{start:.3f}초 발표 경로 IK 실패: {result.message}')
            self.q = result.q_rad.copy()
            self.max_ik_error_m = max(self.max_ik_error_m, result.position_error_m)
            path.append(self.q.copy())
        self.motions.append(ArmMotion(start, end, anchor, np.array(path)))
        if fixed == 'tip_L':
            self.right = target.copy()
        else:
            self.left = target.copy()

    def _correction(self, run, target):
        """실제 MOVING 구간에만 이동을 분배하고 관측 중에는 자세를 유지한다."""
        rows = self.recording.rows[run['run_id']]
        windows = []
        for index, row in enumerate(rows):
            if row.get('state') != 'MOVING':
                continue
            next_row = next((r for r in rows[index + 1:] if r.get('state') != 'MOVING'), rows[-1])
            windows.append((self.recording.relative(row['monotonic_s']), self.recording.relative(next_row['monotonic_s'])))
        if not windows:
            raise ValueError(f'{run["stage"]}단계에 이동 구간이 없습니다.')
        initial = self.right.copy()
        for index, (start, end) in enumerate(windows):
            self._move(start, end, 'tip_L', pose_between(initial, target, (index + 1) / len(windows)))

    def _prepare(self):
        r"""관측 시작·정면 보정·상승·외측 이동·삽입을 실제 시각에 배치한다.

        $$q_j(u)=\operatorname{clip}(u/d_j,0,1)q_{j,goal}$$
        """
        rec = self.recording
        settings = self.settings
        sagged = self.grasp_right.copy()
        # 관측 높이에 설명용 처짐을 추가: $$z_s=z_g-h_{obs}-h_{sag}$$
        sagged[2, 3] -= rec.observation_pose['drop_m'] + settings.sag_mm / 1000
        # 설명용 기울기를 팁 방향에 적용: $$R_s=R_y(\theta)R_g$$
        sagged[:3, :3] = Rotation.from_euler('y', settings.tilt_deg, degrees=True).as_matrix() @ sagged[:3, :3]
        anchor = TipAnchor('tip_L', self.left)
        result = self.solver.solve(anchor.to_relative_target(sagged), np.asarray(rec.observation_pose['q_rad']))
        if not result.success:
            raise ValueError('처짐을 표현할 시작 자세를 구하지 못했습니다.')
        stage1 = rec.stage1
        fractions = np.linspace(0, 1, 100)
        # 각 관절의 목표 이동 크기: $$a_j=|q_{j,goal}|$$
        magnitudes = np.abs(result.q_rad)
        # 같은 속도로 움직일 때의 상대 도착 시간: $$d_j=\max(0.05,a_j/\max_k a_k)$$
        durations = np.maximum(.05, magnitudes / np.max(magnitudes))
        # 각 관절의 진행 비율: $$u_j=\operatorname{clip}(u/d_j,0,1)$$
        progress = np.clip(fractions[:, None] / durations, 0, 1)
        # 이론적인 관절 목표까지 전개: $$q_j=u_jq_{j,goal}$$
        unfolding = progress * result.q_rad
        self.motions.append(ArmMotion(rec.relative(stage1['started_at_s']), rec.relative(stage1['arrival']['observed_at_s']), anchor, unfolding))
        self.q = result.q_rad.copy()
        self.right = sagged
        for run in rec.runs:
            stage = run['stage']
            if stage > 6:
                continue
            target = self.grasp_right.copy()
            if stage == 2:
                target[2, 3] -= rec.observation_pose['drop_m']
            elif stage == 3:
                target[2, 3] -= .025
            elif stage == 4:
                target[2, 3] -= .025
                target[1, 3] += settings.side_clearance_mm / 1000
            elif stage == 5:
                target[1, 3] += settings.side_clearance_mm / 1000
            self._correction(run, target)

    def _walking(self):
        r"""뒷발 당김과 앞발 전진의 실제 하위 단계 시간에 이상적인 이동을 배치한다.

        $$p_{goal}=p_{start}+d e_x$$
        """
        for run in self.recording.runs:
            if run['stage'] < 7:
                continue
            phases = {row['phase']: self.recording.relative(row['monotonic_s']) for row in run['phases']}
            # 기록된 진행량을 미터로 변환: $$d=d_{mm}/1000$$
            distance = run['distance_mm'] / 1000
            if run['stage'] == 7:
                target = self.left.copy()
                target[1, 3] -= self.settings.rear_clearance_mm / 1000
                exit_index = next(index for index, phase in enumerate(run['phases']) if phase['phase'] == 'side_exit')
                exit_end = self.recording.relative(run['phases'][exit_index + 1]['monotonic_s'])
                self._move(phases['side_exit'], exit_end, 'tip_R', target)
                # 빔 길이 방향으로 목표 위치 전진: $$x_{goal}=x_{start}+d$$
                target[0, 3] += distance
                self._move(phases['pulling'], phases['pull_arrival'], 'tip_R', target)
                target[1, 3] += self.settings.rear_clearance_mm / 1000
                self._move(phases['side_return'], phases['side_return_arrival'], 'tip_R', target)
            else:
                target = self.right.copy()
                # 빔 길이 방향으로 목표 위치 전진: $$x_{goal}=x_{start}+d$$
                target[0, 3] += distance
                rows = self.recording.rows[run['run_id']]
                active = [row for row in rows if row.get('phase') == 'advancing' and row.get('state') == 'MOVING']
                start = self.recording.relative(active[0]['monotonic_s']) if active else phases['advancing']
                initial = self.right.copy()
                for index, row in enumerate(active):
                    row_index = rows.index(row)
                    stop = next(r for r in rows[row_index + 1:] if r.get('state') != 'MOVING')
                    self._move(self.recording.relative(row['monotonic_s']), self.recording.relative(stop['monotonic_s']),
                               'tip_L', pose_between(initial, target, (index + 1) / len(active)))

    def _gripper_tracks(self):
        r"""실제 개폐 명령 시각·속도를 모델의 열림과 접촉 자세에 대응시킨다.

        $$T=|g_{goal}-g_{start}|/v_g$$

        자동 단계의 실제 종료 이벤트가 있으면 위 예상 시간 대신 그 시각을 사용한다.
        """
        tracks = {'G_L': [(self.recording.start, 0.0)], 'G_R': [(self.recording.start, -120.0)]}
        for event in self.recording.sync:
            name = event['joint']
            if name not in tracks or not event['angle_deg'] or event['detail'] != 'move_many':
                continue
            start = float(event['relative_walk_s'])
            old = float(np.interp(start, *np.array(tracks[name]).T))
            goal = -120.0 if float(event['angle_deg']) < -60 else 0.0
            speed = float(event['speed_deg_s'])
            if speed <= 0:
                raise ValueError('그리퍼 명령 속도가 유효하지 않습니다.')
            # 수동 개폐의 예상 소요 시간: $$T=|g_{goal}-g_{start}|/v_g$$
            duration = abs(goal - old) / speed
            for run in self.recording.runs:
                phases = run['phases']
                for index, phase in enumerate(phases[:-1]):
                    phase_start = self.recording.relative(phase['monotonic_s'])
                    if phase['phase'] in ('opening', 'closing') and abs(phase_start - start) < .02:
                        duration = self.recording.relative(phases[index + 1]['monotonic_s']) - start
            tracks[name] = [point for point in tracks[name] if point[0] < start]
            tracks[name].extend([(start, old), (start + duration, goal)])
        return tracks

    def sample(self, seconds):
        """임의 시각을 조회해도 같은 관절각·고정점·그리퍼 자세를 반환한다."""
        if not np.isfinite(seconds):
            raise ValueError('재생 시각은 유한한 값이어야 합니다.')
        index = max(0, int(np.searchsorted(self.starts, seconds, side='right') - 1))
        motion = self.motions[index]
        q = motion.sample(seconds)
        grips = {name: np.deg2rad(np.interp(seconds, *np.array(points).T)) for name, points in self.grips.items()}
        return q, motion.anchor, grips

    def apply(self, model, data, seconds):
        """재구성 자세만 표시용 모델에 반영하고 물리 시간을 진행하지 않는다."""
        q, anchor, grips = self.sample(seconds)
        for name, angle in grips.items():
            data.joint(name).qpos[0] = angle
        set_arm_angles(model, data, q)
        apply_anchor(model, data, anchor)
