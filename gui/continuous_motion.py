"""관절 경로의 목표를 시간에 맞춰 연속 전송하고 중간 도착 대기를 생략한다.
출발·종료에는 시간 진행을 완만하게 바꾸며 늦어진 주기의 명령을 몰아서 보내지 않는다.
"""

import asyncio
from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class ContinuousMotionSettings:
    """목표 갱신 주기와 앞서 보낼 경로 시간 및 출발·종료 완화 시간을 지정한다."""

    period_s: float = .1
    lookahead_s: float = .3
    ramp_s: float = .5

    def __post_init__(self):
        """유한한 양수인 시간 설정을 검사한다."""
        if any(not np.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError("연속 이동의 시간 설정은 유한한 양수여야 합니다.")
        if self.lookahead_s > self.ramp_s:
            raise ValueError("목표 선행 시간은 출발·종료 완화 시간 이하여야 합니다.")


class ContinuousJointMotion:
    """현재 시각의 관절 목표를 보간해 전송하고 마지막 목표에서 실행을 넘긴다."""

    def __init__(self, *, settings=None, clock=time.monotonic, sleep=asyncio.sleep):
        """실물 통신과 무관한 시간 설정 및 시험용 시계를 연결한다."""
        self.settings = settings if settings is not None else ContinuousMotionSettings()
        self.clock, self.sleep = clock, sleep

    def path_time(self, elapsed_s, path_duration_s):
        r"""출발과 종료의 속도를 완만하게 바꾼 실행 시각을 원래 경로 시각으로 옮긴다.

        $$\tau(t)=\begin{cases}t^2/(2r)&t<r\\
        t-r/2&r\le t\le T-r\\
        T_0-(T-t)^2/(2r)&t>T-r\end{cases},\quad T=T_0+r$$

        T0는 원래 경로 시간, r은 그 이하로 제한한 완화 시간이다.
        """
        if not np.isfinite(path_duration_s) or path_duration_s <= 0 or not np.isfinite(elapsed_s):
            raise ValueError("경로 시간과 실행 시각을 확인해 주세요.")
        ramp = min(self.settings.ramp_s, path_duration_s)
        # 출발·종료 속도 완화에 필요한 실행 시간: $$T=T_0+r$$
        duration = path_duration_s + ramp
        elapsed = float(np.clip(elapsed_s, 0., duration))
        if elapsed < ramp:
            # 출발 시 경로 진행 속도를 선형으로 증가: $$\tau=t^2/(2r)$$
            return elapsed**2 / (2 * ramp)
        if elapsed > duration - ramp:
            # 종료 시 경로 진행 속도를 선형으로 감소: $$\tau=T_0-(T-t)^2/(2r)$$
            return path_duration_s - (duration - elapsed)**2 / (2 * ramp)
        # 중간 구간은 원래 시간 간격을 유지: $$\tau=t-r/2$$
        return elapsed - ramp / 2

    async def run(self, segment, send, progress):
        r"""중간 관절 도착을 기다리지 않고 예정 경로를 보간해 순서대로 전송한다.

        $$q_d(t)=\operatorname{interp}(\tau(t+\ell),t_k,q_k)$$

        send는 취소·소유권을 검사한 공통 모터 전송 함수이고 마지막 전송 결과를 반환한다.
        최종 실제 도착 확인은 호출자가 수행한다. progress는 명령한 경로 비율만 받는다.
        """
        started = self.clock()
        tick = 0
        while True:
            # 이번 실행에서 경과한 단조 시각: $$t=\max(0,t_{now}-t_{start})$$
            elapsed = max(0., self.clock() - started)
            # 모터가 중간 목표에 도착해 멈추기 전에 다음 경로를 지시: $$t_d=t+\ell$$
            target_elapsed = elapsed + self.settings.lookahead_s
            path_time = self.path_time(target_elapsed, segment.duration_s)
            # 같은 경로 시각으로 모든 관절 목표를 함께 보간: $$q_{d,j}=\operatorname{interp}(\tau,t_k,q_{k,j})$$
            q_rad = np.array([np.interp(path_time, segment.time_s, values) for values in segment.q_rad.T])
            receipt = await send(q_rad)
            # 경로 지점 수 기준의 누적 전송 비율: $$s=\operatorname{interp}(\tau,t_k,k/K)$$
            fraction = float(np.interp(path_time, segment.time_s, np.linspace(0., 1., len(segment.time_s))))
            progress(fraction)
            if path_time >= segment.duration_s:
                return receipt
            # 부동소수점 경계에서도 주기를 전진시키고 지연된 주기는 건너뜀: $$k_{next}=\max(k+1,\lfloor(t_{now}-t_{start})/\Delta t\rfloor+1)$$
            tick = max(tick + 1, int(np.floor((self.clock() - started) / self.settings.period_s)) + 1)
            # 다음 전송 시각까지 대기하며 도착 여부는 검사하지 않음: $$w=\max(0,t_{start}+k_{next}\Delta t-t_{now})$$
            delay = max(0., started + tick * self.settings.period_s - self.clock())
            await self.sleep(delay)
