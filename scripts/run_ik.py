"""빔을 잡는 목표 자세로 IK를 실행하고 결과 및 뷰어 실행 명령을 출력한다."""

import argparse
import time

import numpy as np

from kinematics.fk import ForwardKinematics
from kinematics.ik import IKSettings, InverseKinematics
from kinematics.joints import as_joint_angles
from planning.targets import beam_grasp_target


def main() -> None:
    """목표 간격과 현재 관절각으로 IK를 계산하고 선택하면 결과 뷰어를 연다."""
    parser = argparse.ArgumentParser(description="빔 앞쪽을 잡는 R팁의 IK 계산")
    parser.add_argument("--distance-m", type=float, default=0.20, help="L팁에서 빔 전진 방향의 간격(m)")
    angles = parser.add_mutually_exclusive_group()
    angles.add_argument("--q-start-deg", nargs=7, type=float, metavar="각도", help="실제 시작 관절각(deg)")
    angles.add_argument("--q-start-rad", nargs=7, type=float, metavar="각도", help="실제 시작 관절각(rad)")
    parser.add_argument("--starts", type=int, default=24, help="계산 시작값의 개수")
    parser.add_argument("--seed", type=int, default=0, help="계산 시작값을 재현할 난수 시드")
    parser.add_argument("--view", action="store_true", help="결과 뷰어 열기(macOS에서는 mjpython 필요)")
    args = parser.parse_args()
    try:
        if args.q_start_deg is not None:
            # 시작 관절각을 라디안으로 변환: $$q_{s,\mathrm{rad}}=q_{s,\mathrm{deg}}\pi/180$$
            q_start = np.deg2rad(as_joint_angles(args.q_start_deg))
        else:
            q_start = as_joint_angles(args.q_start_rad if args.q_start_rad is not None else np.zeros(7))
        target = beam_grasp_target(args.distance_m)
        solver = InverseKinematics(settings=IKSettings(starts=args.starts, random_seed=args.seed))
        print("목표: 같은 높이·횡방향 위치, 빔 앞으로 이동, X·Y축 반대 / Z축 동일", flush=True)
        print(f"목표 간격: {args.distance_m:g} m", flush=True)
        print(f"IK 계산 시작: 최대 {args.starts}개 시작값, 각 관절 동일 가중치", flush=True)
        started = time.perf_counter()
        result = solver.solve(target, q_start)
        elapsed = time.perf_counter() - started
    except ValueError as error:
        parser.error(str(error))

    print(result.message)
    print(f"성공한 시도: {result.solved_attempts}/{result.attempts}, 계산 시간: {elapsed:.3f}초")
    if not result.success:
        raise SystemExit(1)
    print(f"중복을 제외한 보관 후보: {len(result.candidates)}개 (최대 {solver.settings.max_candidates}개)")
    for rank, candidate in enumerate(result.candidates, start=1):
        print(f"  후보 {rank}: 비용 {candidate.cost:.9f}")
    selected = result.candidates[0]
    np.set_printoptions(precision=9, suppress=True)
    print("첫 화면: 후보 1 / 후보 뷰어에서 Enter 키로 다음 후보 보기")
    print("관절 순서: J1 J2 J3 J4 J5 J6 J7")
    print("관절각(rad):", selected.q_rad)
    # 결과 관절각을 읽기 쉬운 도 단위로 변환: $$q_{\mathrm{deg}}=q_{\mathrm{rad}}180/\pi$$
    q_degrees = np.rad2deg(selected.q_rad)
    print("관절각(deg):", q_degrees)
    print(f"비용: {selected.cost:.9f}")
    print(f"위치 오차(m): {selected.position_error_m:.3e}")
    print(f"방향 오차(rad): {selected.rotation_error_rad:.3e}")
    print("L팁 기준 R팁 목표:\n", target)
    actual = ForwardKinematics().forward(selected.q_rad).T_tip_L_tip_R
    print("L팁 기준 R팁 계산 결과:\n", actual)
    command_angles = " ".join(f"{angle:.6f}" for angle in q_degrees)
    print("이 각도 그대로 다시 보기:")
    print(f"uv run mjpython -m scripts.view_model --q-deg {command_angles}")
    print("도착 자세의 기구학 결과입니다. 충돌·이동 경로·실물 보정은 아직 평가하지 않습니다.")
    if args.view:
        from simulation.viewer import show_pose

        show_pose(selected.q_rad, other_candidates=[item.q_rad for item in result.candidates[1:]])


if __name__ == "__main__":
    main()
