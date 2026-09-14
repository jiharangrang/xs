"""R팁을 고정하고 L팁을 빔 방향으로 당기는 직선 경로를 계산해 결과를 출력한다."""

import argparse
from pathlib import Path
import shlex

import numpy as np

from kinematics.anchoring import TipAnchor
from kinematics.fk import ForwardKinematics
from kinematics.ik import InverseKinematics
from kinematics.joints import as_joint_angles
from planning.linear_motion import plan_linear_motion
from planning.motion_path import MotionPath, MotionSegment, save_path
from planning.targets import beam_grasp_target


def main() -> None:
    """입력 자세 또는 빔 파지 IK 결과에서 시작해 뒷그리퍼 직선 이동을 계산한다."""
    parser = argparse.ArgumentParser(description="R 고정 상태에서 L을 빔 방향으로 당기기")
    angles = parser.add_mutually_exclusive_group()
    angles.add_argument("--q-start-deg", nargs=7, type=float, metavar="각도", help="시작 관절각(deg)")
    angles.add_argument("--q-start-rad", nargs=7, type=float, metavar="각도", help="시작 관절각(rad)")
    parser.add_argument(
        "--grasp-distance-m", type=float, default=0.20,
        help="시작 각도 생략 시 파지 IK로 만들 양쪽 팁 간격(m)",
    )
    parser.add_argument("--candidate", type=int, default=2, help="시작 각도 생략 시 사용할 파지 IK 후보 번호(기본 2)")
    parser.add_argument("--distance-m", type=float, default=0.10, help="L을 당길 거리(m)")
    parser.add_argument("--steps", type=int, default=20, help="직선을 나눌 구간 수")
    parser.add_argument("--duration-s", type=float, default=4.0, help="화면에서 재생할 시간(s), 모터 속도 계획과 별개")
    parser.add_argument("--save", type=Path, default=Path("outputs/rear_pull.json"), help="경로를 저장할 JSON 파일")
    args = parser.parse_args()
    try:
        if not np.isfinite(args.distance_m) or args.distance_m <= 0:
            raise ValueError("당길 거리는 유한한 양수여야 합니다.")
        if args.steps < 1:
            raise ValueError("구간 수는 1 이상이어야 합니다.")
        if args.candidate < 1:
            raise ValueError("후보 번호는 1 이상이어야 합니다.")
        if not np.isfinite(args.duration_s) or args.duration_s <= 0:
            raise ValueError("재생 시간은 유한한 양수여야 합니다.")
        segment_name = "rear_pull"
        if args.q_start_deg is not None:
            # 도 단위 입력을 라디안으로 변환: $$q_{\mathrm{rad}}=q_{\mathrm{deg}}\pi/180$$
            q_start = np.deg2rad(as_joint_angles(args.q_start_deg))
        elif args.q_start_rad is not None:
            q_start = as_joint_angles(args.q_start_rad)
        else:
            print(f"시작 자세 계산: 팁 간격 {args.grasp_distance_m:g} m의 파지 IK", flush=True)
            grasp = InverseKinematics().solve(beam_grasp_target(args.grasp_distance_m), np.zeros(7))
            if not grasp.success:
                print(grasp.message)
                raise SystemExit(1)
            if args.candidate > len(grasp.candidates):
                raise ValueError(f"구해진 후보는 {len(grasp.candidates)}개입니다. 그 범위에서 후보를 선택해 주세요.")
            q_start = grasp.candidates[args.candidate - 1].q_rad
            segment_name = f"rear_pull_candidate_{args.candidate}"
            print(f"파지 IK 후보 {args.candidate}/{len(grasp.candidates)}에서 시작합니다.")
            print("쭉 편 자세부터의 진입 경로는 이 예제에 포함되지 않습니다.")

        fk = ForwardKinematics()
        start = fk.forward(q_start)
        anchor = TipAnchor("tip_R", start.T_world_tip_R)
        # 현재 모델의 빔 전진 방향인 월드 X축으로 변위 지정: $$\Delta p_W=d[1,0,0]^T$$
        displacement = np.array([args.distance_m, 0.0, 0.0])
        result = plan_linear_motion(q_start, anchor, displacement, steps=args.steps)
    except ValueError as error:
        parser.error(str(error))

    print(result.message)
    if not result.success:
        raise SystemExit(1)
    # 계산된 경로 지점에 일정한 재생 시간 간격 지정: $$t_k=kT/K$$
    time_s = np.linspace(0.0, args.duration_s, len(result.q_path_rad))
    motion_path = MotionPath((MotionSegment(segment_name, anchor, time_s, result.q_path_rad),))
    try:
        saved_path = save_path(motion_path, args.save)
    except OSError as error:
        parser.error(str(error))
    end = anchor.place(fk.forward(result.q_path_rad[-1]))
    np.set_printoptions(precision=9, suppress=True)
    print("관절 순서: J1 J2 J3 J4 J5 J6 J7")
    print("관절 경로 배열 형상:", result.q_path_rad.shape)
    # 시작 관절각을 도 단위로 변환: $$q_{s,\mathrm{deg}}=q_{s,\mathrm{rad}}180/\pi$$
    start_degrees = np.rad2deg(result.q_path_rad[0])
    # 마지막 관절각을 도 단위로 변환: $$q_{e,\mathrm{deg}}=q_{e,\mathrm{rad}}180/\pi$$
    end_degrees = np.rad2deg(result.q_path_rad[-1])
    print("시작 관절각(deg):", start_degrees)
    print("도착 관절각(deg):", end_degrees)
    print("L팁 월드 시작 위치(m):", start.T_world_tip_L[:3, 3])
    print("L팁 월드 도착 위치(m):", end.T_world_tip_L[:3, 3])
    print("고정 R팁 월드 시작 위치(m):", start.T_world_tip_R[:3, 3])
    print("고정 R팁 월드 도착 위치(m):", end.T_world_tip_R[:3, 3])
    print(f"최대 FK 표본 위치 오차(m): {result.max_position_error_m:.9g}")
    print(f"최대 FK 표본 방향 오차(rad): {result.max_rotation_error_rad:.9g}")
    print(f"인접 지점 사이 최대 관절각 변화(rad): {result.max_joint_step_rad:.9g}")
    print("시간·속도 계획, 충돌 검증, 실물 오차 보정과 전체 보행 최적화는 포함하지 않습니다.")
    print(f"경로 저장: {saved_path.resolve()}")
    print(f"재생: uv run mjpython -m scripts.view_model --path {shlex.quote(str(saved_path))}")


if __name__ == "__main__":
    main()
