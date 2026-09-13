"""관절각을 입력받아 FK 모듈을 실행하고 결과를 터미널에 출력한다."""

import argparse

import numpy as np

from kinematics.fk import ARM_JOINT_NAMES, ForwardKinematics


def main() -> None:
    """명령행 관절각 또는 초기 관절각으로 FK를 계산해 출력한다."""
    parser = argparse.ArgumentParser(description="XS 로봇의 FK 계산 확인")
    parser.add_argument(
        "--q-rad",
        nargs=7,
        type=float,
        default=[0.0] * 7,
        metavar="각도",
        help="J1부터 J7 순서의 관절각(rad). 생략하면 모두 0을 사용합니다.",
    )
    args = parser.parse_args()
    fk = ForwardKinematics()
    try:
        result = fk.forward(args.q_rad)
    except ValueError as error:
        parser.error(str(error))

    np.set_printoptions(precision=6, suppress=True)
    print("입력 순서:", ", ".join(ARM_JOINT_NAMES))
    print("입력 관절각(rad):", np.asarray(args.q_rad))
    print("tip_L의 월드 위치(m):", result.T_world_tip_L[:3, 3])
    print("tip_R의 월드 위치(m):", result.T_world_tip_R[:3, 3])
    print("tip_L 기준 tip_R의 위치(m):", result.T_tip_L_tip_R[:3, 3])
    print("tip_L 기준 tip_R의 회전행렬:\n", result.T_tip_L_tip_R[:3, :3])
    print("tip_L 기준 tip_R의 동차변환:\n", result.T_tip_L_tip_R)


if __name__ == "__main__":
    main()
