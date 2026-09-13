"""명령행에서 입력한 일곱 관절각을 자세 뷰어에 전달한다."""

import argparse

import numpy as np

from kinematics.fk import DEFAULT_MODEL_PATH
from kinematics.joints import as_joint_angles
from simulation.model import load_model


def main() -> None:
    """도 또는 라디안 관절각으로 뷰어를 열거나 창 없이 좌표를 확인한다."""
    parser = argparse.ArgumentParser(description="XS 로봇 관절각 시각화")
    angles = parser.add_mutually_exclusive_group()
    angles.add_argument("--q-deg", nargs=7, type=float, metavar="각도", help="J1부터 J7 순서의 각도(deg)")
    angles.add_argument("--q-rad", nargs=7, type=float, metavar="각도", help="J1부터 J7 순서의 각도(rad)")
    parser.add_argument("--check", action="store_true", help="창 없이 모델 로드와 팁 좌표 확인")
    args = parser.parse_args()
    try:
        if args.q_deg is not None:
            # 도 단위 입력을 모듈의 라디안 단위로 변환: $$q_{\mathrm{rad}}=q_{\mathrm{deg}}\pi/180$$
            q_rad = np.deg2rad(as_joint_angles(args.q_deg))
        else:
            q_rad = as_joint_angles(args.q_rad if args.q_rad is not None else np.zeros(7))
        if args.check:
            model, data = load_model(q_rad)
            print(f"모델 로드 완료: {DEFAULT_MODEL_PATH}")
            print(f"관절 {model.njnt}개 / 메시 {model.nmesh}개")
            print("입력 관절각(rad):", q_rad)
            print("tip_L 월드 위치(m):", data.site("tip_L").xpos)
            print("tip_R 월드 위치(m):", data.site("tip_R").xpos)
        else:
            from simulation.viewer import show_pose

            show_pose(q_rad)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
