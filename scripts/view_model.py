"""명령행 관절각 또는 저장된 경로 파일을 공용 MuJoCo 뷰어에 전달한다."""

import argparse
from pathlib import Path

import numpy as np

from kinematics.fk import DEFAULT_MODEL_PATH
from kinematics.joints import as_joint_angles
from planning.motion_path import load_path
from simulation.model import load_model, set_path_time


def main() -> None:
    """도 또는 라디안 관절각으로 뷰어를 열거나 창 없이 좌표를 확인한다."""
    parser = argparse.ArgumentParser(description="XS 로봇 관절각 시각화")
    angles = parser.add_mutually_exclusive_group()
    angles.add_argument("--q-deg", nargs=7, type=float, metavar="각도", help="J1부터 J7 순서의 각도(deg)")
    angles.add_argument("--q-rad", nargs=7, type=float, metavar="각도", help="J1부터 J7 순서의 각도(rad)")
    angles.add_argument("--path", type=Path, help="재생할 경로 JSON 파일")
    parser.add_argument("--check", action="store_true", help="창 없이 모델 로드와 팁 좌표 확인")
    args = parser.parse_args()
    try:
        motion_path = load_path(args.path) if args.path is not None else None
        if motion_path is not None:
            q_rad = motion_path.segments[0].q_rad[0]
        elif args.q_deg is not None:
            # 도 단위 입력을 모듈의 라디안 단위로 변환: $$q_{\mathrm{rad}}=q_{\mathrm{deg}}\pi/180$$
            q_rad = np.deg2rad(as_joint_angles(args.q_deg))
        else:
            q_rad = as_joint_angles(args.q_rad if args.q_rad is not None else np.zeros(7))
        if args.check:
            model, data = load_model(q_rad)
            if motion_path is not None:
                set_path_time(model, data, motion_path, 0.0)
                print(f"경로: {len(motion_path.segments)}구간 / 재생 {motion_path.duration_s:g}초")
            print(f"모델 로드 완료: {DEFAULT_MODEL_PATH}")
            print(f"관절 {model.njnt}개 / 메시 {model.nmesh}개")
            print("입력 관절각(rad):", q_rad)
            print("tip_L 월드 위치(m):", data.site("tip_L").xpos)
            print("tip_R 월드 위치(m):", data.site("tip_R").xpos)
        else:
            from simulation.viewer import show_path, show_pose

            if motion_path is not None:
                show_path(motion_path)
            else:
                show_pose(q_rad)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
