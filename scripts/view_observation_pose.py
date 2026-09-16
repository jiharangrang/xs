"""저장된 최저 비용 파지 자세로 관측 시작 자세를 만들고 공용 MuJoCo 뷰어에 표시한다.
계산 결과 저장과 명령행 처리를 담당하며 실물 장치에는 연결하지 않는다.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from planning.observation_pose import plan_observation_pose
from simulation.beam import beam_reference
from simulation.model import load_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs" / "ik_search_10000.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "stage1_observation_pose.json"


def main() -> None:
    r"""내릴 높이를 받아 시작 자세를 저장하고 선택에 따라 파지 자세와 비교 표시한다.

    $$h=h_{mm}/1000$$
    """
    parser = argparse.ArgumentParser(description="FSM 1단계: 빔 아래 관측 시작 자세")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="기존 IK 탐색 결과 JSON")
    parser.add_argument("--drop-mm", type=float, default=50.0, help="파지 자세에서 낮출 높이(mm), 기본 50")
    parser.add_argument("--save", type=Path, default=DEFAULT_OUTPUT, help="시작 자세를 저장할 JSON")
    parser.add_argument("--check", action="store_true", help="자세를 계산·저장하고 창 없이 종료")
    args = parser.parse_args()
    try:
        source = json.loads(args.source.read_text(encoding="utf-8"))
        if source.get("format") != "xs.ik_search.v1" or not source.get("top_candidates"):
            raise ValueError("최저 비용 후보가 포함된 xs.ik_search.v1 파일이 필요합니다.")
        q_grasp = source["top_candidates"][0]["q_rad"]
        # 밀리미터 입력을 미터로 변환: $$h=h_{mm}/1000$$
        drop_m = args.drop_mm / 1000.0
        pose = plan_observation_pose(q_grasp, drop_m=drop_m)
        model, data = load_model(pose.q_rad)
        reference = beam_reference(model, data)
        # 결과 관절각을 도 단위로 변환: $$q_{deg}=q_{rad}180/\pi$$
        q_deg = np.rad2deg(pose.q_rad)
        payload = {
            "format": "xs.observation_pose.v1",
            "stage": 1,
            "source": str(args.source.resolve()),
            "fixed_tip": "tip_L",
            "drop_m": pose.drop_m,
            "q_grasp_rad": pose.q_grasp_rad.tolist(),
            "q_rad": pose.q_rad.tolist(),
            "q_deg": q_deg.tolist(),
            "T_world_tip_R_goal": pose.target_world.tolist(),
            "position_error_m": pose.position_error_m,
            "rotation_error_rad": pose.rotation_error_rad,
            "camera_beam_reference": reference,
        }
        if args.save.resolve() == args.source.resolve():
            raise ValueError("기존 IK 탐색 결과에 시작 자세를 덮어쓸 수 없습니다.")
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"관측 시작 자세: 파지 위치에서 {args.drop_mm:g} mm 아래 / L 고정", flush=True)
        print("J1~J7 각도(deg):", np.round(q_deg, 6), flush=True)
        print("R팁 월드 위치(m):", data.site("tip_R").xpos, flush=True)
        print("빔 하부면의 카메라 기준 법선:", reference["normal"], flush=True)
        print(f"카메라와 빔 하부면 거리(m): {reference['plane_offset_m']:.6f}", flush=True)
        print(f"저장: {args.save.resolve()}", flush=True)
        if not args.check:
            from simulation.viewer import show_pose

            print("후보 1: 관측 시작 자세 / 후보 2: 원래 파지 자세 / Enter: 전환", flush=True)
            show_pose(pose.q_rad, other_candidates=[pose.q_grasp_rad])
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
