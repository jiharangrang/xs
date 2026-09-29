"""발표 재구성의 재생·검사·MP4 내보내기를 명령행에서 실행한다."""

import argparse
import json
from pathlib import Path

import numpy as np

from presentation.choreography import Choreography, PresentationSettings
from presentation.recording import DEFAULT_RECORDING, Recording, recording_directory
from presentation.rendering import CAMERAS, export_video
from presentation.video import probe_video


def main():
    """모든 처리를 발표 폴더의 독립 경로로 실행한다."""
    parser = argparse.ArgumentParser(description='실제 시간에 맞춘 발표용 기구학 재구성. 실물 모터에 연결하지 않습니다.')
    parser.add_argument('mode', choices=['play', 'export', 'inspect', 'assemble', 'quad'], nargs='?', default='play')
    parser.add_argument('--recording', type=Path, default=DEFAULT_RECORDING)
    parser.add_argument('--start', type=float, help='6단계 완료 기준 시작 초, 음수는 준비 과정')
    parser.add_argument('--end', type=float, help='6단계 완료 기준 종료 초')
    parser.add_argument('--speed', type=float, help='기본값: 발표 편집은 5배속, 원본 재생·단일 내보내기는 1배속')
    parser.add_argument('--sag-mm', type=float, default=15.0, help='실측 처짐이 아닌 설명용 추가 처짐')
    parser.add_argument('--tilt-deg', type=float, default=8.0)
    parser.add_argument('--camera', choices=list(CAMERAS), default='oblique')
    parser.add_argument('--output', type=Path, default=Path('outputs/presentation/reconstruction.mp4'))
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--video', type=Path, action='append', default=[], help='촬영 시각을 읽을 원본 영상; 반복 지정 가능')
    parser.add_argument('--video-offset', type=float, action='append', default=[], help='각 --video의 시계 보정 초; 양수면 로그상 시작을 늦춤')
    parser.add_argument('--match-videos', action='store_true', help='원본 클립별로 공백 없이 이어 편집할 수 있는 영상 저장')
    parser.add_argument('--edit', type=Path, help='뷰어가 따를 공통 편집표 JSON')
    parser.add_argument('--depth-video', type=Path, help='SIDE 왼쪽 아래에 동기화해 넣을 뎁스 녹화')
    parser.add_argument('--depth-offset', type=float, default=0.0, help='뎁스 녹화 메타데이터 시각 보정 초')
    parser.add_argument('--clean', action='store_true', help='화면의 재구성 표시 생략; 별도 SRT에는 표시 유지')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.speed is None:
        args.speed = 5.0 if args.mode == 'assemble' else 1.0
    if args.mode == 'quad' and args.edit is None:
        parser.error('quad에는 실영상과 공통 시간축을 지정하는 --edit가 필요합니다.')
    if args.depth_video and args.mode != 'quad':
        parser.error('--depth-video는 quad에서만 사용할 수 있습니다.')
    if not np.isfinite(args.depth_offset):
        parser.error('뎁스 시계 보정은 유한한 값이어야 합니다.')
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('배속은 유한한 양수이어야 합니다.')
    if args.video_offset and len(args.video_offset) != len(args.video):
        parser.error('--video-offset은 --video마다 하나씩 지정해 주세요.')
    if not all(np.isfinite(value) for value in args.video_offset):
        parser.error('영상 오프셋은 유한한 값이어야 합니다.')
    if args.match_videos and (not args.video or args.mode != 'export' or args.start is not None or args.end is not None):
        parser.error('--match-videos는 export와 --video를 함께 사용하며 --start/--end와 함께 쓸 수 없습니다.')
    with recording_directory(args.recording) as root:
        recording = Recording(root)
        start = recording.start if args.start is None else args.start
        end = recording.end if args.end is None else args.end
        if not np.isfinite(start) or not np.isfinite(end) or not recording.start <= start < end <= recording.end:
            parser.error(f'재생 범위는 {recording.start:.3f}초부터 {recording.end:.3f}초 사이이어야 합니다.')
        offsets = args.video_offset or [0.0] * len(args.video)
        clips = [probe_video(path, recording, offset) for path, offset in zip(args.video, offsets, strict=True)]
        if args.mode == 'inspect':
            print(json.dumps({'basis': 'stage6_complete', 'origin_kst': recording.wall_origin.isoformat(), 'range_s': [recording.start, recording.end],
                              'runs': [{'stage': run['stage'], 'start_s': run['start_relative_walk_s'], 'end_s': run['end_relative_walk_s'], 'result': run['state']} for run in recording.runs],
                              'videos': [{'file': clip.path.name, 'start_s': clip.start, 'duration_s': clip.duration, 'fps': clip.fps, 'sync': 'metadata_plus_manual_offset'} for clip in clips]}, ensure_ascii=False, indent=2))
            return
        choreography = Choreography(recording, PresentationSettings(sag_mm=args.sag_mm, tilt_deg=args.tilt_deg))
        print(f'발표 경로 준비: {len(choreography.motions)}개 이동 구간', flush=True)
        if args.mode == 'quad':
            from presentation.editing import EditTimeline
            from presentation.four_view import compose_four_view
            edit = EditTimeline.load(args.edit)
            depth_clip = probe_video(args.depth_video, recording, args.depth_offset) if args.depth_video else None
            output = compose_four_view(choreography, edit, args.edit.parent, args.output,
                                       overwrite=args.overwrite, depth_clip=depth_clip)
            print(f'저장 완료: {output}', flush=True)
            return
        if args.mode == 'assemble':
            from presentation.editing import assemble, build_edit
            if not clips:
                parser.error('assemble에는 --video가 필요합니다.')
            edit = build_edit(choreography, clips, speed=args.speed, fps=args.fps)
            for output in assemble(choreography, edit, args.output, overwrite=args.overwrite):
                print(f'저장 완료: {output}', flush=True)
            return
        if args.mode == 'play':
            from presentation.player import show
            from presentation.editing import EditTimeline
            edit = EditTimeline.load(args.edit) if args.edit else None
            show(choreography, start=start, end=end, speed=args.speed, camera_name=args.camera, edit=edit)
            return
        intervals = [(args.output, start, end, args.fps)]
        if args.match_videos:
            intervals = []
            for index, clip in enumerate(clips, 1):
                clip_end = clip.start + clip.duration
                if clip.start < recording.start or clip_end > recording.end:
                    parser.error(f'{clip.path.name}의 영상 범위가 기록 범위를 벗어납니다.')
                output = args.output.parent / f'{args.output.stem}_{index:02}_{args.camera}{args.output.suffix}'
                intervals.append((output, clip.start, clip_end, clip.fps))
        for output, first, last, fps in intervals:
            export_video(choreography, output, start=first, end=last, speed=args.speed, fps=fps, width=args.width, height=args.height,
                         camera_name=args.camera, overlay=not args.clean, overwrite=args.overwrite)
            print(f'저장 완료: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
