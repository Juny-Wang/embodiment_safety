"""Render a recorded demonstration at its 10 Hz simulation rate."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.episode/'transitions.jsonl').read_text().splitlines()]
    observations = [rows[0]['observations']]+[r['next_observations'] for r in rows]
    cameras = ('external', 'wrist_left', 'wrist_right')
    def frame(obs):
        views = []
        for name in cameras:
            with Image.open(args.episode/obs['images'][name]['file']) as img:
                views.append(np.array(img.convert('RGB')))
        return np.concatenate(views, axis=1)
    first = frame(observations[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(args.output), (first.shape[1], first.shape[0]),
        fps=10, codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', quality=None,
        output_params=['-crf', '20', '-preset', 'fast', '-threads', '1', '-movflags', '+faststart'])
    writer.send(None)
    try:
        for obs in observations:
            writer.send(frame(obs))
    finally:
        writer.close()
    info = dict(episode=str(args.episode), video=str(args.output), fps=10,
                frames=len(observations), duration_s=len(observations)/10,
                playback_clock='simulation', cameras_left_to_right=cameras)
    args.output.with_suffix('.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
