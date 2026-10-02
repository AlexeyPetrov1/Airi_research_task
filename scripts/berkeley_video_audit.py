"""Decode real encoded comparison videos into reproducible audit contact sheets."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def audit(scene):
    source = scene / 'viz/side_by_side.mp4'
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f'Cannot decode {source}')
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = ''.join(chr((fourcc >> (8*i)) & 255) for i in range(4))
    if count != 10 or abs(fps-5) > 1e-6:
        raise ValueError(f'Wrong frame/time geometry: {count} at {fps} FPS')
    selected = []
    for index in (0, 4, 9):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = cap.read()
        if not success or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index+1:
            raise ValueError(f'Cannot decode exact encoded frame {index}')
        selected.append(frame)
    cap.release()
    output = scene / 'evaluation/video_audit.png'
    if not cv2.imwrite(str(output), np.vstack(selected)):
        raise IOError(output)
    info = {'source': str(source), 'source_encoded_frame_count': count, 'fps': fps, 'codec_fourcc': codec,
            'decoded_frame_indices': [0, 4, 9], 'physical_times_after_t0_s': [.2, 1., 2.],
            'single_frame_shape': list(selected[0].shape), 'contact_sheet': str(output),
            'source_is_encoded_video': True}
    (scene / 'evaluation/video_audit.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, nargs='+', required=True)
    for scene in parser.parse_args().scene_dir:
        audit(scene)
