"""Inspect a generated clip without loading a GPU tracker or real future."""
import argparse
from pathlib import Path
from das_evaluate import frames
from das_prepare_control import sheet
p=argparse.ArgumentParser();p.add_argument('video',type=Path);a=p.parse_args()
rgb=frames(a.video)
sheet(a.video.with_suffix('.png'),[rgb],['GENERATED'],indices=(0,4,8,12,16,32,48))
print(a.video.with_suffix('.png'))
