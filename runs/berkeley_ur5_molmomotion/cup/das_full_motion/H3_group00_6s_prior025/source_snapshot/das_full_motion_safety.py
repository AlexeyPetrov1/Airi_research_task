"""Fail closed if preparation/generation attempts to read real future data."""
import os
from pathlib import Path
import sys


def forbid_real_future():
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=Path(os.fsdecode(args[0])).resolve()
        if 'evaluation' in path.parts and ('AIRI_task' in path.parts or 'runs' in path.parts):
            raise RuntimeError(f'Real future input is prohibited in this experiment: {path}')
    sys.addaudithook(audit)
