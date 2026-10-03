"""Dataset registry — release version.

The release ships only the `trajectory_3d_*` family used to train the
MolmoMotion checkpoints. Training mode always uses every annotated clip
in each dataset; there is no held-out split for the human corpus.
Evaluation is run against PointMotionBench (HOT3D / WorldTrack / DAVIS),
which has its own benchmark dataset names.

Dataset-name grammar:

    trajectory_3d[_test][_<dataset_token>...]_p{P}_h{H}_f{F}

Examples:

    trajectory_3d_human_p8_h3_f8                  — Stage-1 pretrain recipe
    trajectory_3d_human_p8_h3_f30                 — Stage-2 finetune recipe (H=3, F=30)
    trajectory_3d_human_p8_h1_f32                 — Stage-2 finetune recipe (H=1, F=32)
    trajectory_3d_egodex_p8_h3_f8                 — single-dataset ablation
    trajectory_3d_test_hot3d_bench_p8_h3_f30      — PointMotionBench eval
    trajectory_3d_test_worldtrack_bench_p8_h3_f30
    trajectory_3d_test_davis_bench_p8_h3_f30

Tokens:
  _p{N}       num_points          (default 8)
  _h{N}       history_size        (default 3)
  _f{N}       num_future_frames   (default 8)
  _human      shorthand for `egodex,ytvis,hepic,xperience,stereo4d`
              (5 human-video datasets — the public training recipe)
  _test       evaluation mode: load the test slice of the dataset(s).
              Only meaningful for benchmarks and per-dataset ablations.
  _2d         enable 2D point-feature conditioning (SigLIP2 grid-sample)
  _2dcoord    encode 2D coords as text instead of features
"""

import inspect
import os
import re

from molmo_motion.data.dataset import Dataset
from molmo_motion.data.trajectory_3d_dataset import Trajectory3DDataset


# Five human-video datasets used by the public training recipe.
_HUMAN_DATASETS = ("egodex", "ytvis", "hepic", "xperience", "stereo4d")
# All dataset tokens the parser recognizes. Benchmarks come first so
# substring matches don't grab partial names (e.g. avoid matching `davis`
# inside `davis_bench`).
_PER_DATASET_TOKENS = ("hot3d_bench", "worldtrack_bench", "davis_bench",
                       "egodex", "hepic", "xperience", "ytvis", "stereo4d",
                       "droid", "molmospaces")


def get_dataset_by_name(dataset_name, split) -> Dataset:
    if not dataset_name.startswith("trajectory_3d"):
        raise NotImplementedError(
            f"Dataset {dataset_name!r} is not part of the public release. "
            f"Only 'trajectory_3d_*' dataset names are supported."
        )

    m_p = re.search(r"_p(\d+)", dataset_name)
    m_h = re.search(r"_h(\d+)", dataset_name)
    m_f = re.search(r"_f(\d+)", dataset_name)
    num_points = int(m_p.group(1)) if m_p else 8
    history = int(m_h.group(1)) if m_h else 3
    future = int(m_f.group(1)) if m_f else 8

    _split = "test" if "_test" in dataset_name else split

    if "_human" in dataset_name:
        datasets = list(_HUMAN_DATASETS)
    else:
        datasets = [ds for ds in _PER_DATASET_TOKENS if ds in dataset_name]
        if not datasets:
            datasets = list(_HUMAN_DATASETS)

    use_2d = ("_2d_" in dataset_name or dataset_name.endswith("_2d"))
    use_2d_coordinate = "_2dcoord" in dataset_name

    max_eval = os.environ.get("TRAJ3D_MAX_EVAL_PER_DATASET")
    max_eval = int(max_eval) if max_eval else None

    return Trajectory3DDataset(
        split=_split,
        num_points=num_points,
        num_future_frames=future,
        history_size=history,
        datasets=tuple(datasets),
        use_2d_point_features=use_2d,
        use_2d_coordinate=use_2d_coordinate,
        max_eval_per_dataset=max_eval,
        dataset_weighting="sqrt",
        use_camera_frame=True,
    )


def get_all_dataset_classes():
    return [x for x in globals().values() if (inspect.isclass(x) and issubclass(x, Dataset))]


def get_dataset_class_by_name(dataset_name):
    """Resolve dataset class without invoking its __init__ (used by `download`)."""
    split = "train"
    dataset_classes = get_all_dataset_classes()
    originals = {cls: cls.__init__ for cls in dataset_classes}
    try:
        for cls in dataset_classes:
            cls.__init__ = lambda *a, **kw: None
        dataset = get_dataset_by_name(dataset_name, split)
        return type(dataset)
    finally:
        for cls, init in originals.items():
            cls.__init__ = init


def download_dataset_by_name(dataset_name, n_procs=8):
    get_dataset_class_by_name(dataset_name).download(n_procs=n_procs)
