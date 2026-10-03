"""MolmoMotion: 3D point-trajectory prediction.

Public API exposed at the top level so users can do:

    from molmo_motion import MolmoMotion, MolmoMotionProcessor, MolmoMotionConfig

The model and processor are HuggingFace-compatible
(`from_pretrained` / `from_config` / `save_pretrained`).
"""

from molmo_motion._version import __version__
from molmo_motion.public_config import MolmoMotionConfig
from molmo_motion.modeling import MolmoMotion, MolmoMotionOutput
from molmo_motion.processor import MolmoMotionProcessor

__all__ = [
    "MolmoMotion",
    "MolmoMotionConfig",
    "MolmoMotionOutput",
    "MolmoMotionProcessor",
    "__version__",
]
