"""HuggingFace adapter for MolmoMotion.

Hosts the HF-compatible `Molmo2Config` / `Molmo2ForConditionalGeneration` /
`Molmo2Processor` classes plus the CLI converter that turns an OLMo-native
training checkpoint into a directory `AutoModel.from_pretrained(...)` can load.

Implementation note — *parameter-equivalence with the base Molmo2*:
the trajectory variant adds a `PointFeatureConditioner` whose `extractor`
is parameter-free (adaptive pool + grid-sample) and whose `_projector`
is a *reference* to the existing vision-backbone image-projector (not a
child `nn.Module`). The result is that a `Molmo2Trajectory` checkpoint
has the **same parameter set** as base `Molmo2`, so the existing
`convert_molmo2` weight mapping just works on trajectory checkpoints
without modification.

The 2D-point-feature conditioning logic itself is currently exposed only
through the native `MolmoMotion` wrapper in `molmo_motion.modeling`. A
direct HF-port of that conditioning into the `Molmo2ForConditionalGeneration`
forward pass is on the roadmap; for now, HF users who want the trajectory
feature path should go through the native wrapper.
"""

from molmo_motion.hf_model.configuration_molmo_motion import (
    Molmo2AdapterConfig,
    Molmo2Config,
    Molmo2TextConfig,
    Molmo2VitConfig,
)
from molmo_motion.hf_model.modeling_molmo_motion import (
    Molmo2ForConditionalGeneration,
    Molmo2Model,
    Molmo2PreTrainedModel,
)
from molmo_motion.hf_model.processing_molmo_motion import Molmo2Processor

__all__ = [
    "Molmo2AdapterConfig",
    "Molmo2Config",
    "Molmo2ForConditionalGeneration",
    "Molmo2Model",
    "Molmo2PreTrainedModel",
    "Molmo2Processor",
    "Molmo2TextConfig",
    "Molmo2VitConfig",
]
