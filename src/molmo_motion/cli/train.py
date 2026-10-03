"""`molmo-motion-train` console script."""

from __future__ import annotations

import sys

from molmo_motion.exceptions import OLMoCliError
from molmo_motion.train.run_trainer import run_trainer
from molmo_motion.train.trainer_config import TrainConfig
from molmo_motion.util import clean_opt, prepare_torchrun_environment


def main():
    prepare_torchrun_environment()
    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError as e:
        raise OLMoCliError(
            "Usage: molmo-motion-train <CONFIG_YAML> [DOTTED-PATH=VALUE ...]"
        ) from e
    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    run_trainer(cfg)


if __name__ == "__main__":
    main()
