"""`molmo-motion-eval` console script."""

from __future__ import annotations

import sys

from molmo_motion.eval.model_evaluator import EvalConfig
from molmo_motion.exceptions import OLMoCliError
from molmo_motion.util import clean_opt, prepare_torchrun_environment


def main():
    prepare_torchrun_environment()
    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError as e:
        raise OLMoCliError(
            "Usage: molmo-motion-eval <CONFIG_YAML> [DOTTED-PATH=VALUE ...]"
        ) from e
    cfg = EvalConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    cfg.build().run()


if __name__ == "__main__":
    main()
