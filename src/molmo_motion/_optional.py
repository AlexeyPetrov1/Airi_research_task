"""Optional integration dependencies (wandb, beaker).

Training-time logging integrations are optional installs:

    pip install molmo-motion[train]    # wandb
    pip install beaker-py             # only useful on Ai2's Beaker cluster

When a package is missing, this module exposes a stub (wandb) or None
(beaker symbols) so the trainer imports cleanly; anything that actually
needs the integration raises a helpful error at use time.
"""

try:
    import wandb
    from wandb.sdk.data_types.base_types.wb_value import WBValue
except ModuleNotFoundError:
    WBValue = None

    class _WandbStub:
        """Stands in for the wandb module when it isn't installed.

        `wandb.run` is None (the universal "wandb not active" check used
        throughout the trainer); any other attribute access raises.
        """
        run = None

        def __getattr__(self, name):
            raise ModuleNotFoundError(
                "wandb is required for this feature — "
                "install it with `pip install molmo-motion[train]`")

    wandb = _WandbStub()

try:
    from beaker import Beaker
    from beaker.exceptions import BeakerError
    try:
        from beaker.client import ExperimentClient
    except ImportError:  # older versions of beaker-py
        from beaker import Experiment as ExperimentClient
except ModuleNotFoundError:
    Beaker = None
    ExperimentClient = None

    class BeakerError(Exception):
        """Placeholder so `except BeakerError` clauses stay valid."""
