"""Console-script entry points for `pip install molmo-motion`.

These are tiny re-exports of the top-level `launch_scripts/*.py` files;
having them inside the installable package lets `pip` register them as
`molmo-motion-train` / `molmo-motion-eval` / `molmo-motion-visualize`
console scripts. The top-level launchers stay for users who clone the
repo and run with `torchrun launch_scripts/train.py ...` directly.
"""
