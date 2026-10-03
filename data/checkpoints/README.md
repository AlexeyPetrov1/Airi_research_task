# Model checkpoints

The exact original `model.pt` files for MolmoMotion H1-F32 and H3-F30 are distributed as ordered 1 GiB parts in the repository's [GitHub Release](https://github.com/AlexeyPetrov1/Airi_research_task/releases/tag/research-local-assets-2026-10-03).

From the repository root, restore either checkpoint:

```powershell
python scripts/restore_remaining_assets.py --only data/checkpoints/MolmoMotion-4B-H1-F32/model.pt
python scripts/restore_remaining_assets.py --only data/checkpoints/MolmoMotion-4B-H3-F30/model.pt
```

The script checks every part and the reconstructed file against SHA-256. The [manifest](../../report/remaining_assets_manifest.json) records all sizes, original file hashes, verified download URLs and publication status. A checkpoint is ready when its `complete` field is `true`.

See the [complete input restoration instructions](../../report/remaining_assets.md) for the other large research inputs.
