# Repository packaging

Logical final distributable tree; excludes .git, ignored checkpoints, generated outputs, build/dist and environments.

| Measurement | BEFORE | AFTER |
|---|---:|---:|
| Files | 948 | 191 |
| Bytes | 307592043 | 63389987 |
| MiB | 293.34 | 60.45 |
| Duplicate bytes by SHA-256 | 156076517 | 0 |

Saved **244202056 bytes (79.39%)**.

| Top-level path | BEFORE bytes | AFTER bytes |
|---|---:|---:|
| .gitattributes | 166 | 166 |
| .gitignore | 103 | 116 |
| LICENSE | 11555 | 11555 |
| README.md | 11964 | 5764 |
| configs | 2737 | 2906 |
| data | 151696772 | 0 |
| docs | 2177892 | 409483 |
| fixtures | 0 | 60538661 |
| legacy | 16213 | 0 |
| pyproject.toml | 891 | 2717 |
| scripts | 17114 | 0 |
| src | 1844915 | 1836266 |
| tests | 151811721 | 569065 |
| tools | 0 | 13288 |

Removed the duplicated legacy dataset trees, obsolete intermediate variants, full processor packets and legacy Python helpers. The six comparison videos, exact forecasts and required observed/evaluation arrays remain once. RGB and arrays use measured lossless deflate. Dobb-E shares its history frames across two bundles. No Git history was rewritten.

All retained arrays were decoded and checked exactly. Original dtype, shape, pixel values and NumPy strides are preserved. Compression details and reconstructed content hashes are in `provenance.json`.

Largest 20 artifacts after packaging:

| Path | Bytes |
|---|---:|
| fixtures/davis_bmx_trees/evaluation.npz | 27742191 |
| fixtures/berkeley_bottle/evaluation.npz | 4218078 |
| fixtures/berkeley_cup/evaluation.npz | 4133182 |
| fixtures/dobbe_pure_vipe/evaluation.npz | 3617520 |
| fixtures/fmb_wrist_2/evaluation.npz | 3057665 |
| fixtures/davis_bmx_trees/observed.npz | 2528739 |
| fixtures/fmb_wrist_1/evaluation.npz | 2516044 |
| fixtures/davis_bmx_trees/media/legacy_comparison.mp4 | 2367074 |
| fixtures/berkeley_bottle/media/legacy_comparison.mp4 | 1835913 |
| fixtures/berkeley_cup/media/legacy_comparison.mp4 | 1804908 |
| fixtures/berkeley_cup/observed.npz | 1288990 |
| fixtures/fmb_wrist_2/media/legacy_comparison.mp4 | 1288293 |
| fixtures/berkeley_bottle/observed.npz | 1258382 |
| fixtures/fmb_wrist_1/media/legacy_comparison.mp4 | 1128112 |
| tests/golden_manifest.json | 551692 |
| fixtures/fmb_wrist_2/observed.npz | 473215 |
| fixtures/fmb_wrist_1/observed.npz | 399645 |
| fixtures/shared/dobbe_history.npz | 369285 |
| fixtures/dobbe_pure_vipe/media/legacy_comparison.mp4 | 367852 |
| docs/packaging_report.json | 211320 |

Full before/after largest-20 lists and SHA-256 duplicate groups: [packaging_report.json](packaging_report.json).

The reports are generated documentation; totals are measured immediately before writing this report. The Git object database and historical branches retain the original history.
