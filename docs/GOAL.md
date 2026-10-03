# GOAL: Final repository packaging
## Minimize and simplify the repository without changing any experiment behavior

Baseline commit:

    70ed2c413eee5e3b9dc1324e2c87ff333ee88334
    branch: molmo-motion-final

The current repository is already functionally correct.

DO NOT redesign the research.
DO NOT improve the algorithms.
DO NOT change the selected examples.
DO NOT change model inputs, forecasts, metrics, postprocessing or visualizations
unless required purely for packaging.

This task is a second-stage cleanup:

> Turn the current verified repository into a compact, self-contained,
> presentation-ready repository while preserving the existing CLI and all
> numerical behavior.

The existing repository is the behavioral reference.


# 1. Current verified behavior must remain unchanged

The following commands and concepts must continue to work:

    molmo-motion-experiment --config ...
    python -m motion_experiments.run ...

Modes:

    --mode inference
    --mode replay
    --predictions-from ...

The following seven cases must remain supported:

    author_davis
    fmb_wrist_1
    fmb_wrist_2
    berkeley_bottle
    berkeley_cup
    dobbe
    dobbe_blocked

Expected semantics must remain unchanged:

- DAVIS: completed author example;
- FMB wrist 1/2: completed prepared experiments;
- Berkeley bottle/cup: completed prepared experiments;
- Dobb-E pure_vipe: completed conditional diagnostic;
- Dobb-E hybrid: expected `SKIPPED_GEOMETRY_GATE`.

In particular:

    dobbe_blocked

must still terminate BEFORE forecast/evaluation with the same meaningful
failure reason.

A negative experiment is part of the expected behavior, not an error to fix.


# 2. Important observation about the current repository

Before changing anything, verify this independently.

At the baseline commit the repository currently contains approximately:

    data/legacy/               ~144.7 MiB
    tests/golden/              ~144.8 MiB

and:

    data/legacy/
    tests/golden/source/

contain the same legacy artifact tree.

Their Git tree/blob identities show that the second copy does not provide
independent experimental information.

It mainly duplicates the checkout and repository structure.

The new repository must have ONE canonical copy of each large experimental
artifact.

Do not keep a second physical copy merely to call it "golden".


# 3. Target architecture

Move from:

    old research directory layout
           ↓
    dataset adapter
           ↓
    pipeline

to:

    compact immutable experiment bundle
           ↓
    dataset-specific interpretation
           ↓
    common pipeline

Suggested structure:

    .
    ├── README.md
    ├── pyproject.toml
    │
    ├── configs/
    │   ├── author_davis.json
    │   ├── fmb_wrist_1.json
    │   ├── fmb_wrist_2.json
    │   ├── berkeley_bottle.json
    │   ├── berkeley_cup.json
    │   ├── dobbe.json
    │   └── dobbe_blocked.json
    │
    ├── src/
    │   ├── motion_experiments/
    │   │   ├── datasets/
    │   │   ├── run.py
    │   │   ├── sample.py
    │   │   ├── geometry.py
    │   │   ├── metrics.py
    │   │   ├── molmomotion.py
    │   │   └── visualization.py
    │   │
    │   └── molmo_motion/
    │
    ├── fixtures/
    │   ├── davis_bmx_trees/
    │   ├── fmb_wrist_1/
    │   ├── fmb_wrist_2/
    │   ├── berkeley_bottle/
    │   ├── berkeley_cup/
    │   ├── dobbe_pure_vipe/
    │   └── dobbe_hybrid/
    │
    ├── tests/
    │   ├── golden_manifest.json
    │   ├── test_inputs.py
    │   ├── test_regression.py
    │   ├── test_outputs.py
    │   └── test_visualizations.py
    │
    └── docs/
        ├── experiments.md
        ├── provenance.md
        └── verification.json

Exact filenames may differ.

The important property is:

> Final experiment artifacts are organized by experiment, not by the historical
> directory structure of the research process.


# 4. Replace `data/legacy` with minimal experiment bundles

The final repository must not require understanding paths such as:

    runs/fmb_wrist_v3/...
    runs/fmb_wrist_v4_improvement/...
    runs/berkeley_ur5_molmomotion/...
    runs/berkeley_ur5_improvement_v1/CASE-AUGE/...
    runs/berkeley_ur5_arc_expansion_v1/...

Those paths describe how the research evolved.

They are useful historically, but they are not the correct public API of the
final repository.

For every supported experiment build one immutable bundle containing ONLY the
information required to:

1. reconstruct the exact MolmoMotion input;
2. reproduce the existing forecast in replay mode;
3. perform fresh inference;
4. calculate the existing metrics;
5. generate the existing visualizations;
6. prove provenance/parity.


# 5. Define a small explicit bundle schema

Each bundle should be self-describing.

For example:

    fixtures/fmb_wrist_1/
        manifest.json
        observed.npz
        evaluation.npz
        legacy_prediction.npy
        processor_fingerprints.json
        media/
            future.*
            legacy_comparison.*

`manifest.json` should contain fields such as:

    schema_version
    dataset
    episode_id
    status
    action

    source_fps
    model_fps

    geometry_description
    reference_description
    limitation

    expected_prediction_sha256
    expected_status

    original_source_paths
    original_source_sha256

Do not put values into the manifest merely because they are available.

The schema should remain small.


# 6. Canonical observed input

The bundle must contain the exact causal model input:

    history_frames
    history_timestamps
    points_2d
    points_3d_history
    point_ids

and, where applicable:

    camera_intrinsics
    c2w_at_t0

The model input reconstructed from the new bundle must be numerically identical
to the baseline repository.

Future information must remain physically separated from observed input.


# 7. Evaluation bundle

Evaluation data should be stored separately from observed input.

For example:

    evaluation.npz

may contain:

    future_times
    reference_xyz
    reference_uv
    mask_2d
    mask_3d
    evaluation_camera_poses

depending on the experiment.

Fields that do not exist for a dataset must remain absent.

Do NOT invent:

- physical K for Dobb-E;
- confirmed 3D GT where it does not exist;
- camera poses for DAVIS where they are unavailable.


# 8. Do not preserve unnecessary intermediate research artifacts

For every file currently under `data/legacy`, answer:

    Is this file actually necessary for:
    - inference?
    - replay?
    - evaluation?
    - visualization?
    - provenance/regression verification?

If all answers are "no", it must not be copied into the final fixture set.

Examples of things that should generally NOT remain just because they existed:

- abandoned intermediate variants;
- temporary diagnostic arrays;
- duplicated rendered images;
- old scratch outputs;
- intermediate depth arrays no longer consumed by the final experiment;
- duplicate copies of history RGB;
- entire previous experiment folders when only several arrays are needed.

Never delete the historical source branch.

We are pruning only the final distributable repository representation.


# 9. Remove physical duplication of golden data

Delete the architecture:

    data/legacy/
    tests/golden/source/

where the same bytes are stored twice.

Instead use:

    fixtures/
        # single canonical artifact copy

and:

    tests/golden_manifest.json
        # expected hashes / values / statuses

Golden verification must validate the ONE fixture copy against immutable
expected SHA-256 values.


# 10. Golden references must contain assertions, not a cloned dataset

A regression reference should preferably contain:

- SHA-256;
- shape;
- dtype;
- small expected arrays;
- expected scalar metrics;
- expected experiment status;
- expected method names;
- expected masks/statistics;
- processor fingerprints.

Do not duplicate large:

- RGB arrays;
- depth arrays;
- MP4;
- PNG;
- processor `.pt`;

merely for regression testing.


# 11. Replace stored processor packets with deterministic fingerprints where safe

The current implementation stores full legacy `processor_inputs.pt` packets
for many P8 groups and compares the complete structures.

Preserve the strength of this check without unnecessarily storing all large
serialized packets.

Create a deterministic field-wise fingerprint representation.

For every processor field record, as applicable:

    key
    type
    dtype
    shape
    SHA-256 of raw tensor/array bytes

For nested structures fingerprint them recursively.

For scalar/string values preserve their exact value.

Example:

    {
      "input_ids": {
        "dtype": "int64",
        "shape": [...],
        "sha256": "..."
      },
      ...
    }

Then:

    new processor output
        ↓
    canonical fingerprint
        ↓
    compare with frozen legacy fingerprint

This must be EXACT.

Do not replace exact processor parity with a weak approximate check.

If a field cannot be fingerprinted reliably, keep the original artifact for
that field and document why.


# 12. Preserve prediction parity

For all six successful neural forecasts:

    fresh prediction ≈ baseline prediction

must retain the existing strict tolerance:

    atol = 1e-7 m
    rtol = 0

Record:

    max_difference_m

as before.

Do not loosen tolerance to make the refactor pass.


# 13. Preserve metric parity

All currently verified metrics must remain equal to the baseline.

This includes where applicable:

- ADE;
- FDE;
- median;
- P90;
- PWT;
- errors by time;
- errors by point;
- outside-image fraction;
- trajectory direction diagnostics;
- speed error;
- path-length ratio;
- endpoint displacement ratio;
- rigidity/pair-distance diagnostics;
- static baseline;
- constant-velocity baseline;
- object-translation baseline;
- FMB-specific methods;
- Berkeley-specific variants.

Do not silently redefine a metric.


# 14. Preserve all experiment-specific semantics

This packaging task must NOT normalize away scientifically important
differences.

DAVIS:

- keep common first-camera 3D interpretation;
- do not introduce false video-frame 2D metrics.

FMB:

- retain 10 Hz historical timing;
- retain selected motion policy;
- retain bounded Molmo correction;
- retain estimated geometry limitations.

Berkeley:

- retain 5 Hz historical timing;
- retain existing bottle variants;
- retain existing cup `original_physical`;
- do not turn exploratory postprocessing into a blind-test result.

Dobb-E:

- `pure_vipe` stays conditional 2D;
- no confirmed 3D GT may be claimed;
- `hybrid` stays rejected by the real geometry gate.

The packaging refactor must not alter scientific interpretation.


# 15. Keep the unified adapters, but simplify them

The existing adapter idea is good:

    davis.py
    fmb.py
    berkeley.py
    dobbe.py

Keep that separation.

However, after fixture canonicalization adapters should no longer know dozens
of historical research paths.

Ideal adapter:

    bundle -> ExperimentSample

plus the existing dataset-specific temporal/geometric `finish()` logic.

An adapter should describe dataset semantics, not archaeological filesystem
history.


# 16. Keep `ExperimentSample`

Do NOT replace the existing simple dataclass with a framework.

`ExperimentSample` is already approximately the right abstraction.

It may be adjusted slightly to fit the fixture schema, but avoid:

- abstract base-class hierarchies;
- plugin registries;
- dependency injection frameworks;
- Hydra;
- database layers;
- unnecessary Pydantic models.

This repository contains seven fixed research cases.

Prefer explicit code.


# 17. Keep the upstream MolmoMotion source vendored for now

Do NOT optimize repository size by removing:

    src/molmo_motion/

and dynamically installing an arbitrary new upstream version.

The current repository verifies that this source is unchanged from the pinned
author version.

It is small compared with the experimental artifacts and provides strong
reproducibility.

Preserve:

    docs/model_source_snapshot.json

or an equivalent compact immutable provenance manifest.


# 18. Simplify `legacy/`

The final runtime must not depend on:

    legacy/

The current regression suite imports several old functions from this folder to
prove parity.

Replace those dependencies carefully.

Procedure:

1. run current regression tests;
2. freeze their expected outputs/hashes;
3. implement equivalent assertions using the final package + golden manifest;
4. run both old and new tests;
5. only after equality is demonstrated, remove `legacy/` from the final runtime
   repository.

Do not simply delete the tests using legacy code.

The verification must become independent from the old Python files, not weaker.


# 19. Simplify one-time maintenance scripts

Current one-off scripts such as inventory/capture tools were useful while the
repository was being built.

Classify each as:

    runtime
    verification
    migration-only

Migration-only scripts do not belong in the primary user workflow.

Either:

- move them under `tools/migration/`;
- or remove them from the final branch after their outputs are frozen.

The README should not expose these as normal usage.


# 20. Compress large lossless arrays where it is safe

Large RGB arrays currently dominate fixture size.

Investigate lossless representations only.

Allowed examples:

- `np.savez_compressed`;
- lossless PNG frame sequences;
- another deterministic lossless format.

Not allowed:

- JPEG replacement when exact RGB matters;
- lossy video recompression if pixel-level parity is checked;
- changed resizing/cropping.

For every changed representation:

    decode(new representation) == original array

must be asserted exactly.

Choose the smaller representation only after measuring it.

Do not optimize size at the cost of data equality.


# 21. Do not mix generated outputs with immutable fixtures

Repository:

    fixtures/

contains immutable experiment inputs/references.

Runtime:

    outputs/

contains generated runs and remains gitignored.

No command may modify fixtures.

Every run must continue to create a unique:

    outputs/<run_id>/

directory.


# 22. Preserve provenance

Removing historical directory structure must NOT remove provenance.

For every packed artifact preserve:

    original path
    original SHA-256
    new path
    new SHA-256
    transformation, if any

Create one compact document:

    docs/provenance.json

or equivalent.

If representation is unchanged:

    original SHA == packed SHA

If a lossless storage transformation is performed:

record both the container SHA and the reconstructed content SHA.


# 23. Repository-size report

Before modifying files, calculate:

- number of files;
- logical working-tree size;
- size by top-level directory;
- duplicate content by SHA-256;
- largest 20 artifacts.

Repeat after refactoring.

Save:

    docs/packaging_report.md

It must show:

    BEFORE
    AFTER
    saved bytes
    saved percentage
    what was removed/deduplicated/compressed

No arbitrary size target.

Correctness has priority over size.


# 24. Backward-compatible CLI

Existing user commands must continue to work.

For example:

    molmo-motion-experiment --config configs/fmb_wrist_1.json
    molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json
    molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json
    molmo-motion-experiment --config configs/author_davis.json

and:

    --mode replay
    --mode inference
    --predictions-from

Do not require users to learn a new interface merely because storage changed.

Internally configs may point to new fixture bundles.


# 25. Preserve output contract

A successful run must continue to produce the same conceptual artifacts:

    config.json
    inputs/
    validation.json
    model_input_parity.json
    predictions/
    prediction_parity.json
    metrics.json
    errors.npz
    evaluation/
    metadata.json
    status.json
    visualizations/
    index.html

Exact internal filenames may only be changed if there is a strong reason and
the README/tests are updated.

Prefer not to change them.


# 26. Verification procedure

Before refactor save baseline results from commit:

    70ed2c413eee5e3b9dc1324e2c87ff333ee88334

Then perform THREE levels of verification.


## Level A — unit/regression tests

Baseline currently reports:

    51 tests passing

The new repository must pass an equivalent or stronger suite.

Removing a test is not evidence of success.


## Level B — replay all seven configs

Run from outside the repository working directory, as the current repository
already verifies:

    all 7 configs

Expected:

    6 COMPLETE
    1 SKIPPED_GEOMETRY_GATE

Compare:

- canonical input;
- prediction;
- metrics;
- generated plotting coordinates;
- expected status.


## Level C — fresh model inference

Run all successful neural cases using the same checkpoint:

    allenai/MolmoMotion-4B-H3-F30
    revision:
    3f5e790a511ff2cdf21c8d2a14cb4d8409c94629

The six neural forecasts must retain the current parity result.

Do not accept replay-only verification as proof that inference was preserved.


# 27. Visual regression

Preserve the scientific content of the existing visualizations.

At minimum verify:

- number of rendered frames;
- image/video dimensions;
- selected point IDs;
- projected point coordinates;
- trajectory coordinates;
- method names;
- shown point subset;
- reference/prediction association.

Produce temporary before/after comparisons during migration.

They do not all need to remain committed after PASS.


# 28. README after cleanup

The final README should become SHORTER than the current internal research
documentation.

A reviewer should see:

1. what the repository demonstrates;
2. the seven examples;
3. installation;
4. checkpoint download;
5. one inference command;
6. one replay command;
7. where outputs appear;
8. known scientific limitations;
9. how to run tests.

Detailed migration/inventory information belongs in `docs/`, not the main
README.


# 29. Do not rewrite Git history in this task

This GOAL concerns the final working tree and package architecture.

Do NOT run destructive operations such as:

    git filter-repo
    git lfs migrate
    force-pushing rewritten shared history

as part of this refactor.

First produce a correct compact final tree.

Repository-history cleanup, if needed for final publication, is a separate
explicit task.


# 30. Things explicitly forbidden

Do NOT:

- rerun depth estimation;
- rerun AllTracker;
- recalibrate cameras;
- change K;
- change camera poses;
- change hand-eye transforms;
- change temporal alignment;
- resample history;
- change prompts;
- change point selection;
- change seeds;
- change decoder;
- change postprocessing coefficients;
- choose a different Berkeley winner;
- choose a different FMB policy;
- turn Dobb-E failure into success;
- loosen parity tolerances;
- use future data in model preparation;
- introduce a new experiment framework;
- optimize the model itself.


# 31. Definition of DONE

The task is PASS only when ALL are true:

[x] Current baseline commit has been recorded.

[x] Large artifacts exist only once in the final working tree.

[x] `tests/golden/source` no longer duplicates the fixture dataset.

[x] Historical research paths have been replaced by compact experiment bundles.

[x] Every bundle has provenance and immutable hashes.

[x] All seven existing configs still run.

[x] Existing CLI syntax still works.

[x] `replay` still works.

[x] `inference` still works.

[x] `--predictions-from` still works.

[x] Successful forecasts preserve `atol=1e-7 m, rtol=0` parity.

[x] Existing metrics reproduce baseline values.

[x] Baselines reproduce baseline values.

[x] Important visualization coordinates reproduce baseline values.

[x] Dobb-E hybrid still exits as expected before inference/evaluation.

[x] No future information enters model input.

[x] Tests are equivalent to or stronger than the previous 51-test suite.

[x] Runtime no longer depends on `legacy/`.

[x] The vendored MolmoMotion source remains unchanged.

[x] Generated outputs remain outside immutable fixture data.

[x] README documents only the final public workflow.

[x] `docs/packaging_report.md` contains measured before/after repository sizes.


# Final target

The repository should stop looking like:

    "a cleaned wrapper around a copied research directory"

and look like:

    config
       ↓
    immutable experiment bundle
       ↓
    small dataset adapter
       ↓
    one common MolmoMotion pipeline
       ↓
    metrics + visualizations + reproducible outputs


The final repository must contain enough information to reproduce the verified
experiments, but not the filesystem history of how those experiments were
discovered.

Correctness is defined by behavioral parity with commit
70ed2c413eee5e3b9dc1324e2c87ff333ee88334.

# 32. Mandatory full end-to-end rerun after refactor

After ALL packaging/refactoring changes are complete, perform a clean end-to-end verification of the final repository.

This step is mandatory.

Do not rely only on:
- unit tests;
- replay mode;
- previously generated outputs;
- cached success receipts.

Run the final repository from the new packaged layout.

Required sequence:

1. Install/use the final package.
2. Run:

       pytest -q

3. Run replay for ALL seven configs.

4. Run fresh MolmoMotion inference for ALL six successful neural cases:

       author_davis
       fmb_wrist_1
       fmb_wrist_2
       berkeley_bottle
       berkeley_cup
       dobbe

5. Run:

       dobbe_blocked

   and verify that it stops at the expected geometry gate WITHOUT inference.

6. Recompute from the newly produced forecasts:
   - aligned trajectories;
   - baselines;
   - metrics;
   - plots;
   - videos;
   - HTML pages;
   - run metadata/status.

7. Compare the newly generated results against the baseline commit
   `70ed2c413eee5e3b9dc1324e2c87ff333ee88334`.

Fresh inference verification must include:

    prediction parity
    metric parity
    baseline parity
    visualization-coordinate parity
    expected experiment status

For successful model runs require:

    atol = 1e-7 m
    rtol = 0

for neural forecast parity.

Do not reuse old prediction files as the final proof of correctness.

The final proof must come from NEW inference outputs generated by the refactored repository.

Save a final machine-readable report:

    docs/final_verification.json

and a concise human-readable summary:

    docs/final_verification.md

The report must contain for every config:

    config
    status
    fresh_inference
    prediction_max_difference_m
    metrics_match
    visualization_geometry_match
    runtime
    peak_cuda_memory
    PASS / FAIL

The entire GOAL is FAIL if this final end-to-end rerun is not completed successfully.

