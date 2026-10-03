# Final verification

PASS — baseline `70ed2c413eee5e3b9dc1324e2c87ff333ee88334`.

Installed package; all seven replay cases; six fresh neural forecasts; expected pre-inference geometry gate; predictions-from; complete tests.

| Config | Status | Fresh inference | Max difference, m | Metrics | Geometry | Result |
|---|---|---|---:|---|---|---|
| author_davis | COMPLETE | True | 0.0 | True | True | PASS |
| fmb_wrist_1 | COMPLETE | True | 0.0 | True | True | PASS |
| fmb_wrist_2 | COMPLETE | True | 0.0 | True | True | PASS |
| berkeley_bottle | COMPLETE | True | 0.0 | True | True | PASS |
| berkeley_cup | COMPLETE | True | 0.0 | True | True | PASS |
| dobbe | COMPLETE | True | 0.0 | True | True | PASS |
| dobbe_blocked | SKIPPED_GEOMETRY_GATE | False | None | True | True | PASS |

Fresh model calls: 14. Runtime: 2719.8 s.

Tolerance: atol=1e-7 m, rtol=0. All metrics, aligned baselines, errors and visualization coordinates are compared exactly.

Machine-readable details: [final_verification.json](final_verification.json). Logs: `outputs/packaging_final_logs`.

All 77 generated figures also match baseline image dimensions and decoded pixels exactly: [media_verification.json](media_verification.json).

Both complete pytest passes: 61 passed, zero skipped. The expected blocked case creates no forecast, evaluation or metrics. Vendored model, checkpoint bytes and fixture hashes were verified; fixtures remained unchanged.
