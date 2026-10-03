# Research reports

- [FMB wrist v4: improvements and comparison with v3](fmb_wrist_v4_improvement.md) — past-TCP forecast, native H3/H1 controls, unchanged reference, videos and verified audit.
- [FMB wrist v3: geometry and native forecasts](fmb_wrist_v3.md) — four-view diagnostics, wrist-camera geometry, 24-point forecasts and estimated-reference limitations.
- [FMB: official automatic COLMAP on two episodes and two cameras](fmb_colmap_official_baseline.md) — ten native runs, side camera CAD/depth diagnostics, local wrist reconstructions and PINHOLE controls.
- [Dobb·E: official automatic COLMAP baseline](dobbe_colmap_official_baseline.md) — causal 96/96 registered, full sequence fragmented, independent depth and trajectory diagnostics.
- [Dobb·E: corrected COLMAP controls, four reruns and verified match graphs](dobbe_colmap_clean_rerun.md) — clean defaults, frozen foreground masks, causal versus oracle, depth validation.
- [Dobb·E episode 3651: approximate COLMAP/F2-NeRF transform and MolmoMotion run](dobbe_approx_colmap_f2nerf_molmo.md) — user-requested exploratory `(3,8,3)` and partial future RGB-D evaluation.
- [Dobb·E episode 3651 RGB-only COLMAP and measured-depth geometry study](dobbe_rgbd_colmap_study.md) — causal protocol, independent second scene, geometry gate `BLOCKED`.
- [MolmoMotion on FMB: historical 3D geometry and observed 2D future](fmb_geometry_forecast_study.md) — combined two-episode study, figures, videos and reproducibility notes.
- [First FMB / ShareRobot quantitative 2D trial](fmb_quantitative_2d_episode_5201.md).
- [Second raw FMB quantitative 2D trial](fmb_quantitative_2d_second_trial.md).
- [Effective 256×256 camera calibration audit](fmb_effective_k_256_calibration.md).
- [CAD and RGB-derived metric-depth diagnostics](fmb_cad_metric_depth_experiment.md).
- [Alternative FMB calibration checks A–C](fmb_alternative_calibration_abc.md).
- [Independent second FMB scene and export audit](fmb_second_scene_export.md).
- [FMB episode 5201 geometry preflight](sharerobot_fmb_episode_5201.md).
- [ShareRobot transfer and source identification](sharerobot_transfer.md).
- [Dobb·E source-record recovery](dobbe_episode_3651_recovery.md).
- [Dobb·E and PLEX source preflight](plex_dobbe_sharerobot_preflight.md).
- [DAVIS author example](author_davis.md).
- [DAVIS coordinate audit](author_davis_coordinate_audit.md).
- [WorldTrack camera-coordinate experiment](second_episode_geometry.md).

Model outputs and generated media are kept under [`../runs`](../runs). Setup and environment details are in [`../SETUP_DIAGNOSTICS.md`](../SETUP_DIAGNOSTICS.md).

[Published artifacts and reproduction requirements](PUBLICATION.md).

- [FMB v2 — Berkeley-matched pipeline](fmb_v2_berkeley_matched.md) — independent 24-point AllTracker experiment, H3/H1 controls, sensor-depth geometry audits, metrics and videos; FMB v1 retained as a geometry sensitivity study.
