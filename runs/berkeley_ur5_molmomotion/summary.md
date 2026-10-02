# Berkeley UR5 → MolmoMotion: две реальные сцены

Эксперимент использует внешнюю камеру `observation.images.image`, исходную метрическую RealSense depth и intrinsics, оценённые UniDepthV2 только по наблюдаемым кадрам. Прогнозы получены реальными вызовами `allenai/MolmoMotion-4B-H3-F30`; все 30 шагов сохранены.

Основной пример для отчёта — **cup**. Выбор зафиксирован до predictions: видимый рисунок на чашке, открытая боковая поверхность, отдельное положение целевой brown cup и более простая геометрия движения помогают проверять соответствия. У bottle меньше текстуры, есть вращение и частичная окклюзия захватом. Ошибка модели не использовалась для выбора сцены или эпизода. [Протокол](analysis_protocol.json)

## Фактические результаты

| Сцена | Episode | t0, кадр / с | Points / chunks | Prediction | ADE 2D, px | FDE 2D, px | Coverage 2D | ADE 3D_est, м | FDE 3D_est, м |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| cup | 10 | 63 / 12.6 | 24 / 3 | 24×30×3 | 193.571 | 224.485 | 100.0% | 0.2222 | 0.2764 |
| bottle | 9 | 47 / 9.4 | 24 / 3 | 24×30×3 | 223.409 | 194.015 | 100.0% | 0.2546 | 0.2381 |

`Coverage` — доля валидных пар point×future time среди всех выбранных points и 10 реальных будущих кадров. `GT` здесь означает независимый tracker-based reference: AllTracker запускается после замораживания predictions на t0 и реальном продолжении. Это не ручная разметка и не инструментально измеренная траектория материальных точек. Ошибки трекера, особенно на гладкой поверхности bottle, могут влиять на численные результаты.

## Сопоставление с бесплатными baselines

| Сцена | Метод | ADE 2D, px | FDE 2D, px | ADE 3D_est, м | FDE 3D_est, м |
|---|---|---:|---:|---:|---:|
| cup | MolmoMotion | 193.571 | 224.485 | 0.2222 | 0.2764 |
| cup | static | 46.114 | 83.608 | 0.0583 | 0.1032 |
| cup | constant_velocity | 2.906 | 7.374 | 0.0143 | 0.0258 |
| bottle | MolmoMotion | 223.409 | 194.015 | 0.2546 | 0.2381 |
| bottle | static | 60.809 | 102.199 | 0.0677 | 0.1138 |
| bottle | constant_velocity | 11.045 | 11.934 | 0.0422 | 0.0974 |

В обеих сценах **MolmoMotion хуже static и constant velocity по ADE и FDE в 2D и 3D_est**. В этих двух примерах модель не улучшает простые baselines. Различие частот истории 5 Hz и обучения 15 Hz — ограничение эксперимента; его причинный вклад в ошибку здесь не установлен. Основной пример cup выбран заранее по качеству данных, до получения predictions и метрик.

Static сохраняет XYZ на t0. Constant velocity оценивает скорость линейной регрессией по H3 с реальными временными метками, продолжает XYZ и проецирует через тот же K. Все методы используют одну visibility mask; выход predictions за изображение не исключается из ошибки. Для 3D_est общая маска дополнительно требует валидной native depth в будущем. Название 3D_est сохранено, поскольку K оценена и соответствия получены трекером.

## CUP

Instruction: **Pick up the blue cup and put it into the brown cup.**

Источник: episode **10**, t0 **63**, timestamp **12.600000 s**. H3 indices: `[61, 62, 63]`; timestamps: `[12.199999809265137, 12.399999618530273, 12.600000381469727]`. Evaluation indices: `[64, 65, 66, 67, 68, 69, 70, 71, 72, 73]`.

Depth: **RealSense measured aligned metric depth (TFDS float32)**. K: **UniDepthV2 estimated intrinsics, observed robust median**. `fx=510.4755, fy=507.9270, cx=324.2278, cy=245.5075` px.

```text
[510.475464  0.000000  324.227814]
[0.000000  507.927002  245.507492]
[0.000000  0.000000  1.000000]
```

Focal range / median: **3.767%**. [Покадровые K и стабильность](cup/geometry/intrinsics_stability.json); [Проверка неподвижности камеры](cup/geometry/camera_motion_audit.json). Identity poses задают общую систему координат OpenCV camera XYZ в метрах; extrinsics робота не подменяют камеру.

Валидных future pairs: **240 / 240**; по горизонтам: `[24, 24, 24, 24, 24, 24, 24, 24, 24, 24]`. Количество predicted nonpositive Z: **0**. Outside-image predictions по горизонтам: `[0, 0, 0, 7, 14, 13, 11, 7, 6, 6]`.

| Group | Parse status | Shape | Prediction, s | Peak allocated GPU, GiB | Raw output / arrays |
|---|---|---|---:|---:|---|
| group_00 | FULL_8x30x3 | `[8, 30, 3]` | 165.292 | 9.644 | [text](cup/predictions/group_00/raw_model_output.txt); [NPZ](cup/predictions/group_00/prediction.npz); [H3 XYZ](cup/groups/group_00/points_3d_history.npy); [t0 UV](cup/groups/group_00/points_2d_at_t0.npy) |
| group_01 | FULL_8x30x3 | `[8, 30, 3]` | 145.093 | 9.627 | [text](cup/predictions/group_01/raw_model_output.txt); [NPZ](cup/predictions/group_01/prediction.npz); [H3 XYZ](cup/groups/group_01/points_3d_history.npy); [t0 UV](cup/groups/group_01/points_2d_at_t0.npy) |
| group_02 | FULL_8x30x3 | `[8, 30, 3]` | 160.627 | 9.656 | [text](cup/predictions/group_02/raw_model_output.txt); [NPZ](cup/predictions/group_02/prediction.npz); [H3 XYZ](cup/groups/group_02/points_3d_history.npy); [t0 UV](cup/groups/group_02/points_2d_at_t0.npy) |

Model: `allenai/MolmoMotion-4B-H3-F30`; checkpoint revision `3f5e790a511ff2cdf21c8d2a14cb4d8409c94629`; code commit `9810ce220ec1de9ca9f1ef7fce433043ed346a52`; config SHA256 `7e518af747f12fec9e1a249209380330207adf11aad8475a7362671b6ff912a0`. Decoding: official default greedy, dtype `bfloat16`, seed `0`.

| Этап | Время, s | Peak GPU allocated, GiB |
|---|---:|---:|
| MolmoPoint grounding | 3.019 | 9.255 |
| SAM2.1 segmentation | 38.676 | недоступно |
| UniDepthV2 / K | 3.909 | 0.396 |
| Observed AllTracker | 2.330 | 1.881 |
| Observed lift/filter/smooth | 0.315 | недоступно |
| Evaluation AllTracker | 2.420 | 1.888 |
| MolmoMotion: сумма forward calls | 471.011 | 9.656 |
| Evaluation + визуализации | 7.079 | не измерялось |

Загрузка MolmoMotion: **67.782 s**; вес модели загружается один раз для нескольких сцен, поэтому это общее время нельзя суммировать по сценам. Peak reserved CUDA для сцены: **10.590 GiB**.

Визуальные материалы:

- [MolmoPoint grounding на t0](cup/observed/molmopoint_overlay.png)
- [SAM2.1 object mask](cup/observed/mask.png)
- [SAM2.1 mask overlay](cup/observed/mask_overlay.png)
- [history_contact_sheet.png](cup/viz/history_contact_sheet.png)
- [mask_and_100_queries.png](cup/viz/mask_and_100_queries.png)
- [selected_24_points.png](cup/viz/selected_24_points.png)
- [depth_t0.png](cup/viz/depth_t0.png)
- [intrinsics_stability.png](cup/viz/intrinsics_stability.png)
- [trajectory_3d.png](cup/viz/trajectory_3d.png)
- [final_overlay_t0.png](cup/viz/final_overlay_t0.png)
- [error_by_time.png](cup/viz/error_by_time.png)
- [visibility_coverage.png](cup/viz/visibility_coverage.png)
- [trajectory_2d_full_extent.png](cup/viz/trajectory_2d_full_extent.png)
- [pred_vs_gt_2d.mp4](cup/viz/pred_vs_gt_2d.mp4)
- [side_by_side.mp4](cup/viz/side_by_side.mp4)

Контрольные кадры из итоговых видео: [video_audit.png](cup/evaluation/video_audit.png).

Числа и исходные массивы: [metrics.json](cup/metrics.json); [all predictions NPZ](cup/predictions/prediction.npz); [observed tracks](cup/observed/observed_tracks_2d.npz); [independent future tracks](cup/evaluation/evaluation_tracks_2d.npz); [evaluation NPZ](cup/evaluation/evaluation_results.npz); [future 3D_est reference](cup/evaluation/ground_truth_3d_est.npz); [grounding metadata](cup/observed/grounding_metadata.json); [MolmoPoint metadata](cup/observed/molmopoint_grounding.json).

Предупреждения из фактического metrics.json:

- H3 uses real adjacent 5 FPS frames [-0.4,-0.2,0] seconds; temporal distribution shift from training at 15 FPS.
- Future tracking is a tracker-based reference, not manually measured physical point ground truth.
- Berkeley history uses real 5 FPS frames; MolmoMotion training/inference forecast is 15 FPS.
- GT is independently inferred AllTracker correspondence, not human-labeled or instrumented motion ground truth.
- Common visibility mask uses tracker channel 0 > 0.5, finite in-image GT; errors retain out-of-image predictions.
- Camera intrinsics are estimated from observed RGB, not published calibration.
- GT tracks may drift during gripper/object/container occlusion; coverage and video require joint interpretation.

## BOTTLE

Instruction: **Put the ranch bottle into the pot.**

Источник: episode **9**, t0 **47**, timestamp **9.400000 s**. H3 indices: `[45, 46, 47]`; timestamps: `[9.0, 9.199999809265137, 9.399999618530273]`. Evaluation indices: `[48, 49, 50, 51, 52, 53, 54, 55, 56, 57]`.

Depth: **RealSense measured aligned metric depth (TFDS float32)**. K: **UniDepthV2 estimated intrinsics, observed robust median**. `fx=485.1409, fy=481.4539, cx=324.3864, cy=245.7472` px.

```text
[485.140930  0.000000  324.386353]
[0.000000  481.453888  245.747238]
[0.000000  0.000000  1.000000]
```

Focal range / median: **10.387%**. [Покадровые K и стабильность](bottle/geometry/intrinsics_stability.json); [Проверка неподвижности камеры](bottle/geometry/camera_motion_audit.json). Identity poses задают общую систему координат OpenCV camera XYZ в метрах; extrinsics робота не подменяют камеру.

Валидных future pairs: **240 / 240**; по горизонтам: `[24, 24, 24, 24, 24, 24, 24, 24, 24, 24]`. Количество predicted nonpositive Z: **0**. Outside-image predictions по горизонтам: `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`.

| Group | Parse status | Shape | Prediction, s | Peak allocated GPU, GiB | Raw output / arrays |
|---|---|---|---:|---:|---|
| group_00 | FULL_8x30x3 | `[8, 30, 3]` | 155.561 | 9.654 | [text](bottle/predictions/group_00/raw_model_output.txt); [NPZ](bottle/predictions/group_00/prediction.npz); [H3 XYZ](bottle/groups/group_00/points_3d_history.npy); [t0 UV](bottle/groups/group_00/points_2d_at_t0.npy) |
| group_01 | FULL_8x30x3 | `[8, 30, 3]` | 152.404 | 9.648 | [text](bottle/predictions/group_01/raw_model_output.txt); [NPZ](bottle/predictions/group_01/prediction.npz); [H3 XYZ](bottle/groups/group_01/points_3d_history.npy); [t0 UV](bottle/groups/group_01/points_2d_at_t0.npy) |
| group_02 | FULL_8x30x3 | `[8, 30, 3]` | 160.042 | 9.649 | [text](bottle/predictions/group_02/raw_model_output.txt); [NPZ](bottle/predictions/group_02/prediction.npz); [H3 XYZ](bottle/groups/group_02/points_3d_history.npy); [t0 UV](bottle/groups/group_02/points_2d_at_t0.npy) |

Model: `allenai/MolmoMotion-4B-H3-F30`; checkpoint revision `3f5e790a511ff2cdf21c8d2a14cb4d8409c94629`; code commit `9810ce220ec1de9ca9f1ef7fce433043ed346a52`; config SHA256 `7e518af747f12fec9e1a249209380330207adf11aad8475a7362671b6ff912a0`. Decoding: official default greedy, dtype `bfloat16`, seed `0`.

| Этап | Время, s | Peak GPU allocated, GiB |
|---|---:|---:|
| MolmoPoint grounding | 1.678 | 9.239 |
| SAM2.1 segmentation | 8.832 | недоступно |
| UniDepthV2 / K | 3.086 | 0.396 |
| Observed AllTracker | 0.682 | 1.882 |
| Observed lift/filter/smooth | 0.281 | недоступно |
| Evaluation AllTracker | 2.171 | 1.888 |
| MolmoMotion: сумма forward calls | 468.006 | 9.654 |
| Evaluation + визуализации | 7.204 | не измерялось |

Загрузка MolmoMotion: **67.782 s**; вес модели загружается один раз для нескольких сцен, поэтому это общее время нельзя суммировать по сценам. Peak reserved CUDA для сцены: **10.586 GiB**.

Визуальные материалы:

- [MolmoPoint grounding на t0](bottle/observed/molmopoint_overlay.png)
- [SAM2.1 object mask](bottle/observed/mask.png)
- [SAM2.1 mask overlay](bottle/observed/mask_overlay.png)
- [history_contact_sheet.png](bottle/viz/history_contact_sheet.png)
- [mask_and_100_queries.png](bottle/viz/mask_and_100_queries.png)
- [selected_24_points.png](bottle/viz/selected_24_points.png)
- [depth_t0.png](bottle/viz/depth_t0.png)
- [intrinsics_stability.png](bottle/viz/intrinsics_stability.png)
- [trajectory_3d.png](bottle/viz/trajectory_3d.png)
- [final_overlay_t0.png](bottle/viz/final_overlay_t0.png)
- [error_by_time.png](bottle/viz/error_by_time.png)
- [visibility_coverage.png](bottle/viz/visibility_coverage.png)
- [trajectory_2d_full_extent.png](bottle/viz/trajectory_2d_full_extent.png)
- [pred_vs_gt_2d.mp4](bottle/viz/pred_vs_gt_2d.mp4)
- [side_by_side.mp4](bottle/viz/side_by_side.mp4)

Контрольные кадры из итоговых видео: [video_audit.png](bottle/evaluation/video_audit.png).

Числа и исходные массивы: [metrics.json](bottle/metrics.json); [all predictions NPZ](bottle/predictions/prediction.npz); [observed tracks](bottle/observed/observed_tracks_2d.npz); [independent future tracks](bottle/evaluation/evaluation_tracks_2d.npz); [evaluation NPZ](bottle/evaluation/evaluation_results.npz); [future 3D_est reference](bottle/evaluation/ground_truth_3d_est.npz); [grounding metadata](bottle/observed/grounding_metadata.json); [MolmoPoint metadata](bottle/observed/molmopoint_grounding.json).

Предупреждения из фактического metrics.json:

- H3 uses real adjacent 5 FPS frames [-0.4,-0.2,0] seconds; temporal distribution shift from training at 15 FPS.
- Future tracking is a tracker-based reference, not manually measured physical point ground truth.
- Berkeley history uses real 5 FPS frames; MolmoMotion training/inference forecast is 15 FPS.
- GT is independently inferred AllTracker correspondence, not human-labeled or instrumented motion ground truth.
- Common visibility mask uses tracker channel 0 > 0.5, finite in-image GT; errors retain out-of-image predictions.
- Camera intrinsics are estimated from observed RGB, not published calibration.
- GT tracks may drift during gripper/object/container occlusion; coverage and video require joint interpretation.

## Данные и native depth

Найдены 248 cup и 250 bottle candidates. Визуально проверены cup 4/7/10/12 и bottle 2/5/6/9. Полный список и причины ранжирования: [selection audit](sources/selection_and_depth_audit.json); [все candidates](sources/candidates_all.json).

Для обеих выбранных сцен достаточно одного TFRecord `berkeley_autolab_ur5-train.tfrecord-00004-of-00412` размером **181,825,519 bytes**. Record 0 соответствует bottle9, record 1 — cup10. Все pose/gripper states и translation actions каждого эпизода совпадают с LeRobot точно. Lossless native RGB сравнивался с AV1 RGB на наблюдаемых кадрах; отличия обусловлены видеокодеком. [SHA256 и количественная проверка соответствия](sources/native_observed_receipt.json).

TFDS хранит float32 depth как RGBA PNG с побитовым преобразованием четырёх uint8 каналов в float32. Восстановление выполняется без нормировки и без подгонки масштаба. [Официальный TFDS decoder](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/features/image_feature.py) и локальная проверка exact bitcast round-trip подтверждают формат. [Исходный Berkeley capture code](https://github.com/yunliangchen/ur5bc/blob/a4285610da52cb30215951c7ac37454fcfae01c8/ur5/robot_env.py#L109) умножает aligned RealSense depth на depth_scale для перевода в метры и выравнивает depth к color через `rs.align(rs.stream.color)`. [Сохранённый code provenance](sources/capture_source/receipt.json).

LeRobot `image_with_depth` не использовалось как измеренная depth: его metadata описывает AV1/yuv420p и `video.is_depth_map=false`. Native depth содержит нулевые holes на границах и отдельные дальние/насыщенные значения в фоне. Для XYZ использована медиана валидных значений в окне 5×5. [Cup RGB-D alignment](sources/native_alignment_cup.png); [Bottle RGB-D alignment](sources/native_alignment_bottle.png).

## Causal split, official recipe и отклонения

Observed и evaluation физически разделены. Segmentation, queries, K, input depth, input tracking и smoothing используют только frames ≤t0. Для выбора эпизода/t0 просмотрено продолжение в отдельном data-screening этапе. Независимые future tracking и depth extraction запускаются после реального prediction и проверки сохранённых receipts; hashes frozen inputs проверяются повторно.

- Object phrases взяты из явных пользовательских targets (`blue cup`, `ranch bottle`); Qwen recaption пропущен.
- Реальный MolmoPoint-Vid-4B вызван через native image-pointing API на единственном наблюдаемом external RGB кадре t0 с robot-gripper prompt; видеоконтекст для grounding не используется. Point получен моделью, ручная подстановка отсутствует.
- SAM3 weights вернули HTTP 403 / GatedRepo. По явному выбору пользователя применён официальный Meta SAM2.1 Hiera-Large, prompted настоящим MolmoPoint. [Проверка доступа SAM3](grounding_availability.json).
- N=100 queries получены исходной K-means функцией vendored MolmoMotion pipeline. Из прошедших filtering points выбраны максимум 24 spatially distributed points и полные группы P=8.
- AllTracker — vendored author Net(16), 4 iterations. Размер входа и способ sampling записаны ниже; dense flow bilinearly sampled в точных query positions вместо nearest-pixel rounding авторского CLI.
  - cup: observed network W×H `[512, 384]`, evaluation input `[1, 11, 3, 384, 512]`; outputs возвращены в исходные `[640, 480]` pixels.
  - bottle: observed network W×H `[512, 384]`, evaluation input `[1, 11, 3, 384, 512]`; outputs возвращены в исходные `[640, 480]` pixels.
- Geometry wrapper подаёт native measured depth, observed median K и identity poses; ViPE moving-camera reconstruction не запускается. UniDepth depth сохранена для аудита, основной depth остаётся native.
- Official trust gating/filter/ray-direction smoothing переиспользованы на восьми реальных наблюдаемых кадрах с 16 anchor tracks; модель получает только последние H3. Spatial auto-splitting не требуется для одного grounded object. Robust 5×5 depth lifting — адаптация к holes реального сенсора.
- **Temporal distribution shift:** Berkeley H3 = `[-0.4,-0.2,0] s` при 5 FPS; training convention MolmoMotion = 15 FPS. Кадры не дублировались и не интерполировались. F30 сопоставляется десяти реальным кадрам 0.2…2.0 s по zero-based indices `[2,5,8,11,14,17,20,23,26,29]`; primary GT не интерполируется.
- K не является опубликованной калибровкой. Поэтому даже measured-depth XYZ comparison обозначается **3D_est**; систематическая ошибка K и tracker reference остаются ограничениями.

## Воспроизводимость

Точные версии модели, commits, SHA256 checkpoints/code, raw generated text и frozen-input hashes находятся в receipts каждой сцены. Команды ниже показывают выполненный порядок из WSL в `F:/AIRI_task/molmo-motion`. Ниже создаётся новый уникальный run root; completed run остаётся исходным evidence. Для воспроизведения нужны уже установленные checkpoints и две имеющиеся environments.

```bash
set -e
cd /mnt/f/AIRI_task/molmo-motion
PY=/mnt/f/AIRI_task/.venv/bin/python
PRE=/mnt/f/AIRI_task/.venv-vipe/bin/python
RUN=$(mktemp -d runs/berkeley_ur5_reproduce_XXXXXX)
cp runs/berkeley_ur5_molmomotion/analysis_protocol.json "$RUN/analysis_protocol.json"
cp runs/berkeley_ur5_molmomotion/grounding_availability.json "$RUN/grounding_availability.json"
$PY scripts/berkeley_data_discover.py
$PY scripts/berkeley_data_inspect.py
$PY scripts/berkeley_data_native.py
$PY scripts/berkeley_data_extract_native.py
$PY scripts/berkeley_scene.py cup --t0 63 --run-root "$RUN"
$PY scripts/berkeley_scene.py bottle --t0 47 --run-root "$RUN"
for SCENE in cup bottle; do
  T0=63; [ "$SCENE" = bottle ] && T0=47
  $PY scripts/berkeley_scene.py "$SCENE" --t0 "$T0" --attach-native --run-root "$RUN"
  $PY scripts/berkeley_preprocess.py depth --scene-dir "$RUN/$SCENE"
done
$PY scripts/berkeley_ground_molmopoint.py --scene-dir "$RUN/cup" "$RUN/bottle"
for SCENE in cup bottle; do
  $PRE scripts/berkeley_preprocess.py ground --scene-dir "$RUN/$SCENE" \
    --pointing-json "$RUN/$SCENE/observed/molmopoint_grounding.json" \
    --sam-checkpoint ../models/sam2.1_hiera_large.pt
  $PRE scripts/berkeley_preprocess.py track --scene-dir "$RUN/$SCENE" --max-side 512
  $PRE scripts/berkeley_preprocess.py filter --scene-dir "$RUN/$SCENE"
done
$PY scripts/berkeley_input_audit.py --scene-dir "$RUN/cup" "$RUN/bottle"
$PY scripts/berkeley_infer.py --scene-dir "$RUN/cup" "$RUN/bottle"
$PY scripts/berkeley_scene.py cup --t0 63 --future --run-root "$RUN"
$PY scripts/berkeley_scene.py bottle --t0 47 --future --run-root "$RUN"
$PY scripts/berkeley_data_export_run.py --future --run-root "$RUN"
for SCENE in cup bottle; do
  $PRE scripts/berkeley_evaluate.py "$RUN/$SCENE" --stage track --max-side 512
  $PY scripts/berkeley_evaluate.py "$RUN/$SCENE" --stage evaluate
done
$PY scripts/berkeley_video_audit.py --scene-dir "$RUN/cup" "$RUN/bottle"
$PY scripts/berkeley_verify.py --run-root "$RUN"
$PY scripts/berkeley_report.py --run-root "$RUN"
```

`PY` использован для UniDepth, MolmoPoint, MolmoMotion и CPU evaluation; `PRE` — для SAM2.1, AllTracker и author filtering. Execution receipts фиксируют фактические размеры и версии. Для проверки готовых результатов достаточно `berkeley_verify.py --run-root <completed-run>`; повторная генерация predictions не нужна.

Артефактная проверка: [verification.json](verification.json). Визуальная проверка графиков и декодированных кадров видео: [visual_review.json](visual_review.json). Машиночитаемый отчёт: [summary.json](summary.json).
