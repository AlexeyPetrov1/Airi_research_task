"""Build the final Berkeley report strictly from completed experimental artifacts.

This command never modifies observed/, geometry/, groups/, or predictions/.
It refuses missing final metrics, uncompleted inference, and existing reports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / "data/berkeley_ur5_two_scenes"
DEFAULT_RUN = REPO / "runs/berkeley_ur5_molmomotion"
PNG_NAMES = ["history_contact_sheet", "mask_and_100_queries", "selected_24_points",
             "depth_t0", "intrinsics_stability", "trajectory_3d", "final_overlay_t0", "error_by_time",
             "visibility_coverage", "trajectory_2d_full_extent"]
VIDEO_NAMES = ["pred_vs_gt_2d", "side_by_side"]


def read(path: Path, required: bool = True):
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, digits: int = 3) -> str:
    if value is None:
        return "недоступно"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{float(value):.{digits}f}"


def relative(path: Path, run: Path) -> str:
    return os.path.relpath(path, run).replace("\\", "/")


def link(path: Path, run: Path, title: str | None = None) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return f"[{title or path.name}]({relative(path, run)})"


def optional_link(path: Path, run: Path, title: str | None = None) -> str:
    return link(path, run, title) if path.exists() else "не сохранено"


def scene_bundle(name: str, run: Path) -> dict:
    scene = run / name
    meta = read(scene / "metadata.json")
    metrics = read(scene / "metrics.json")
    model = read(scene / "predictions/model_run.json")
    if model.get("success") is not True:
        raise ValueError(f"Actual completed model inference required: {scene}")
    prediction = np.load(scene / "predictions/future_3d.npy", allow_pickle=False)
    if prediction.shape != tuple(metrics["prediction_shape"]):
        raise ValueError(f"Prediction/metrics shape mismatch: {name}")
    if not np.isfinite(prediction).all() or not np.any(prediction):
        raise ValueError(f"Invalid prediction: {name}")
    if prediction.shape[1:] != (30, 3) or prediction.shape[0] not in (8, 16, 24):
        raise ValueError(f"Incomplete horizon or point groups: {name}")
    if len(model["groups"]) != prediction.shape[0] // 8 or not all(g.get("success") for g in model["groups"]):
        raise ValueError(f"Incomplete model groups: {name}")
    if metrics["gt_interpolated"] or metrics["evaluation_prediction_indices_zero_based"] != list(range(2, 30, 3)):
        raise ValueError(f"Unexpected physical-time evaluation: {name}")
    k = np.load(scene / "geometry/K_median.npy", allow_pickle=False)
    if not np.allclose(k, metrics["K_median"]):
        raise ValueError(f"K/metrics mismatch: {name}")
    artifacts = [scene / "viz" / f"{n}.png" for n in PNG_NAMES]
    artifacts += [scene / "viz" / f"{n}.mp4" for n in VIDEO_NAMES]
    artifacts += [scene / p for p in ["predictions/prediction.npz", "evaluation/evaluation_results.npz",
                  "evaluation/evaluation_tracks_2d.npz", "observed/observed_tracks_2d.npz",
                  "observed/molmopoint_overlay.png", "observed/mask.png", "observed/mask_overlay.png",
                  "observed/query_points_100.npy"]]
    if (scene / "evaluation/video_audit.png").exists():
        artifacts.append(scene / "evaluation/video_audit.png")
    for path in artifacts:
        if not path.exists() or not path.stat().st_size:
            raise FileNotFoundError(f"Required final artifact absent/empty: {path}")
    return {"scene": name, "metadata": meta, "metrics": metrics, "model": model,
            "grounding": read(scene / "observed/grounding_metadata.json"),
            "pointing": read(scene / "observed/molmopoint_grounding.json"),
            "filtering": read(scene / "geometry/filter_metadata.json"),
            "geometry": read(scene / "geometry/depth_source.json"),
            "observed_tracking": read(scene / "observed/alltracker_execution.json"),
            "evaluation_tracking": read(scene / "evaluation/alltracker_execution.json"),
            "camera_motion": read(scene / "geometry/camera_motion_audit.json"),
            "intrinsics": read(scene / "geometry/intrinsics_stability.json"),
            "source_metric_sha256": digest(scene / "metrics.json"),
            "prediction_sha256": digest(scene / "predictions/future_3d.npy"),
            "artifacts": [relative(p, run) for p in artifacts]}


def build(run: Path):
    protocol = read(run / "analysis_protocol.json")
    if protocol.get("protocol_fixed_before_predictions") is not True or protocol.get("primary_scene") != "cup":
        raise ValueError("Expected preregistered cup primary-scene selection")
    source = read(DATA / "native_observed_receipt.json")
    selection = read(DATA / "selection_and_depth_audit.json")
    verification = read(run / "verification.json", required=False)
    visual_review = read(run / "visual_review.json", required=False)
    bundles = {name: scene_bundle(name, run) for name in ("cup", "bottle")}
    baseline_comparisons = {}
    for name, bundle in bundles.items():
        metrics = bundle["metrics"]
        comparison = {}
        for dimension, values, keys in (
                ("2D", metrics["methods"], ("ADE_2D_px", "FDE_2D_px")),
                ("3D_est", metrics.get("3D_est") or {}, ("ADE_3D_est_m", "FDE_3D_est_m"))):
            complete = all(values.get(method, {}).get(key) is not None
                           for method in ("MolmoMotion", "static", "constant_velocity") for key in keys)
            comparison[dimension] = (all(values["MolmoMotion"][key] > values[baseline][key]
                                        for baseline in ("static", "constant_velocity") for key in keys)
                                     if complete else None)
        baseline_comparisons[name] = comparison
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "report_generator_sha256": digest(Path(__file__)),
               "primary_scene": "cup", "primary_scene_selected_before_predictions": True,
               "primary_scene_reason": protocol["primary_scene_reason"],
               "model_worse_than_both_baselines_on_ADE_and_FDE": baseline_comparisons,
               "metrics_origin": "Only completed per-scene metrics.json; no synthetic or filled-in measurements",
               "ground_truth_type": "independent AllTracker correspondences on real future RGB, not manual labels",
               "depth_source": "measured native RealSense depth, TFDS float32 lossless decoding",
               "K_source": "UniDepthV2 observed-only robust median",
               "temporal_mismatch": {"observed_fps": 5, "model_nominal_fps": 15,
                                     "H3_relative_times_s": [-0.4, -0.2, 0],
                                     "gt_interpolated": False,
                                     "prediction_indices_zero_based": list(range(2, 30, 3))},
               "source_receipt": source, "selection_audit": selection,
               "protocol": protocol, "verification": verification, "visual_review": visual_review,
               "scenes": bundles}
    lines = ["# Berkeley UR5 → MolmoMotion: две реальные сцены", "",
             "Эксперимент использует внешнюю камеру `observation.images.image`, исходную метрическую "
             "RealSense depth и intrinsics, оценённые UniDepthV2 только по наблюдаемым кадрам. "
             "Прогнозы получены реальными вызовами `allenai/MolmoMotion-4B-H3-F30`; все 30 шагов сохранены.", "",
             "Основной пример для отчёта — **cup**. Выбор зафиксирован до predictions: видимый рисунок на чашке, "
             "открытая боковая поверхность, отдельное положение целевой brown cup и более простая геометрия движения "
             "помогают проверять соответствия. У bottle меньше текстуры, есть вращение и частичная окклюзия захватом. "
             "Ошибка модели не использовалась для выбора сцены или эпизода. " + link(run / "analysis_protocol.json", run, "Протокол"), "",
             "## Фактические результаты", "",
             "| Сцена | Episode | t0, кадр / с | Points / chunks | Prediction | ADE 2D, px | FDE 2D, px | Coverage 2D | ADE 3D_est, м | FDE 3D_est, м |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for name, b in bundles.items():
        m = b["metrics"]
        lines.append(f"| {name} | {m['source_episode_id']} | {m['t0']} / {number(m['t0_timestamp_s'], 1)} "
                     f"| {m['number_of_valid_points']} / {m['number_of_successful_model_chunks']} "
                     f"| {'×'.join(map(str,m['prediction_shape']))} | {number(m['ADE_2D_px'])} | {number(m['FDE_2D_px'])} "
                     f"| {number(100*m['valid_point_coverage'],1)}% | {number(m.get('ADE_3D_est_m'),4)} | {number(m.get('FDE_3D_est_m'),4)} |")
    lines += ["", "`Coverage` — доля валидных пар point×future time среди всех выбранных points и 10 реальных "
              "будущих кадров. `GT` здесь означает независимый tracker-based reference: AllTracker запускается "
              "после замораживания predictions на t0 и реальном продолжении. Это не ручная разметка и не "
              "инструментально измеренная траектория материальных точек. Ошибки трекера, особенно на "
              "гладкой поверхности bottle, могут влиять на численные результаты.", "",
              "## Сопоставление с бесплатными baselines", "",
              "| Сцена | Метод | ADE 2D, px | FDE 2D, px | ADE 3D_est, м | FDE 3D_est, м |",
              "|---|---|---:|---:|---:|---:|"]
    for name, b in bundles.items():
        m = b["metrics"]
        for method in ("MolmoMotion", "static", "constant_velocity"):
            row = m["methods"][method]
            extra = (m.get("3D_est") or {}).get(method, {})
            lines.append(f"| {name} | {method} | {number(row['ADE_2D_px'])} | {number(row['FDE_2D_px'])} "
                         f"| {number(extra.get('ADE_3D_est_m'),4)} | {number(extra.get('FDE_3D_est_m'),4)} |")
    if all(value is True for scene in baseline_comparisons.values() for value in scene.values()):
        lines += ["", "В обеих сценах **MolmoMotion хуже static и constant velocity по ADE и FDE в 2D и 3D_est**. "
                  "В этих двух примерах модель не улучшает простые baselines. Различие частот истории 5 Hz и "
                  "обучения 15 Hz — ограничение эксперимента; его причинный вклад в ошибку здесь не установлен. "
                  "Основной пример cup выбран заранее по качеству данных, до получения predictions и метрик."]
    lines += ["", "Static сохраняет XYZ на t0. Constant velocity оценивает скорость линейной регрессией по H3 "
              "с реальными временными метками, продолжает XYZ и проецирует через тот же K. "
              "Все методы используют одну visibility mask; выход predictions за изображение не исключается из ошибки. "
              "Для 3D_est общая маска дополнительно требует валидной native depth в будущем. Название 3D_est "
              "сохранено, поскольку K оценена и соответствия получены трекером.", ""]
    for name, b in bundles.items():
        scene = run / name
        m, meta, model = b["metrics"], b["metadata"], b["model"]
        k = np.asarray(m["K_median"])
        lines += [f"## {name.upper()}", "", f"Instruction: **{m['task']}**", "",
                  f"Источник: episode **{m['source_episode_id']}**, t0 **{m['t0']}**, timestamp "
                  f"**{number(m['t0_timestamp_s'],6)} s**. H3 indices: `{meta['history_source_indices']}`; "
                  f"timestamps: `{meta['history_timestamps']}`. Evaluation indices: `{meta['future_source_indices']}`.", "",
                  f"Depth: **{m['depth_source']}**. K: **{m['K_source']}**. "
                  f"`fx={k[0,0]:.4f}, fy={k[1,1]:.4f}, cx={k[0,2]:.4f}, cy={k[1,2]:.4f}` px.", "",
                  "```text", *["[" + "  ".join(f"{x:.6f}" for x in row) + "]" for row in k], "```", "",
                  f"Focal range / median: **{number(100*b['intrinsics']['focal_relative_range_max'],3)}%**. "
                  + link(scene / "geometry/intrinsics_stability.json", run, "Покадровые K и стабильность") + "; "
                  + link(scene / "geometry/camera_motion_audit.json", run, "Проверка неподвижности камеры") + ". "
                  "Identity poses задают общую систему координат OpenCV camera XYZ в метрах; extrinsics робота не подменяют камеру.", "",
                  f"Валидных future pairs: **{m['valid_pairs']} / {m['number_of_valid_points']*10}**; "
                  f"по горизонтам: `{m['valid_points_by_horizon']}`. "
                  f"Количество predicted nonpositive Z: **{m['prediction_nonpositive_Z_count']}**. "
                  f"Outside-image predictions по горизонтам: `{m['prediction_outside_image_by_horizon']}`.", "",
                  "| Group | Parse status | Shape | Prediction, s | Peak allocated GPU, GiB | Raw output / arrays |",
                  "|---|---|---|---:|---:|---|"]
        for group in model["groups"]:
            g = group["group"]
            base = scene / "predictions" / g
            lines.append(f"| {g} | {group.get('parse_status','не записано')} | `{group.get('prediction_shape')}` "
                         f"| {number(group.get('prediction_seconds'))} | {number(group.get('peak_cuda_allocated_gib'))} "
                         f"| {link(base/'raw_model_output.txt',run,'text')}; {link(base/'prediction.npz',run,'NPZ')}; "
                         f"{link(scene/'groups'/g/'points_3d_history.npy',run,'H3 XYZ')}; "
                         f"{link(scene/'groups'/g/'points_2d_at_t0.npy',run,'t0 UV')} |")
        lines += ["", "Model: `" + model["model_id"] + "`; checkpoint revision `" + model["checkpoint_hf_revision"]
                  + "`; code commit `" + model["git_commit"] + "`; config SHA256 `" + model["config_sha256"] + "`. "
                  f"Decoding: {model['decoding']}, dtype `{model['dtype']}`, seed `{model['seed']}`.", "",
                  "| Этап | Время, s | Peak GPU allocated, GiB |", "|---|---:|---:|"]
        stages = [("MolmoPoint grounding", b["pointing"]), ("SAM2.1 segmentation", b["grounding"]),
                  ("UniDepthV2 / K", b["geometry"]), ("Observed AllTracker", b["observed_tracking"]),
                  ("Observed lift/filter/smooth", b["filtering"]), ("Evaluation AllTracker", b["evaluation_tracking"])]
        for title, receipt in stages:
            elapsed = receipt.get("runtime_seconds", receipt.get("elapsed_seconds"))
            peak = receipt.get("peak_gpu_memory_gib", receipt.get("peak_cuda_allocated_gib"))
            lines.append(f"| {title} | {number(elapsed)} | {number(peak)} |")
        lines += [f"| MolmoMotion: сумма forward calls | {number(m['timing']['model'])} | {number(m['GPU_memory']['model_peak_gib'])} |",
                  f"| Evaluation + визуализации | {number(m['timing'].get('evaluation_and_visualization_seconds'))} | не измерялось |", "",
                  f"Загрузка MolmoMotion: **{number(model.get('model_load_seconds'))} s**; "
                  "вес модели загружается один раз для нескольких сцен, поэтому это общее время нельзя суммировать по сценам. "
                  f"Peak reserved CUDA для сцены: **{number(model.get('peak_cuda_reserved_gib'))} GiB**.", "",
                  "Визуальные материалы:", "",
                  f"- {link(scene/'observed/molmopoint_overlay.png',run,'MolmoPoint grounding на t0')}",
                  f"- {link(scene/'observed/mask.png',run,'SAM2.1 object mask')}",
                  f"- {link(scene/'observed/mask_overlay.png',run,'SAM2.1 mask overlay')}",
                  *[f"- {link(scene/'viz'/f'{n}.png',run)}" for n in PNG_NAMES],
                  *[f"- {link(scene/'viz'/f'{n}.mp4',run)}" for n in VIDEO_NAMES], "",
                  "Контрольные кадры из итоговых видео: "
                  + optional_link(scene/"evaluation/video_audit.png",run,"video_audit.png") + ".", "",
                  "Числа и исходные массивы: " + "; ".join([
                      link(scene/"metrics.json",run,"metrics.json"),
                      link(scene/"predictions/prediction.npz",run,"all predictions NPZ"),
                      link(scene/"observed/observed_tracks_2d.npz",run,"observed tracks"),
                      link(scene/"evaluation/evaluation_tracks_2d.npz",run,"independent future tracks"),
                      link(scene/"evaluation/evaluation_results.npz",run,"evaluation NPZ"),
                      optional_link(scene/"evaluation/ground_truth_3d_est.npz",run,"future 3D_est reference"),
                      link(scene/"observed/grounding_metadata.json",run,"grounding metadata"),
                      link(scene/"observed/molmopoint_grounding.json",run,"MolmoPoint metadata")]) + ".", "",
                  "Предупреждения из фактического metrics.json:", "",
                  *[f"- {warning}" for warning in m["warnings_limitations"]], ""]
    lines += ["## Данные и native depth", "",
              f"Найдены {len(read(DATA/'candidates_all.json')['cup'])} cup и "
              f"{len(read(DATA/'candidates_all.json')['bottle'])} bottle candidates. "
              "Визуально проверены cup 4/7/10/12 и bottle 2/5/6/9. "
              "Полный список и причины ранжирования: " + link(DATA/"selection_and_depth_audit.json",run,"selection audit")
              + "; " + link(DATA/"candidates_all.json",run,"все candidates") + ".", "",
              "Для обеих выбранных сцен достаточно одного TFRecord "
              f"`berkeley_autolab_ur5-train.tfrecord-00004-of-00412` размером **{source['shard_bytes']:,} bytes**. "
              "Record 0 соответствует bottle9, record 1 — cup10. "
              "Все pose/gripper states и translation actions каждого эпизода совпадают с LeRobot точно. "
              "Lossless native RGB сравнивался с AV1 RGB на наблюдаемых кадрах; отличия обусловлены видеокодеком. "
              + link(DATA/"native_observed_receipt.json",run,"SHA256 и количественная проверка соответствия") + ".", "",
              "TFDS хранит float32 depth как RGBA PNG с побитовым преобразованием четырёх uint8 каналов в float32. "
              "Восстановление выполняется без нормировки и без подгонки масштаба. "
              "[Официальный TFDS decoder](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/features/image_feature.py) "
              "и локальная проверка exact bitcast round-trip подтверждают формат. "
              "[Исходный Berkeley capture code](https://github.com/yunliangchen/ur5bc/blob/a4285610da52cb30215951c7ac37454fcfae01c8/ur5/robot_env.py#L109) "
              "умножает aligned RealSense depth на depth_scale для перевода в метры и выравнивает depth к color через `rs.align(rs.stream.color)`. "
              + link(DATA/"capture_source/receipt.json",run,"Сохранённый code provenance") + ".", "",
              "LeRobot `image_with_depth` не использовалось как измеренная depth: его metadata описывает AV1/yuv420p "
              "и `video.is_depth_map=false`. Native depth содержит нулевые holes на границах и отдельные дальние/насыщенные "
              "значения в фоне. Для XYZ использована медиана валидных значений в окне 5×5. "
              + link(DATA/"native_alignment_cup.png",run,"Cup RGB-D alignment") + "; "
              + link(DATA/"native_alignment_bottle.png",run,"Bottle RGB-D alignment") + ".", "",
              "## Causal split, official recipe и отклонения", "",
              "Observed и evaluation физически разделены. Segmentation, queries, K, input depth, input tracking и smoothing "
              "используют только frames ≤t0. Для выбора эпизода/t0 просмотрено продолжение в отдельном data-screening этапе. "
              "Независимые future tracking и depth extraction запускаются после реального prediction и проверки сохранённых "
              "receipts; hashes frozen inputs проверяются повторно.", "",
              "- Object phrases взяты из явных пользовательских targets (`blue cup`, `ranch bottle`); Qwen recaption пропущен.",
              "- Реальный MolmoPoint-Vid-4B вызван через native image-pointing API на единственном наблюдаемом external RGB кадре t0 с robot-gripper prompt; "
              "видеоконтекст для grounding не используется. Point получен моделью, ручная подстановка отсутствует.",
              "- SAM3 weights вернули HTTP 403 / GatedRepo. По явному выбору пользователя применён официальный "
              "Meta SAM2.1 Hiera-Large, prompted настоящим MolmoPoint. "
              + link(run/"grounding_availability.json",run,"Проверка доступа SAM3") + ".",
              "- N=100 queries получены исходной K-means функцией vendored MolmoMotion pipeline. "
              "Из прошедших filtering points выбраны максимум 24 spatially distributed points и полные группы P=8.",
              "- AllTracker — vendored author Net(16), 4 iterations. Размер входа и способ sampling записаны ниже; "
              "dense flow bilinearly sampled в точных query positions вместо nearest-pixel rounding авторского CLI."]
    for name, b in bundles.items():
        obs, ev = b["observed_tracking"], b["evaluation_tracking"]
        lines.append(f"  - {name}: observed network W×H `{obs['network_size_wh']}`, evaluation input `{ev['input_shape']}`; "
                     f"outputs возвращены в исходные `{obs['native_size_wh']}` pixels.")
    lines += ["- Geometry wrapper подаёт native measured depth, observed median K и identity poses; "
              "ViPE moving-camera reconstruction не запускается. UniDepth depth сохранена для аудита, основной depth остаётся native.",
              "- Official trust gating/filter/ray-direction smoothing переиспользованы на восьми реальных наблюдаемых кадрах "
              "с 16 anchor tracks; модель получает только последние H3. Spatial auto-splitting не требуется для одного grounded object. "
              "Robust 5×5 depth lifting — адаптация к holes реального сенсора.",
              "- **Temporal distribution shift:** Berkeley H3 = `[-0.4,-0.2,0] s` при 5 FPS; training convention MolmoMotion = 15 FPS. "
              "Кадры не дублировались и не интерполировались. F30 сопоставляется десяти реальным кадрам 0.2…2.0 s "
              "по zero-based indices `[2,5,8,11,14,17,20,23,26,29]`; primary GT не интерполируется.",
              "- K не является опубликованной калибровкой. Поэтому даже measured-depth XYZ comparison обозначается **3D_est**; "
              "систематическая ошибка K и tracker reference остаются ограничениями.", "",
              "## Воспроизводимость", "",
              "Точные версии модели, commits, SHA256 checkpoints/code, raw generated text и frozen-input hashes находятся "
              "в receipts каждой сцены. Команды ниже показывают выполненный порядок из WSL в `F:/AIRI_task/molmo-motion`. "
              "Ниже создаётся новый уникальный run root; completed run остаётся исходным evidence. "
              "Для воспроизведения нужны уже установленные checkpoints и две имеющиеся environments.", "",
              "```bash",
              "set -e",
              "cd /mnt/f/AIRI_task/molmo-motion",
              "PY=/mnt/f/AIRI_task/.venv/bin/python",
              "PRE=/mnt/f/AIRI_task/.venv-vipe/bin/python",
              "RUN=$(mktemp -d runs/berkeley_ur5_reproduce_XXXXXX)",
              f"cp {relative(run/'analysis_protocol.json',REPO)} \"$RUN/analysis_protocol.json\"",
              f"cp {relative(run/'grounding_availability.json',REPO)} \"$RUN/grounding_availability.json\"",
              "$PY scripts/berkeley_data_discover.py",
              "$PY scripts/berkeley_data_inspect.py",
              "$PY scripts/berkeley_data_native.py",
              "$PY scripts/berkeley_data_extract_native.py",
              "$PY scripts/berkeley_scene.py cup --t0 63 --run-root \"$RUN\"",
              "$PY scripts/berkeley_scene.py bottle --t0 47 --run-root \"$RUN\"",
              "for SCENE in cup bottle; do",
              "  T0=63; [ \"$SCENE\" = bottle ] && T0=47",
              "  $PY scripts/berkeley_scene.py \"$SCENE\" --t0 \"$T0\" --attach-native --run-root \"$RUN\"",
              "  $PY scripts/berkeley_preprocess.py depth --scene-dir \"$RUN/$SCENE\"",
              "done",
              "$PY scripts/berkeley_ground_molmopoint.py --scene-dir \"$RUN/cup\" \"$RUN/bottle\"",
              "for SCENE in cup bottle; do",
              "  $PRE scripts/berkeley_preprocess.py ground --scene-dir \"$RUN/$SCENE\" \\",
              "    --pointing-json \"$RUN/$SCENE/observed/molmopoint_grounding.json\" \\",
              "    --sam-checkpoint ../models/sam2.1_hiera_large.pt",
              "  $PRE scripts/berkeley_preprocess.py track --scene-dir \"$RUN/$SCENE\" --max-side 512",
              "  $PRE scripts/berkeley_preprocess.py filter --scene-dir \"$RUN/$SCENE\"",
              "done",
              "$PY scripts/berkeley_input_audit.py --scene-dir \"$RUN/cup\" \"$RUN/bottle\"",
              "$PY scripts/berkeley_infer.py --scene-dir \"$RUN/cup\" \"$RUN/bottle\"",
              "$PY scripts/berkeley_scene.py cup --t0 63 --future --run-root \"$RUN\"",
              "$PY scripts/berkeley_scene.py bottle --t0 47 --future --run-root \"$RUN\"",
              "$PY scripts/berkeley_data_export_run.py --future --run-root \"$RUN\"",
              "for SCENE in cup bottle; do",
              "  $PRE scripts/berkeley_evaluate.py \"$RUN/$SCENE\" --stage track --max-side 512",
              "  $PY scripts/berkeley_evaluate.py \"$RUN/$SCENE\" --stage evaluate",
              "done",
              "$PY scripts/berkeley_video_audit.py --scene-dir \"$RUN/cup\" \"$RUN/bottle\"",
              "$PY scripts/berkeley_verify.py --run-root \"$RUN\"",
              "$PY scripts/berkeley_report.py --run-root \"$RUN\"",
              "```", "",
              "`PY` использован для UniDepth, MolmoPoint, MolmoMotion и CPU evaluation; `PRE` — для SAM2.1, "
              "AllTracker и author filtering. Execution receipts фиксируют фактические размеры и версии. "
              "Для проверки готовых результатов достаточно `berkeley_verify.py --run-root <completed-run>`; "
              "повторная генерация predictions не нужна.", "",
              "Артефактная проверка: " + (link(run/"verification.json",run,"verification.json") if verification else "на момент генерации отдельный verification.json отсутствует")
              + ". Визуальная проверка графиков и декодированных кадров видео: "
              + optional_link(run/"visual_review.json",run,"visual_review.json")
              + ". Машиночитаемый отчёт: [summary.json](summary.json).", ""]
    return summary, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run = args.run_root.resolve()
    summary, markdown = build(run)
    if args.check_only:
        print(json.dumps({"ready": True, "scenes": list(summary["scenes"]), "markdown_characters": len(markdown)}))
        return
    for name in ("summary.md", "summary.json"):
        if (run / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {run/name}")
    (run / "summary.md").write_text(markdown, encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary_md": str(run/"summary.md"), "summary_json": str(run/"summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
