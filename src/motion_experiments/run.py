"""One command: prepare, validate, infer/replay, evaluate, render and save."""
from __future__ import annotations

import argparse
import html
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .datasets import ADAPTERS
from .io import ROOT, read_json, sha, write_json
from .metrics import compute_metrics
from .molmomotion import ModelRunner, build_inputs
from .sample import validate_sample
from .visualization import compare_legacy, error_plots, make_visualizations, trajectory_plots


def save_inputs(sample, destination):
    destination.mkdir(parents=True, exist_ok=True)
    arrays = dict(history_rgb=sample.history_frames, history_timestamps=sample.history_timestamps,
                  points_2d=sample.points_2d, points_3d_history=sample.points_3d_history,
                  point_ids=sample.point_ids, future_times=sample.future_times)
    for name in ("camera_intrinsics", "c2w_at_t0"):
        value = getattr(sample, name)
        if value is not None:
            arrays[name] = value
    np.savez_compressed(destination/"canonical.npz", **arrays)
    write_json(destination/"metadata.json", dict(dataset=sample.dataset, episode_id=sample.episode_id,
                                               action=sample.action, metadata=sample.metadata))


def run_case(config, source_root, destination, checkpoint, mode, runner=None, predictions_from=None):
    start = time.monotonic()
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination/"config.json", config)
    adapter = ADAPTERS[config["dataset"]]
    # Observed preparation does not open evaluation RGB, reference or poses.
    sample = adapter.load(config, source_root, evaluation=False)
    validation = validate_sample(sample)
    save_inputs(sample, destination/"inputs")
    write_json(destination/"validation.json", validation)
    observed_hashes = {str(p.relative_to(source_root)):sha(p) for p in sample.model_input_files}
    write_json(destination/"inputs/freeze.json", observed_hashes)
    if sample.failure_reason:
        write_json(destination/"status.json", dict(validation, success=False, expected_failure=True,
                                                  mode=mode, runtime_seconds=time.monotonic()-start))
        message=sample.metadata.get("failure_detail") or sample.failure_reason
        (destination/"index.html").write_text('<meta charset="utf-8"><h1>'+html.escape(sample.metadata["title"])+': '
            +html.escape(sample.status)+'</h1><p>'+html.escape(message)
            +'</p><p>Forecast and metrics remain blocked.</p>', encoding="utf8")
        print(config["name"], "expected SKIPPED_GEOMETRY_GATE", flush=True)
        return dict(name=config["name"], status=sample.status, expected_failure=True)
    batches, input_checks = build_inputs(sample, checkpoint, destination/"inputs/model")
    write_json(destination/"model_input_parity.json", dict(success=True, group_checks=input_checks,
               scope="full stored processor packet" if sample.legacy_processor_files else
               "stored input_ids SHA-256 and camera/world equivalence; old full packet unavailable"))
    if predictions_from is not None:
        previous = read_json(predictions_from/"status.json")
        if not previous.get("success") or previous["mode"] != "inference":
            raise ValueError("Only a complete actual new inference can be re-rendered")
        if read_json(predictions_from/"inputs/freeze.json") != observed_hashes:
            raise ValueError("Saved inference used different observed inputs")
        prediction_path = predictions_from/"predictions/future_3d.npy"
        if sha(prediction_path) != read_json(predictions_from/"future_access_receipt.json")["prediction_sha256"]:
            raise ValueError("Saved fresh inference changed")
        raw = np.load(prediction_path)
        receipts = read_json(predictions_from/"predictions/model_run.json")["groups"]
        (destination/"predictions").mkdir(exist_ok=True)
    elif mode == "inference":
        raw, receipts = runner.run(sample, batches, destination/"predictions")
    else:
        raw = sample.saved_prediction.copy()
        receipts = [{"success": True, "fresh_inference": False, "source": "frozen legacy prediction"}]
        (destination/"predictions").mkdir(exist_ok=True)
    np.save(destination/"predictions/future_3d.npy", raw)
    max_difference = float(np.max(np.abs(raw.astype(float)-sample.saved_prediction.astype(float))))
    np.testing.assert_allclose(raw, sample.saved_prediction, atol=1e-7, rtol=0)
    write_json(destination/"prediction_parity.json", dict(success=True, mode=mode, atol_m=1e-7,
                                                         max_difference_m=max_difference,
                                                         fresh_inference=mode == "inference"))
    write_json(destination/"predictions/model_run.json", dict(mode=mode, success=True, groups=receipts,
                                                            config_sha256=sha(checkpoint/"config.yaml"),
                                                            predictions_from=str(predictions_from) if predictions_from else None))
    write_json(destination/"future_access_receipt.json", dict(prediction_saved_first=True,
               prediction_sha256=sha(destination/"predictions/future_3d.npy"), purpose="evaluation only"))
    evaluated = adapter.load(config, source_root, evaluation=True)
    for field in ("history_frames", "points_2d", "points_3d_history", "history_timestamps", "point_ids"):
        np.testing.assert_array_equal(getattr(evaluated, field), getattr(sample, field))
    if evaluated.action != sample.action:
        raise ValueError("Evaluation changed the action")
    sample = evaluated
    # Dataset-specific temporal/coordinate preparation lives in the adapters;
    # everything from here through metrics, media and artifacts is shared.
    xyz, uv = adapter.finish(sample, raw)
    np.savez_compressed(destination/"predictions/aligned.npz", **xyz, **{n+"_uv":v for n,v in uv.items()})
    metrics, errors = compute_metrics(sample, xyz, uv)
    write_json(destination/"metrics.json", metrics)
    np.savez_compressed(destination/"errors.npz", **errors)
    # Keep model-input files exactly as sealed before inference. Future camera
    # poses and references belong only to evaluation artifacts.
    reference_arrays = {name:value for name,value in sample.evaluation.items()
                        if name in ("xyz","uv","mask2","mask3") and value is not None}
    if sample.camera_poses is not None:
        reference_arrays["evaluation_camera_poses"] = sample.camera_poses
    evaluation_dir = destination/"evaluation"
    evaluation_dir.mkdir()
    np.savez_compressed(evaluation_dir/"reference.npz", **reference_arrays)
    source_hashes = {str(p.relative_to(source_root)):sha(p) for p in set(sample.source_files)}
    write_json(destination/"metadata.json", dict(sample.metadata, source_sha256=source_hashes,
               mode=mode, checkpoint="allenai/MolmoMotion-4B-H3-F30",
               checkpoint_revision="3f5e790a511ff2cdf21c8d2a14cb4d8409c94629",
               dtype="bfloat16", seed=0, future_used_for_model_input=False))
    if predictions_from is not None:
        metadata=read_json(destination/"metadata.json")
        metadata["prediction_origin"]="verified fresh inference"
        metadata["predictions_from"]=str(predictions_from)
        write_json(destination/"metadata.json",metadata)
    viz = destination/"visualizations"
    viz.mkdir()
    error_plots(sample, metrics, viz)
    trajectory_plots(sample, xyz, uv, viz)
    videos = make_visualizations(sample, xyz, uv, metrics, viz)
    compare_legacy(sample, source_root, viz)
    status = dict(status="COMPLETE", success=True, mode=mode, fresh_inference=mode == "inference",
                  runtime_seconds=time.monotonic()-start, videos=videos,
                  legacy_projection_scope_preserved=True,
                  metric_3d_status=metrics["metric_3d_status"])
    if any(sha(source_root/p) != digest for p,digest in observed_hashes.items()):
        raise ValueError("Observed source changed during the run")
    write_json(destination/"status.json", status)
    write_case_page(sample, metrics, destination)
    print(config["name"], "COMPLETE", f"{status['runtime_seconds']:.1f}s", flush=True)
    return dict(name=config["name"], **status)


def write_case_page(sample, metrics, destination):
    metadata = read_json(destination/"metadata.json")
    origin = metadata.get("prediction_origin", "new model inference" if metadata["mode"] == "inference" else "frozen legacy prediction")
    rows = []
    for name, data in metrics["methods"].items():
        values = []
        for key in ("2D_px", "3D_m"):
            m = data[key]
            values.extend([f"{m[field]:.5f}" if m is not None and m[field] is not None else "—" for field in ("ADE", "FDE")])
        rows.append("<tr><td>"+html.escape(name)+"</td>"+"".join(f"<td>{v}</td>" for v in values)+"</tr>")
    description = metrics["methods"][metrics["primary"]].get("trajectory_2D")
    criteria = ""
    if description:
        fields = [("Медианная ошибка направления, °", "direction_error_median_degrees"),
                  ("Амплитуда конечного смещения / reference", "endpoint_displacement_ratio_median"),
                  ("Длина пути / reference", "path_length_ratio_median"),
                  ("Средняя ошибка скорости, px/s", "speed_MAE_px_per_s")]
        items = ''.join(f'<li>{label}: {description[key]:.3f}</li>' if description[key] is not None else f'<li>{label}: недостаточно наблюдений</li>' for label,key in fields)
        criteria = "<h2>Характер движения</h2><ul>"+items+f"</ul><p>Для направления доступны {description['moving_direction_samples']} движущихся сегментов; полностью наблюдаемы {description['complete_point_paths']} траекторий. Скорость учитывает физические интервалы кадров. Если реальное смещение почти нулевое, отношение амплитуд чувствительно к малой ошибке reference и не означает большой физической разницы.</p>"
    else:
        criteria = "<h2>Характер движения</h2><p>Движение велосипеда сравнивается в общей 3D системе по форме пути, XYZ и ошибке во времени. Изображение меняется вместе с камерой, поэтому сравнение амплитуды 2D проекции не является метрикой модели.</p>"
    method_media = '<h2>Сопоставление методов</h2><video controls loop src="visualizations/methods.mp4"></video>' if (destination/"visualizations/methods.mp4").exists() else ""
    page = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(sample.metadata["title"])}</title>
<style>body{{font:16px/1.5 system-ui;max-width:1250px;margin:25px auto;padding:0 18px;background:#111923;color:#edf1f7}}a{{color:#80bcff}}img,video{{width:100%;margin:12px 0}}td,th{{padding:8px;border-bottom:1px solid #384453;text-align:left}}table{{border-collapse:collapse}}.note{{background:#263347;padding:16px;border-radius:8px}}</style>
<p><a href="../index.html">Все примеры</a> · <a href="metrics.json">Метрики JSON</a> · <a href="metadata.json">Протокол</a></p>
<h1>{html.escape(sample.metadata["title"])}</h1><p>{html.escape(sample.action)}</p>
<p>Источник прогноза: {html.escape(origin)}.</p>
<p class="note">{html.escape(sample.metadata["limitation"])}</p>
<p>Основной вариант: <b>{html.escape(metrics["primary"])}</b>. В видео показаны 8 точек; метрики используют все {len(sample.point_ids)}. Розовый — прогноз, зелёный — независимый reference, голубой — входные точки. Стрелки у края показывают уход за изображение.</p>
<h2>Вход</h2><img src="visualizations/inputs.png">
<h2>Прогноз и реальное продолжение</h2><video controls loop src="visualizations/comparison.mp4"></video><p><a href="visualizations/prediction.mp4">Прогноз на наблюдаемом фоне</a></p>
<p><a href="visualizations/legacy_vs_unified.png">Старая и единая отрисовка рядом</a></p>
{method_media}
<table><tr><th>Метод</th><th>ADE 2D, px</th><th>FDE 2D, px</th><th>ADE 3D_est, m</th><th>FDE 3D_est, m</th></tr>{''.join(rows)}</table>
<p>ADE — средняя ошибка по фиксированной маске точка–время. FDE — ошибка на фиксированном последнем шаге. 3D_est — оценённый reference, а не подтверждённое физическое измерение.</p>
<img src="visualizations/errors.png"><img src="visualizations/ade_fde.png"><img src="visualizations/metric_comparison.png">
{criteria}<h2>Форма и координаты движения</h2><img src="visualizations/trajectories_2d.png"><img src="visualizations/trajectories_3d.png"><img src="visualizations/xyz_over_time.png">
</html>'''
    (destination/"index.html").write_text(page, encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--mode", choices=("replay", "inference"), default="inference")
    parser.add_argument("--source-root", type=Path, default=ROOT/"data/legacy")
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"data/checkpoints/MolmoMotion-4B-H3-F30")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions-from", type=Path, help="Re-render a verified completed new inference without another model call")
    args = parser.parse_args()
    if args.predictions_from is not None and args.mode != "replay":
        parser.error("--predictions-from requires --mode replay")
    configs = [read_json(p if p.is_absolute() else ROOT/p) for p in args.config]
    output = args.output or ROOT/"outputs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("HF_HOME", str(ROOT.parent/".cache/huggingface"))
    os.environ.setdefault("TORCH_HOME", str(ROOT.parent/".cache/torch"))
    runner = ModelRunner(args.checkpoint.resolve()) if args.mode == "inference" else None
    results = []
    for config in configs:
        results.append(run_case(config, args.source_root.resolve(), output/config["name"],
                                args.checkpoint.resolve(), args.mode, runner,
                                args.predictions_from.resolve()/config["name"] if args.predictions_from else None))
        write_json(output/"summary.json", results)
        cards = ''.join(f'<li><a href="{html.escape(r["name"])}/index.html">{html.escape(r["name"])}</a> — {r["status"]}</li>' for r in results)
        origin = "verified fresh inference, repeated rendering" if args.predictions_from else args.mode
        (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>MolmoMotion final examples</title><style>body{font:18px/1.8 system-ui;max-width:1100px;margin:40px auto;padding:0 20px}</style><h1>MolmoMotion: финальные примеры</h1><p>Режим: '+origin+'</p><ul>'+cards+'</ul>', encoding="utf8")
    print("OUTPUT", output, flush=True)


if __name__ == "__main__":
    main()
