"""CPU-only audit of the existing DAVIS result. Writes only beside this script.

Run with the existing WSL Python environment. No model/weights are loaded.
The baseline manifest must have been captured before this audit.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / "runs/author_davis_bmx_trees_f30"
SAMPLE = ROOT / "examples/data/davis_bmx_trees"
SOURCE = ROOT / "data/pointmotionbench/davis"
UPSTREAM = "61f5b21b694ad8f854ec7ecd2400005acc73f685"
DATA_REV = "564ffa2e3cdb0ba443db8590b40e9691f487436c"
IDS = np.array([9, 12, 17, 18, 27, 29, 32, 34])


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(2**20), b""):
            result.update(chunk)
    return result.hexdigest()


def lf(data):
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {"count": int(values.size), "mean": float(values.mean()) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "rmse": float(np.sqrt(np.mean(values**2))) if values.size else None,
            "max": float(values.max()) if values.size else None}


def guard_originals():
    manifest = json.loads((OUT / "baseline_manifest.json").read_text())
    changed, eol_only = [], []
    for name, record in manifest["files"].items():
        path = ROOT / name
        if sha(path) == record["sha256"]:
            continue
        if "sha256_lf" in record and hashlib.sha256(lf(path.read_bytes())).hexdigest() == record["sha256_lf"]:
            eol_only.append(name)
        else:
            changed.append(name)
    if changed:
        raise RuntimeError(f"Original files changed: {changed}")
    return {"files_checked": len(manifest["files"]), "byte_differences_only_in_line_endings": eol_only,
            "binary_or_normalized_content_mismatches": []}


def audit_sources():
    remote = {"pointmotionbench_readme.md": "README.md",
              "reconstruct_davis_source.py": "davis/reconstruct_davis.py",
              "bmx-trees_filter_meta.npz": "davis/tracks/bmx-trees_filter_meta.npz"}
    records = {name: {"url": f"https://huggingface.co/datasets/allenai/PointMotionBench/resolve/{DATA_REV}/{path}",
                      "sha256": sha(OUT/name)} for name,path in remote.items()}
    tree = json.loads((OUT/"source_tree_davis.json").read_text())
    records["source_tree_davis.json"] = {
        "url": f"https://huggingface.co/api/datasets/allenai/PointMotionBench/tree/{DATA_REV}/davis?recursive=true&limit=1000",
        "sha256": sha(OUT/"source_tree_davis.json"), "entries": len(tree),
        "camera_files": [r["path"] for r in tree if any(s in r["path"] for s in ("camera","pose","intrinsic","depth"))]}
    for name in ("bmx-trees_2d.npz","bmx-trees_3d.npz","bmx-trees_filter_meta.npz"):
        entry = next(r for r in tree if r["path"] == f"davis/tracks/{name}")
        path = OUT/name if "filter_meta" in name else SOURCE/name
        assert sha(path) == entry["lfs"]["oid"]
        records.setdefault(name, {}).update(sha256=sha(path), matches_pinned_hub_lfs_sha256=True)
    code_paths = ["src/molmo_motion/processor.py","src/molmo_motion/modeling.py",
                  "src/molmo_motion/data/trajectory_3d_dataset.py","src/molmo_motion/models/molmo2/molmo2.py",
                  "src/molmo_motion/models/molmo2/molmo2_trajectory.py"]
    for path in code_paths:
        upstream = subprocess.check_output(["git","show",f"{UPSTREAM}:{path}"],cwd=ROOT)
        assert lf((ROOT/path).read_bytes()) == lf(upstream)
        records[path] = {"sha256":sha(ROOT/path),"equals_upstream_after_explicit_lf_normalization":True}
    write_json("sources.json", {"revision":DATA_REV,"upstream_revision":UPSTREAM,"files":records})


def audit_copies():
    old_hashes = json.loads((RUN / "correction_baseline_hashes.json").read_text(encoding="utf-8-sig"))
    result = {}
    for path in sorted((RUN / "inputs").iterdir()):
        current = path.read_bytes()
        example = (SAMPLE / path.name).read_bytes()
        official = subprocess.check_output(["git", "show", f"{UPSTREAM}:examples/data/davis_bmx_trees/{path.name}"], cwd=ROOT)
        head = subprocess.check_output(["git", "show", f"HEAD:runs/author_davis_bmx_trees_f30/inputs/{path.name}"], cwd=ROOT)
        record = {"current_sha256": sha(path), "sample_sha256": hashlib.sha256(example).hexdigest(),
                  "official_example_sha256": hashlib.sha256(official).hexdigest(),
                  "HEAD_blob_sha256": hashlib.sha256(head).hexdigest(),
                  "prior_correction_sha256": old_hashes[f"inputs/{path.name}"],
                  "sample_byte_equal": current == example,
                  "upstream_byte_equal": current == official,
                  "HEAD_byte_equal": current == head}
        if path.suffix in (".txt", ".json"):
            record.update(current_lf_sha256=hashlib.sha256(lf(current)).hexdigest(),
                          sample_lf_sha256=hashlib.sha256(lf(example)).hexdigest(),
                          official_lf_sha256=hashlib.sha256(lf(official)).hexdigest(),
                          HEAD_lf_sha256=hashlib.sha256(lf(head)).hexdigest(),
                          current_crlf=current.count(b"\r\n"),
                          current_bare_lf=current.count(b"\n")-current.count(b"\r\n"),
                          all_equal_after_explicit_lf_normalization=lf(current) == lf(example) == lf(official) == lf(head))
            assert record["all_equal_after_explicit_lf_normalization"]
        else:
            assert current == example == official == head
        result[path.name] = record
    write_json("input_copies_and_eol.json", result)


def load_data():
    with np.load(SOURCE / "bmx-trees_2d.npz", allow_pickle=True) as file:
        xy = file["tracks"].item()["bike_rider"]
        vis = file["visibility"].item()["bike_rider"]
        dim = file["dim"]
    with np.load(SOURCE / "bmx-trees_3d.npz", allow_pickle=True) as file:
        xyz = file["points_3d"].item()["bike_rider"]
    with np.load(RUN / "ground_truth.npz") as file:
        gt = {key: file[key] for key in file.files}
    with np.load(RUN / "prediction.npz") as file:
        pred = file["future_3d"]
    points = torch.load(RUN / "inputs/points_3d_history.pt", weights_only=True).numpy()
    query = torch.load(RUN / "inputs/points_2d_at_t0.pt", weights_only=True).numpy()
    k = torch.load(RUN / "inputs/intrinsics_K.pt", weights_only=True).numpy()
    assert xy.shape == (80, 86, 2) and xyz.shape == (86, 80, 3)
    assert dim.tolist() == [480, 854]
    # Discover (ID, frame) from exact values, independently of meta.json.
    matches = []
    for pi in range(8):
        matches_3d = [[int(j), int(t)] for j in range(86) for t in range(78)
                      if np.array_equal(xyz[j, t:t+3], points[:, pi])]
        matches_2d = np.argwhere(np.all(xy == query[pi], axis=-1)).tolist()
        assert matches_3d == [[int(IDS[pi]), 0]]
        assert matches_2d == [[2, int(IDS[pi])]]
        matches.append({"wire_id": pi + 1, "original_point_id": int(IDS[pi]),
                        "exact_3d_matches_ID_start": matches_3d,
                        "exact_2d_matches_frame_ID": matches_2d})
    checks = {
        "history_3d": np.array_equal(gt["history_3d"], points),
        "history_2d": np.array_equal(gt["history_2d"], xy[:3, IDS]),
        "future_3d": np.array_equal(gt["future_3d"], xyz[IDS, 3:33], equal_nan=True),
        "future_2d": np.array_equal(gt["future_2d"], xy[3:33, IDS].transpose(1, 0, 2)),
        "mask": np.array_equal(gt["valid"], vis[3:33, IDS].T & np.isfinite(xyz[IDS, 3:33]).all(-1)),
        "source_2d_visibility_alone_equals_mask": np.array_equal(gt["valid"], vis[3:33, IDS].T),
        "future_frame_indices": np.array_equal(gt["future_frame_indices"], np.arange(3, 33)),
        "future_times": np.array_equal(gt["future_seconds_from_t0"], np.arange(1, 31) / 24),
        "K": np.array_equal(k, gt["intrinsics_K"]),
    }
    assert all(checks.values())
    inactive_error = np.linalg.norm(xy[IDS,2] - xy[2,IDS],axis=-1)
    write_json("inactive_2d_loader_diagnostic.json", {
        "actual_upstream_indexing": "tracks[chosen_indices, t] on a (T,N,2) array",
        "correct_source_indexing": "tracks[t, chosen_indices]",
        "per_point_pixel_difference": inactive_error.tolist(),
        "mean_pixel_difference": float(inactive_error.mean()), "affects_saved_checkpoint": False})
    write_json("point_mapping.json", {"source_2d_axes": ["time", "point", "xy_pixels"],
        "source_3d_axes": ["point", "time", "XYZ_meters"], "source_shapes": {"2d": list(xy.shape), "3d": list(xyz.shape)},
        "image_height_width": dim.tolist(), "K_saved": k.tolist(), "matches": matches, "reference_checks": checks})
    return xy, xyz, gt, pred, points, query, k


def audit_frames(xy):
    font = ImageFont.truetype("DejaVuSans.ttf", 21)
    sheet = Image.new("RGB", (854 * 2, 540 * 3), "#111111")
    rows = []
    with zipfile.ZipFile(SOURCE / "DAVIS-2017-trainval-480p.zip") as archive:
        members = sorted(n for n in archive.namelist()
                         if "/JPEGImages/480p/bmx-trees/" in n and n.endswith(".jpg"))
        assert len(members) == 80
        # Claimed frames and neighboring candidates. No resizing or cropping.
        candidates = {t: np.asarray(Image.open(io.BytesIO(archive.read(members[t]))).convert("RGB"))
                      for t in range(7)}
        comparisons = []
        for h, offset in enumerate((-2, -1, 0)):
            input_file = RUN / f"inputs/frame_t{offset:+d}.jpg"
            image = Image.open(input_file).convert("RGB")
            array = np.asarray(image)
            errors = []
            for t, original in candidates.items():
                assert array.shape == original.shape == (480, 854, 3)
                delta = array.astype(float) - original
                mae = float(np.abs(delta).mean())
                mse = float(np.mean(delta**2))
                errors.append((mae, t, 10 * np.log10(255**2 / mse)))
                comparisons.append({"history_position": h, "source_candidate": t, "MAE_0_255": mae,
                                    "PSNR_dB": float(errors[-1][2])})
            errors.sort()
            best, source_index, psnr = errors[0]
            assert source_index == h
            original_bytes = archive.read(members[source_index])
            assert original_bytes == (RUN / f"frames/{source_index:05d}.jpg").read_bytes()
            rows.append({"history_position_0based": h, "history_position_1based": h+1,
                         "input_file": str(input_file.relative_to(ROOT)), "source_zip_member": members[source_index],
                         "source_frame_0based": source_index, "time_in_24fps_reconstruction_s": source_index/24,
                         "time_relative_t0_s": (source_index-2)/24, "model_nominal_timestamp": float(h),
                         "annotation_time_index": source_index, "width": 854, "height": 480,
                         "MAE_0_255": best, "PSNR_dB": float(psnr),
                         "runner_up_frame": errors[1][1], "runner_up_MAE": errors[1][0],
                         "source_vs_input_byte_equal": input_file.read_bytes() == original_bytes,
                         "source_vs_saved_frame_byte_equal": True})
            for col, rgb in enumerate((candidates[source_index], array)):
                tile = Image.fromarray(rgb.copy())
                draw = ImageDraw.Draw(tile)
                for pi, point_id in enumerate(IDS):
                    x, y = xy[source_index, point_id]
                    draw.ellipse((x-3, y-3, x+3, y+3), fill="#00ff88", outline="black")
                sheet.paste(tile, (854*col, 540*h+60))
                d = ImageDraw.Draw(sheet)
                label = f"DAVIS archive frame {source_index:05d}" if col == 0 else f"Saved model input, history slot {h}"
                d.text((854*col+12, 540*h+8), label + (" = t0" if h == 2 else ""), font=font, fill="white")
                d.text((854*col+12, 540*h+32), f"854 x 480; MAE {best:.3f}; PSNR {psnr:.2f} dB", font=font, fill="white")
    sheet.save(OUT / "frame_correspondence.png")
    write_json("frame_correspondence.json", {"rows": rows, "neighbor_candidates": comparisons,
        "time_basis": "24 fps is the released reconstruction convention; JPEGs have no measured camera clock."})
    with (OUT / "frame_correspondence.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def audit_processor(points, query):
    from molmo_motion import MolmoMotionProcessor
    from molmo_motion.models.molmo2.molmo2_trajectory import Molmo2TrajectoryConfig
    ckpt = ROOT / "data/checkpoints/MolmoMotion-4B-H3-F30"
    cfg = Molmo2TrajectoryConfig.load(str(ckpt / "config.yaml"), key="model", validate_paths=False)
    proc = MolmoMotionProcessor.from_pretrained(str(ckpt))
    saved = torch.load(RUN / "processor_inputs.pt", map_location="cpu", weights_only=False)
    fresh = proc(history_frames=[Image.open(RUN / f"inputs/frame_t{i:+d}.jpg").convert("RGB") for i in (-2,-1,0)],
                 points_2d_at_t0=torch.from_numpy(query), points_3d_history=torch.from_numpy(points),
                 action=(RUN / "inputs/caption.txt").read_text().strip(), future_horizon=30)
    equality = {key: bool(same(value, fresh[key])) for key, value in saved.items()}
    assert saved.keys() == fresh.keys() and all(equality.values()), equality
    decoded = proc.tokenizer.decode(saved["input_ids"][0].tolist(), truncate_at_eos=False)
    (OUT / "saved_prompt_decoded.txt").write_text(decoded)
    history_block = re.search(r'<tracks coords="([^"]+)">3d object history</tracks>', decoded)
    assert history_block is not None
    qhist = parse_tracks(history_block.group(1), [0,1,2])
    anchor = saved["anchor_3d"].numpy().reshape(3)
    expected = points.transpose(1,0,2) - anchor
    quantization_error = np.abs(qhist - expected)
    assert quantization_error.max() <= 0.0005001
    assert np.array_equal(anchor, points[-1,0])
    assert saved["future_horizon"] == 30 and saved["history_size"] == 3
    assert '<points' not in decoded and '<|point_feat|>' not in decoded
    # Invert only the patch reshape; these are the ACTUAL saved visual tensors.
    patches = saved["images"][0].numpy()
    rgb = patches.reshape(3,27,27,14,14,3).transpose(0,1,3,2,4,5).reshape(3,378,378,3)
    assert rgb.dtype == np.uint8
    sheet = Image.new("RGB", (378*3, 412), "#111111")
    for h in range(3):
        sheet.paste(Image.fromarray(rgb[h]), (378*h,34))
        ImageDraw.Draw(sheet).text((378*h+10,8),f"Saved processor pixels: history {h}",fill="white")
    sheet.save(OUT / "saved_processor_images.png")
    result = {"loaded_config_class": type(cfg).__name__, "model_name": cfg._model_name,
        "use_2d_point_features": bool(getattr(cfg,"use_2d_point_features",False)),
        "all_saved_fields_exactly_reproduced": equality,
        "metadata_type": type(saved["metadata"]).__name__,
        "metadata_2d_units": "original image pixels; inactive in this base Molmo2 checkpoint",
        "point_feature_tokens": 0, "input_token_count": int(saved["input_ids"].numel()),
        "images_shape": list(saved["images"].shape), "images_dtype": str(saved["images"].dtype),
        "K_passed_to_processor": False, "c2w_at_t0_passed": False,
        "anchor_m": anchor.tolist(), "history_quantization_max_abs_m": float(quantization_error.max()),
        "source_image_size": [854,480], "processor_image_size": [378,378],
        "image_transform": "full-frame anisotropic bilinear resize, antialias=False; no crop/pad/rotation",
        "synthetic_video_timestamps": [0.,1.,2.], "future_wire_timestamps": list(range(3,33)),
        "physical_fps_not_encoded_in_prompt": True}
    write_json("processor_audit.json", result)
    return anchor


def parse_tracks(coords, timestamps):
    frames = coords.split(";")
    assert len(frames) == len(timestamps)
    array = np.empty((8, len(frames), 3), np.float32)
    for ti, frame in enumerate(frames):
        parts = frame.split()
        assert float(parts[0]) == timestamps[ti] and len(parts) == 33
        seen = set()
        for off in range(1,33,4):
            assert all(re.fullmatch(r"[+-]?\d+", item) for item in parts[off:off+4])
            pi, x, y, z = map(int, parts[off:off+4])
            assert 1 <= pi <= 8 and pi not in seen
            seen.add(pi)
            array[pi-1,ti] = (x/1000,y/1000,z/1000)
        assert seen == set(range(1,9))
    assert np.isfinite(array).all()
    return array


def transform(points, rotation, translation):
    return np.asarray(points, dtype=np.float64) @ rotation.T + np.asarray(translation).reshape(3)


def project(points, k, rotation=None, translation=None):
    camera = np.asarray(points, dtype=np.float64)
    if rotation is not None:
        camera = transform(camera, rotation, translation)
    finite = np.isfinite(camera).all(-1)
    good = finite & (camera[...,2] > 0)
    pixels = np.full((*camera.shape[:-1],2), np.nan)
    homogeneous = camera[good] @ np.asarray(k,dtype=np.float64).T
    pixels[good] = homogeneous[:,:2] / homogeneous[:,2:3]
    return pixels, good, camera


def geometry_record(points, xy, k, rotation=None, translation=None):
    pixels, good, camera = project(points,k,rotation,translation)
    mask = good & np.isfinite(xy).all(-1)
    errors = np.linalg.norm(pixels-xy,axis=-1)
    return {"statistics_px": summary(errors[mask]),
        "per_point_error_px": [float(v) if np.isfinite(v) else None for v in errors],
        "nonfinite_3d_count": int((~np.isfinite(camera).all(-1)).sum()),
        "nonpositive_depth_count": int((np.isfinite(camera).all(-1) & (camera[:,2]<=0)).sum()),
        "nonfinite_2d_count": int((~np.isfinite(xy).all(-1)).sum()),
        "evaluated_mask": mask.tolist(), "projected_pixels": pixels.tolist(),
        "K": np.asarray(k).tolist(), "rotation_w2c": None if rotation is None else rotation.tolist(),
        "translation_w2c_m": None if translation is None else np.asarray(translation).reshape(3).tolist()}


def solve_pnp(points, xy, k):
    points=np.ascontiguousarray(points,dtype=np.float64)
    xy=np.ascontiguousarray(xy,dtype=np.float64)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(points,xy,k.astype(np.float64),None,flags=cv2.SOLVEPNP_SQPNP)
    candidates=[]
    for rv,tv in zip(rvecs,tvecs):
        # Deterministic local refinement, fitted only to the supplied historical points.
        ok, rv, tv = cv2.solvePnP(points,xy,k.astype(np.float64),None,rv,tv,True,flags=cv2.SOLVEPNP_ITERATIVE)
        rotation=cv2.Rodrigues(rv)[0]
        pixels, good, _=project(points,k,rotation,tv)
        if ok and good.all():
            candidates.append((float(np.mean(np.linalg.norm(pixels-xy,axis=-1))),rotation,tv.reshape(3)))
    if not candidates:
        raise RuntimeError("PnP has no positive-depth solution")
    candidates.sort(key=lambda item:item[0])
    error,rotation,translation=candidates[0]
    return rotation,translation,{"sqpnp_solutions":int(n),"positive_depth_candidates":len(candidates),
                                 "fit_candidate_mean_errors_px":[item[0] for item in candidates]}


def geometry_self_test():
    # Float64, well-conditioned synthetic scene: round-trip/projection tolerance 1e-10.
    k=np.array([[800.,0,320],[0,810,240],[0,0,1.]])
    points=np.array([[-1.,-.2,4],[.2,.4,5],[.8,-.5,6],[-.3,.7,7]])
    rotation=cv2.Rodrigues(np.array([.1,-.2,.04]))[0]
    translation=np.array([.2,.1,.4])
    camera=transform(points,rotation,translation)
    reverse=transform(camera,rotation.T,-rotation.T@translation)
    assert np.allclose(points,reverse,atol=1e-10,rtol=0)
    assert np.allclose(rotation.T@rotation,np.eye(3),atol=1e-10,rtol=0)
    assert abs(np.linalg.det(rotation)-1)<1e-10
    pixels,mask,_=project(points,k,rotation,translation)
    reference,_=cv2.projectPoints(points,cv2.Rodrigues(rotation)[0],translation,k,None)
    assert np.max(np.abs(pixels-reference[:,0]))<1e-10 and mask.all()
    # Scaling BOTH the scene and translation preserves pixels, not metric distances.
    scaled,_,_=project(points*1000,k,rotation,translation*1000)
    assert np.max(np.abs(scaled-pixels))<1e-10
    invalid=np.array([[np.nan,0,1],[0,np.inf,1],[0,0,0],[0,0,-1],[0,0,1.]])
    pix,valid,_=project(invalid,k)
    assert valid.tolist()==[False,False,False,False,True]
    assert np.isnan(pix[:4]).all()
    return {"round_trip_max_abs_m":float(np.max(np.abs(reverse-points))),
            "synthetic_projection_max_abs_px":float(np.max(np.abs(pixels-reference[:,0]))),
            "synthetic_numeric_tolerance":1e-10,"invalid_depth_and_nonfinite_test":"PASS"}


def audit_geometry(points, history_xy, k):
    result={"tests":geometry_self_test(),"projection_units":"pixels at original 854x480; XYZ meters",
        "direct_projection_assumption":"Treat stored XYZ as current-camera coordinates. This is a hypothesis, not an available identity camera pose.",
        "PnP_assumptions":"Known saved K, zero distortion, only 8 historical correspondences in each respective frame; estimated poses, not author extrinsics.",
        "noise_protocol":"20 trials/frame; fixed RNG seed 20260930, Gaussian 0.5px std on fitting observations; sensitivity study, not a calibrated confidence interval.",
        "frames":[]}
    poses=[]
    for t in range(3):
        p=points[t].astype(np.float64); xy=history_xy[t].astype(np.float64)
        direct=geometry_record(p,xy,k)
        singular=np.linalg.svd(p-p.mean(0),compute_uv=False)
        r,tr,candidates=solve_pnp(p,xy,k)
        poses.append((r,tr))
        assert np.linalg.norm(r.T@r-np.eye(3))<1e-10 and abs(np.linalg.det(r)-1)<1e-10
        pnp=geometry_record(p,xy,k,r,tr)
        center=-r.T@tr
        loo=[]
        for hold in range(8):
            fitted=np.array([i for i in range(8) if i != hold])
            rr,tt,_=solve_pnp(p[fitted],xy[fitted],k)
            uv,good,_=project(p[hold:hold+1],k,rr,tt)
            loo.append({"held_out_original_ID":int(IDS[hold]),"positive_depth":bool(good[0]),
                        "held_out_error_px":float(np.linalg.norm(uv[0]-xy[hold])),
                        "camera_center_difference_m":float(np.linalg.norm(-rr.T@tt-center))})
        rng=np.random.default_rng(20260930)
        trials=[]
        for _ in range(20):
            rr,tt,_=solve_pnp(p,xy+rng.normal(0,.5,xy.shape),k)
            angle=np.degrees(np.arccos(np.clip((np.trace(rr@r.T)-1)/2,-1,1)))
            trials.append({"rotation_difference_deg":float(angle),
                           "camera_center_difference_m":float(np.linalg.norm(-rr.T@tt-center))})
        result["frames"].append({"absolute_frame":t,"point_IDs":IDS.tolist(),"direct":direct,"pnp_fitted":pnp,
            "center_m":center.tolist(),"rotation_vector_deg":np.degrees(cv2.Rodrigues(r)[0].reshape(3)).tolist(),
            "geometry_singular_values_m":singular.tolist(),"smallest_to_largest_singular_ratio":float(singular[-1]/singular[0]),
            "solver_candidates":candidates,"leave_one_out":loo,
            "leave_one_out_error_px":summary([v["held_out_error_px"] for v in loo]),
            "noise_trials":trials,"noise_rotation_deg":summary([v["rotation_difference_deg"] for v in trials]),
            "noise_center_m":summary([v["camera_center_difference_m"] for v in trials])})
    write_json("geometry.json",result)
    # Zoom at the actual points, with original pixel coordinates on the axes.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for t in range(3):
        rgb=np.asarray(Image.open(RUN/f"inputs/frame_t{t-2:+d}.jpg").convert("RGB"))
        xy=history_xy[t]
        for row,name in enumerate(("direct","pnp_fitted")):
            ax=axes[row,t]; ax.imshow(rgb)
            record=result["frames"][t][name]
            uv=np.asarray(record["projected_pixels"])
            ax.scatter(xy[:,0],xy[:,1],c="#00e699",s=30,marker="o",label="2D annotation")
            ax.scatter(uv[:,0],uv[:,1],c="#ff3355",s=40,marker="x",label="3D projection")
            for pi,pid in enumerate(IDS):
                ax.plot([xy[pi,0],uv[pi,0]],[xy[pi,1],uv[pi,1]],color="#ffcc00",lw=1)
                label_positions = [(370,252),(435,143),(540,210),(370,214),
                                   (540,275),(370,177),(540,235),(430,289)]
                ax.annotate(str(pid),xy[pi],xytext=label_positions[pi],textcoords="data",fontsize=10,
                            color="white",bbox=dict(facecolor="black",alpha=.75,pad=2),
                            arrowprops=dict(arrowstyle="-",color="white",alpha=.7,linestyle=":",lw=.8))
            ax.set(xlim=(360,555),ylim=(300,130),xlabel="x (original pixels)",ylabel="y (original pixels)")
            prefix="Direct K projection (unverified frame)" if row==0 else "Estimated PnP pose (fit on these points)"
            ax.set_title(f"Frame {t}{' = t0' if t==2 else ''}: {prefix}\nmean {record['statistics_px']['mean']:.4f} px",fontsize=10)
            if t==0: ax.legend(fontsize=8,loc="lower left")
    fig.tight_layout();fig.savefig(OUT/"projection_diagnostics.png",dpi=150);plt.close(fig)
    return poses,result


def audit_smoothing(points, xyz, history_xy, k):
    with np.load(OUT/"bmx-trees_filter_meta.npz") as file:
        meta={key:file[key] for key in file.files}
    matches=[]; prior=[]
    for pid in IDS:
        found=[i for i in range(len(meta["P_smoothed"]))
               if np.array_equal(meta["P_smoothed"][i],xyz[pid],equal_nan=True)]
        assert len(found)==1
        matches.append({"published_point_id":int(pid),"filter_metadata_row":found[0],
                        "kept":bool(meta["keep_mask"][found[0]])})
        prior.append(meta["P_original"][found[0],:3])
    prior=np.stack(prior,axis=1)
    records=[]
    for t in range(3):
        records.append({"frame":t,"original_to_smoothed_displacement_m":summary(np.linalg.norm(prior[t]-points[t],axis=-1)),
                        "unsmoothed_direct_projection":geometry_record(prior[t],history_xy[t],k),
                        "unsmoothed_vs_rounded_2d_annotation":geometry_record(prior[t],np.rint(history_xy[t]),k)})
    result={"published_tracks_exactly_match_filter_P_smoothed":True,"row_matches":matches,"historical_effect":records,
            "metadata_has_camera_poses":False,"metadata_has_intrinsics":False,
            "exact_reconstruction_command_and_causality_log":"not provided; future-frame use in this exact run not established"}
    write_json("reconstruction_provenance.json",result)
    return result


def compute_metrics(pred,gt,valid):
    if pred.shape != gt.shape or not np.isfinite(pred).all():
        raise ValueError("Incomplete/nonfinite prediction")
    if not np.isfinite(gt[valid]).all():
        raise ValueError("Nonfinite valid reference")
    error=np.linalg.norm(pred.astype(np.float64)-gt.astype(np.float64),axis=-1)
    return {"ADE_m":float(error[valid].mean()) if valid.any() else None,
            "FDE_m":float(error[valid[:,-1],-1].mean()) if valid[:,-1].any() else None,
            "error_by_horizon_m":[float(error[valid[:,i],i].mean()) if valid[:,i].any() else None for i in range(30)],
            "valid_points_by_horizon":valid.sum(axis=0).tolist()},error


def audit_metrics(pred,gt,points,anchor,poses):
    raw=(RUN/"model_output_raw.txt").read_text()
    match=re.fullmatch(r'<tracks coords="([^"]+)">[^<]*</tracks>\s*',raw)
    assert match is not None
    reconstructed=parse_tracks(match.group(1),list(range(3,33)))+anchor
    assert np.array_equal(reconstructed,pred)
    valid=gt["valid"]
    assert valid.sum()==215 and valid[:,-1].sum()==8 and not valid[:,14].any()
    predictions={"MolmoMotion":pred,"Static":np.broadcast_to(points[-1,:,None,:],pred.shape).copy(),
                 "Constant velocity":points[-1,:,None,:]+np.arange(1,31,dtype=np.float32)[None,:,None]*(points[-1]-points[-2])[:,None,:]}
    old=json.loads((RUN/"metrics.json").read_text())
    rows={}; arrays={}; invariance={}
    r,t=poses[2] # Historical-only diagnostic change of common frame, not a new inference.
    for name,p in predictions.items():
        row,error=compute_metrics(p,gt["future_3d"],valid)
        for field in ("ADE_m","FDE_m"):
            assert abs(row[field]-old[name][field])<=1e-7
        assert row["error_by_horizon_m"]==old[name]["error_by_horizon_m"]
        transformed=np.linalg.norm(transform(p,r,t)-transform(gt["future_3d"],r,t),axis=-1)
        invariance[name]=float(np.max(np.abs(transformed[valid]-error[valid])))
        assert invariance[name]<1e-12
        rows[name]=row;arrays[name.replace(' ','_')]=error
    np.savez_compressed(OUT/"metric_error_arrays.npz",valid=valid,**arrays)
    result={"raw_timestamps":list(range(3,33)),"wire_ids_per_timestamp":list(range(1,9)),"finite_positions":240,
            "raw_vs_prediction_max_abs_m":0.,"valid_pairs":int(valid.sum()),"fixed_final_frame":32,
            "FDE_valid_points":8,"frame17_valid_points":0,"horizon_at_reconstruction_24fps_s":1.25,
            "methods":rows,"common_rigid_transform_max_error_difference_m":invariance,
            "metric_match_tolerance_m":1e-7}
    write_json("metrics_recomputed.json",result)
    return result


def main():
    preservation=guard_originals();print(f"Original manifest: {preservation}",flush=True)
    audit_sources()
    audit_copies()
    xy,xyz,gt,pred,points,query,k=load_data()
    frame_rows=audit_frames(xy);print("Image and point mapping: PASS",flush=True)
    anchor=audit_processor(points,query);print("Saved processor tensors reproduced exactly",flush=True)
    poses,geometry=audit_geometry(points,gt["history_2d"],k)
    smoothing=audit_smoothing(points,xyz,gt["history_2d"],k)
    metric=audit_metrics(pred,gt,points,anchor,poses)
    preservation=guard_originals()
    write_json("validation.json",{"original_files_checked":preservation["files_checked"],
        "original_byte_differences_only_in_line_endings":preservation["byte_differences_only_in_line_endings"],
        "original_binary_or_normalized_content_mismatches":[],
        "processor_reproduced":True,"model_loaded_or_run":False,"geometry_numeric_tests":"PASS",
        "source_revision":DATA_REV,"upstream_code_revision":UPSTREAM,
        "audit_inputs_revision":json.loads((OUT/"baseline_manifest.json").read_text())["git_head"],
        "python_version":os.sys.version,"numpy_version":np.__version__,"opencv_version":cv2.__version__,
        "torch_version":torch.__version__})
    print(json.dumps({"direct_mean_px":[f["direct"]["statistics_px"]["mean"] for f in geometry["frames"]],
        "pnp_fit_mean_px":[f["pnp_fitted"]["statistics_px"]["mean"] for f in geometry["frames"]],
        "pnp_LOO_mean_px":[f["leave_one_out_error_px"]["mean"] for f in geometry["frames"]],
        "metrics":{k:{f:v[f] for f in ("ADE_m","FDE_m")} for k,v in metric["methods"].items()}},indent=2),flush=True)


if __name__ == "__main__":
    main()
