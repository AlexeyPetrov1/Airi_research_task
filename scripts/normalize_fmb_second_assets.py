"""Describe official FMB calibration and CAD evidence with explicit provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/fmb_second_scene"
CAMERAS = ("side_1", "side_2", "wrist_1", "wrist_2")
SERIALS = {"side_1": "128422270679", "side_2": "127122270146",
           "wrist_1": "127122270350", "wrist_2": "128422271851"}
PROJECT = "https://functional-manipulation-benchmark.github.io/"
CODE = "https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra/"


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    calibration = {
        "status": "UNRESOLVED_ACTIVE_256_INTRINSICS_AND_DEPTH_SCALE",
        "source_scope": "Official FMB setup assets; the NPY does not carry camera serials or per-run calibration records",
        "camera_model": {"value": "Intel RealSense D405", "source": PROJECT + "files/index.html#camera-mounts",
                         "per_run_device_confirmation": "UNKNOWN"},
        "camera_capture": {
            "color_format": "BGR8", "depth_format": "Z16", "configured_resolution_wh": [640, 480],
            "configured_fps": 15, "alignment_requested": "depth to color before capture",
            "source": CODE + "camera/rs_capture.py",
        },
        "published_arrays": {"resolution_wh": [256, 256], "rgb_color_order": "BGR",
                             "depth_dtype": "uint16", "source": PROJECT + "dataset/index.html"},
        "processing": {
            "rgb_resize_to_256": {"value": "cv2.resize(rgb, (256, 256))", "source": CODE + "envs/franka_fmb_env.py#L225-L232"},
            "depth_resize_crop_rectification_after_capture": {"status": "UNKNOWN", "reason": "Published 256x256 depth differs from the 640x480 capture code; no matching transformation is present in published _get_im()."},
            "exact_rgb_depth_correspondence_256": "UNRESOLVED",
        },
        "depth_scale_m_per_sample": {"status": "UNRESOLVED", "value": None,
                                     "reason": "Raw Z16 values and a camera model do not establish a per-run scale; no independently confirmed scale was found for this trajectory"},
        "rgb_depth_extrinsics": {"status": "UNRESOLVED_PUBLISHED_256", "capture_alignment": "depth_to_color_configured",
                                 "source": CODE + "camera/rs_capture.py"},
        "side_1_to_side_2_extrinsics": {"status": "UNKNOWN", "value": None},
        "camera_to_robot_base_extrinsics": {"status": "UNKNOWN", "value": None},
        "wrist_camera_extrinsics": {"status": "UNKNOWN", "value": None},
        "distortion_coefficients_for_published_RGB": {"status": "UNKNOWN", "value": None},
        "cameras": {},
    }
    for camera in CAMERAS:
        path = DATA / "calibration/raw" / camera
        raw = json.loads(path.read_text(encoding="utf-8"))
        profiles = {}
        for prefix in sorted({".".join(k.split(".")[:2]) for k in raw if k.startswith("rectified.")},
                             key=lambda item: int(item.split(".")[1])):
            vals = {key[len(prefix)+1:]: float(value) for key, value in raw.items() if key.startswith(prefix + ".")}
            if vals.get("width", 0) > 0 and vals.get("height", 0) > 0:
                profiles[prefix] = {"resolution_wh": [int(vals["width"]), int(vals["height"])],
                                    "K": [[vals["fx"], 0, vals["ppx"]], [0, vals["fy"], vals["ppy"]], [0, 0, 1]],
                                    "source": PROJECT + "static/files/" + camera + "#" + prefix}
        calibration["cameras"][camera] = {
            "raw_file": str(path.relative_to(ROOT)), "raw_file_sha256": sha256(path),
            "capture_serial_in_official_code": {"value": SERIALS[camera],
                                                "source": CODE + "envs/franka_fmb_env.py#L113-L116",
                                                "confirmed_for_selected_run": False},
            "published_rectified_profiles": profiles,
            "profile_640x480": "rectified.2" if "rectified.2" in profiles else None,
            "active_rgb_K_256": {"status": "UNRESOLVED", "value": None},
            "active_depth_K_256": {"status": "UNRESOLVED", "value": None},
            "raw_vendor_parameters": {key: {"value_as_published": value,
                                            "source": PROJECT + "static/files/" + camera + "#" + key}
                                      for key, value in raw.items() if not key.startswith("rectified.")},
        }
    save(DATA / "calibration/calibration.json", calibration)

    info = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))["object_info"]
    descriptor = {
        "object_info_exact": info,
        "decoded": {"shape": "rectangle", "size": "large", "length": "long", "color": "yellow",
                    "initial_angle": "vertical", "distractor": False},
        "decode_sources": [PROJECT + "dataset/index.html", PROJECT + "files/index.html",
                           "https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/fmb_dataset_builder/fmb_single_object_dataset/fmb_single_object_dataset_dataset_builder.py"],
        "peg_id": 1,
        "official_cad": {"path": "data/fmb_second_scene/object/peg_official.step",
                         "url": PROJECT + "static/files/peg.step", "sha256": sha256(DATA / "object/peg_official.step"),
                         "contains": "54 peg solids: 9 shapes x 3 cross-section sizes x 2 lengths",
                         "exact_solid_for_this_run": "UNRESOLVED; STEP bodies have no per-body shape/size IDs"},
        "reference_sheet": {"path": "data/fmb_second_scene/object/shape_color_reference.pdf",
                            "url": PROJECT + "static/doc/FMB%20Shape%20and%20Color%20Number%20Reference%20Sheet%20-%20Google%20Docs.pdf"},
        "physical_dimensions": {"status": "UNRESOLVED_EXACT_BODY_MAPPING", "value": None,
                                "note": "Official STEP includes the model family; dimensions of an individual body must not be assigned to this run without a proven body-ID map"},
    }
    save(DATA / "object/object_metadata.json", descriptor)
    print("NORMALIZED", len(calibration["cameras"]), "cameras")


if __name__ == "__main__":
    main()
