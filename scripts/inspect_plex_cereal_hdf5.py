"""Inspect the small, selectively fetched official PLEX PickPlaceCereal HDF5."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "plex" / "PickPlaceCereal_demo_act_norm.hdf5"
OUT = ROOT / "runs" / "plex_dobbe_preflight" / "plex_cereal_hdf5_audit.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    demos = []
    with h5py.File(SOURCE, "r") as file:
        data = file["data"]
        env_args = json.loads(data.attrs["env_args"])
        for key in sorted(data, key=lambda x: int(x.split("_")[1])):
            episode = data[key]
            states = episode["states"]
            xml = episode.attrs["model_file"]
            if isinstance(xml, bytes):
                xml = xml.decode()
            scene = ET.fromstring(xml)
            camera = scene.find('.//camera[@name="agentview"]')
            cereal = scene.find('.//body[@name="Cereal_main"]')
            assets = sorted({node.get("file") for node in scene.findall('.//*[@file]')
                             if node.get("file")})
            times = states[:, 0]
            demos.append({"demo": key, "samples": len(states),
                          "states_shape": list(states.shape),
                          "actions_shape": list(episode["actions"].shape),
                          "step_seconds_min_max": np.diff(times).min(initial=np.nan).item()
                          if len(times) < 2 else [float(np.min(np.diff(times))),
                                                    float(np.max(np.diff(times)))],
                          "model_file_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                          "agentview": camera.attrib if camera is not None else None,
                          "cereal_body": cereal.attrib if cereal is not None else None,
                          "cereal_free_joint": cereal.find("joint").attrib
                          if cereal is not None and cereal.find("joint") is not None else None,
                          "asset_count": len(assets),
                          "absolute_asset_count": sum(Path(a).is_absolute() for a in assets),
                          "example_assets": assets[:8]})
        output = {"source": str(SOURCE),
                  "repository_version": data.attrs.get("repository_version"),
                  "collection_date": data.attrs.get("date"),
                  "env_args": env_args,
                  "demo_count": len(demos),
                  "model_file_hash_count": len({d["model_file_sha256"] for d in demos}),
                  "demos": demos}
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"repository_version": output["repository_version"],
                      "env_args": output["env_args"],
                      "demo_count": len(demos),
                      "model_file_hash_count": output["model_file_hash_count"],
                      "demo_1": demos[0],
                      "samples_min_max": [min(d["samples"] for d in demos),
                                          max(d["samples"] for d in demos)]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
