"""History-only candidate checks for the selected WorldTrack episode."""

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


root = Path(__file__).parent
key = "adt_mini/Apartment_release_clean_seq131_0_clip02_obj1_t108-149"
entry = json.loads((root / "source/worldtrack_index_map.json").read_text())[key]
data = np.load(root / "source/Apartment_release_clean_seq131_0.npz", allow_pickle=True)
ids = np.array(entry["point_indices"], int)
objects = np.array(entry["object_ids"], int)
clip_ids = ids[objects == entry["clip_objects"][0]]
xyz = data["tracks_XYZ"]
vis = data["visibility"]
intr = data["fx_fy_cx_cy"]


def project(p):
    return np.stack((intr[0] * p[..., 0] / p[..., 2] + intr[2], intr[1] * p[..., 1] / p[..., 2] + intr[3]), -1)


print("all clip points", len(ids), "object1", len(clip_ids))
print("intr", intr)
for t in (108, 109, 110, 111, 140):
    im = Image.open(io.BytesIO(bytes(data["images_jpeg_bytes"][t]))).convert("RGB")
    print("frame", t, "image", im.size, "visible object1", int(vis[t, clip_ids].sum()))
print("history valid ids", clip_ids[vis[108:111, clip_ids].all(axis=0)].tolist())
for t in range(110, 120):
    print("candidate_t0", t, "three_history_visible", int(vis[t - 2:t + 1, clip_ids].all(axis=0).sum()))
valid = clip_ids[vis[108:111, clip_ids].all(axis=0)]
uv = project(xyz[110, valid])
print("id px@t0 depth query_t")
for n, p in zip(valid, uv):
    print(int(n), tuple(np.round(p, 2)), round(float(xyz[110, n, 2]), 3), float(data["queries_xyt"][n, 2]))

out = root / "candidate_images"
out.mkdir(exist_ok=True)
for t in (108, 109, 110, 140):
    im = Image.open(io.BytesIO(bytes(data["images_jpeg_bytes"][t]))).convert("RGB")
    im.save(out / f"frame_{t}.png")
    draw = ImageDraw.Draw(im)
    for n in valid:
        if not vis[t, n]:
            continue
        x, y = project(xyz[t, n])
        if 0 <= x < im.width and 0 <= y < im.height:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red")
            draw.text((x + 4, y - 5), str(int(n)), fill="yellow")
    im.save(out / f"frame_{t}_overlay.png")

# An eight-point, history-only choice at t0=112, maximizing pixel spread.
t0 = 112
candidate = clip_ids[vis[t0 - 2:t0 + 1, clip_ids].all(axis=0)]
coords = project(xyz[t0, candidate])
chosen = [0]
while len(chosen) < 8:
    distance = np.min(np.linalg.norm(coords[:, None] - coords[chosen][None], axis=-1), axis=1)
    distance[chosen] = -1
    chosen.append(int(np.argmax(distance)))
selected = candidate[chosen]
print("selected source IDs", selected.tolist(), "positions", np.round(project(xyz[t0, selected]), 1).tolist())
print("future visible pairs", int(vis[t0+1:t0+31, selected].sum()), "/ 240; terminal", int(vis[t0+30, selected].sum()), "/ 8")
for t in (110, 111, 112, 113, 142):
    im = Image.open(io.BytesIO(bytes(data["images_jpeg_bytes"][t]))).convert("RGB")
    im.save(out / f"selected_frame_{t}.png")
    draw = ImageDraw.Draw(im)
    for n in selected:
        if not vis[t, n]:
            continue
        x, y = project(xyz[t, n])
        if 0 <= x < im.width and 0 <= y < im.height:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red")
            draw.text((x + 4, y - 5), str(int(n)), fill="yellow")
    im.save(out / f"selected_frame_{t}_overlay.png")
