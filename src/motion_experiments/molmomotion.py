"""Same BF16, greedy author inference, with exact model-input checks."""
from __future__ import annotations

import re
import time
import subprocess
from decimal import Decimal

import numpy as np

from .io import read_json, sha, write_json


def strict_parse(text, history=3, horizon=30):
    match = re.fullmatch(r'<tracks coords="([^"]+)">[^<]*</tracks>\s*', text)
    if match is None:
        raise ValueError("Expected one complete tracks block")
    frames = match.group(1).split(";")
    if len(frames) != horizon:
        raise ValueError("Incomplete future horizon")
    delta = np.empty((8, horizon, 3), dtype=np.float32)
    for step, frame in enumerate(frames):
        tokens = frame.split()
        if len(tokens) != 33 or Decimal(tokens[0]) != Decimal(step+history):
            raise ValueError("Incorrect time ID or point count")
        seen = set()
        for offset in range(1, 33, 4):
            record = tokens[offset:offset+4]
            if any(re.fullmatch(r"[+-]?\d+", s) is None for s in record):
                raise ValueError("Noninteger point record")
            point, x, y, z = map(int, record)
            if point not in range(1, 9) or point in seen:
                raise ValueError("Missing or duplicate point ID")
            seen.add(point)
            delta[point-1, step] = (x/1000, y/1000, z/1000)
    return delta


def build_inputs(sample, checkpoint, dest=None):
    import torch
    from PIL import Image
    from molmo_motion import MolmoMotionProcessor

    processor = MolmoMotionProcessor.from_pretrained(str(checkpoint))
    frames = [Image.fromarray(frame).convert("RGB") for frame in sample.history_frames]
    batches, checks = [], []
    for group in range(len(sample.point_ids)//8):
        sl = slice(group*8, (group+1)*8)
        kwargs = dict(history_frames=frames, points_2d_at_t0=torch.from_numpy(sample.points_2d[sl]).float(),
                      points_3d_history=torch.from_numpy(sample.points_3d_history[:, sl]).float(),
                      action=sample.action, future_horizon=30)
        if sample.c2w_at_t0 is not None:
            kwargs["c2w_at_t0"] = torch.from_numpy(sample.c2w_at_t0).float()
        batch = processor(**kwargs)
        if sample.legacy_processor_files:
            old = torch.load(sample.legacy_processor_files[group], map_location="cpu", weights_only=False)
            if set(batch) != set(old):
                raise ValueError("Processor output keys differ from legacy")
            def same(left, right):
                if torch.is_tensor(left):
                    return bool(torch.equal(left, right))
                if isinstance(left, np.ndarray):
                    return bool(np.array_equal(left, right))
                if isinstance(left, dict):
                    return set(left) == set(right) and all(same(left[k], right[k]) for k in left)
                if isinstance(left, (list, tuple)):
                    return len(left) == len(right) and all(same(a,b) for a,b in zip(left,right))
                return left == right
            equal = {key: same(value, old[key]) for key, value in batch.items()}
        else:
            # Dobb-E did not save its original packet, but did save the exact
            # serialized input checksum plus camera/world processor equivalence.
            equivalence = read_json(next(p.parent/"processor_equivalence.json" for p in sample.model_input_files
                                        if p.name == "points_3d_world.npy"))
            import hashlib
            equal = {"input_ids_sha256": hashlib.sha256(batch["input_ids"].numpy().tobytes()).hexdigest() == equivalence["input_ids_sha256"]}
        if not all(equal.values()):
            raise ValueError(f"Model input differs from legacy: {equal}")
        checks.append(equal)
        batches.append(batch)
        if dest is not None:
            folder = dest / f"group_{group:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(batch, folder/"processor_inputs.pt")
    return batches, checks


class ModelRunner:
    """One loaded model for sequential P8 calls; no dataset logic."""
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model = None

    def run(self, sample, batches, destination):
        import resource
        import torch
        from molmo_motion import MolmoMotion

        torch.set_num_threads(4)
        torch.manual_seed(0)
        if self.model is None:
            while True:
                available = next(int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemAvailable:"))
                other = []
                for line in subprocess.check_output(["ps", "-eo", "pid,rss,args", "--no-headers"], text=True).splitlines():
                    fields = line.split(None, 2)
                    if (len(fields) == 3 and int(fields[0]) != __import__("os").getpid()
                            and "python" in fields[2] and (int(fields[1]) > 4*1024**2
                            or "das_full_motion_generate.py" in fields[2])):
                        other.append(int(fields[0]))
                if available > 16*1024**2 and not other:
                    break
                write_json(destination/"resource_wait.json", dict(status="WAITING_FOR_OTHER_WORK", other_large_python_pids=other,
                                                                  available_ram_gib=available/1024**2))
                print("Waiting for other GPU/RAM work", other, f"available RAM {available/1024**2:.1f} GiB", flush=True)
                time.sleep(20)
            start = time.monotonic()
            prior = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                self.model = MolmoMotion.from_pretrained(str(self.checkpoint))
            finally:
                torch.set_default_dtype(prior)
            self.model._internal = self.model._internal.cuda().eval()
            torch.cuda.synchronize()
            print(f"Model loaded in {time.monotonic()-start:.1f}s", flush=True)
        combined, receipts = [], []
        for group, cpu_batch in enumerate(batches):
            folder = destination/f"group_{group:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            receipt_path = folder/"model_run.json"
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                if not receipt.get("success"):
                    raise RuntimeError("Incomplete generation needs explicit inspection; it will not be silently repeated")
                if sha(folder/"future_3d.npy") != receipt["prediction_sha256"]:
                    raise ValueError("Saved prediction changed")
                combined.append(np.load(folder/"future_3d.npy"))
                receipts.append(receipt)
                continue
            torch.save(cpu_batch, folder/"processor_inputs.pt")
            batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in cpu_batch.items()}
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            write_json(receipt_path, {"success": False, "generation_started": True})
            print(f"{sample.dataset}/{sample.episode_id} group {group:02d}: fresh inference", flush=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = self.model.predict_trajectory(**batch)
            torch.cuda.synchronize()
            raw = output.future_text
            future = output.future_3d.detach().cpu().float().numpy()
            (folder/"raw_model_output.txt").write_text(raw, encoding="utf8")
            np.save(folder/"future_3d.npy", future)
            reconstructed = strict_parse(raw)+cpu_batch["anchor_3d"].numpy()
            np.testing.assert_allclose(future, reconstructed, atol=1e-7, rtol=0)
            if future.shape != (8, 30, 3) or not np.isfinite(future).all():
                raise ValueError("Incomplete neural forecast")
            receipt = dict(success=True, generation_started=True, dtype="bfloat16", seed=0,
                           decoding="official default greedy", runtime_seconds=time.monotonic()-start,
                           parsed_point_frames=240, prediction_sha256=sha(folder/"future_3d.npy"),
                           raw_sha256=sha(folder/"raw_model_output.txt"),
                           peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                           peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                           peak_ram_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
            write_json(receipt_path, receipt)
            receipts.append(receipt)
            combined.append(future)
            del batch, output
            torch.cuda.empty_cache()
        return np.concatenate(combined), receipts
