"""One pre-reconstruction correction: source-faithful action text."""
from dobbe_vipe_v1 import RUN, SCENES, read, write, sha

for scene in ["A", "B"]:
    folder = RUN / scene
    assert not (folder / "vipe").exists() and not (folder / "tracks.npz").exists()
    path = folder / "protocol.json"
    prior = read(path)
    if prior["action"] != SCENES[scene]["action"]:
        write(folder / "protocol_action_correction.json", {
            "prior_protocol_sha256": sha(path), "prior_action": prior["action"],
            "action": SCENES[scene]["action"],
            "reason": "A: reuse prior exact instruction. B: identify tape roll from observed prefix only.",
            "before_reconstruction_tracking_or_prediction": True})
        prior["action"] = SCENES[scene]["action"]
        write(path, prior)
