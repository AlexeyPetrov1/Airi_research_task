"""Run all predeclared A controls before opening any A future frames."""
from dobbe_vipe_inference import infer, export_variants
from prepare_dobbe_vipe_depth_pair import prepare_pair

if __name__ == "__main__":
    export_variants("A", "sift_audited")
    prepare_pair()
    cache = {}
    for variant in ["smoke_vipe", "pure_vipe", "pure_vipe_unsmoothed", "hybrid", "paired_vipe", "paired_hybrid"]:
        infer("A", variant, "sift_audited", reuse=cache)
