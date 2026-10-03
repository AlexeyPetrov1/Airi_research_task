"""Copy proven small functions verbatim; retain independent originals for tests."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

DEST = Path(__file__).resolve().parents[1]
FUNCTIONS = {
    "fmb_wrist_math": (["transform", "project", "from_anchor"], "import numpy as np\n"),
    "fmb_wrist_evaluate": (["scores"], "import numpy as np\n"),
    "fmb_wrist_v4_evaluate": (["align"], "import numpy as np\nTIMES=np.arange(1,21)/10\n"),
    "fmb_wrist_v4_prepare": (["forecast_tcp"], "import numpy as np\nfrom scipy.spatial.transform import Rotation\n"),
    "fmb_wrist_v4_refine": (["centroid_forecast", "predict"], "import numpy as np\nfrom .fmb_wrist_math import transform\nfrom .fmb_wrist_v4_prepare import forecast_tcp\n"),
    "evaluate_author_davis": (["metrics", "parse_raw_forecast", "project"], "import numpy as np\nimport re\nfrom decimal import Decimal\n"),
    "berkeley_evaluate": (["project", "velocity", "metric_values"], "import numpy as np\n"),
    "berkeley_temporal_diagnostics": (["scale_displacement"], "import numpy as np\n"),
    "berkeley_arc_expansion": (["expand"], "import numpy as np\nfrom .berkeley_temporal_diagnostics import scale_displacement\n"),
    "berkeley_arc_phase": (["temporal_expand"], "import numpy as np\n"),
    "dobbe_vipe_geometry": (["project", "stats"], "import numpy as np\n"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-repo", type=Path, required=True)
    args = p.parse_args()
    receipts = []
    for module, (names, header) in FUNCTIONS.items():
        source_path = args.source_repo / "scripts" / (module+".py")
        source = source_path.read_text(encoding="utf8")
        tree = ast.parse(source)
        chunks = []
        for name in names:
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            chunk = ast.get_source_segment(source, node)
            chunks.append(chunk)
            receipts.append(dict(original_path=f"scripts/{module}.py", function=name,
                                 original_file_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                                 function_sha256=hashlib.sha256(chunk.encode()).hexdigest()))
        (DEST / "legacy" / (module+".py")).write_text(header+"\n\n"+"\n\n".join(chunks)+"\n", encoding="utf8")
    (DEST / "docs/borrowed_functions.json").write_text(json.dumps(receipts, indent=2)+"\n", encoding="utf8")
    print(f"Borrowed {len(receipts)} unchanged functions from {len(FUNCTIONS)} modules.")


if __name__ == "__main__":
    main()
