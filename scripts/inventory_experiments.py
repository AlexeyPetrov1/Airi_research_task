"""Record legacy code and artifacts before adapting the final experiments."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.source_repo.resolve()
    docs = Path(__file__).resolve().parents[1] / "docs"
    docs.mkdir(exist_ok=True)
    borrowed = {}
    for item in json.loads((docs / "borrowed_functions.json").read_text(encoding="utf8")):
        borrowed.setdefault(item["original_path"], []).append(item["function"])
    commands = {
        "fmb": "molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json",
        "berkeley": "molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json",
        "dobbe": "molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json",
        "davis": "molmo-motion-experiment --config configs/author_davis.json",
    }
    rows = []
    for folder in ("scripts", "examples", "launch_scripts"):
        for path in sorted((ROOT / folder).glob("*.py")):
            if path.name == Path(__file__).name:
                continue
            source = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source)
            name = path.name.lower()
            dataset = next((s for s in ("fmb", "berkeley", "dobbe", "davis") if s in name), "shared")
            strings = sorted({n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                              and isinstance(n.value, str) and len(n.value) < 250
                              and ("/" in n.value or n.value.endswith((".npy", ".npz", ".json", ".mp4", ".csv", ".pt",
                                  ".png", ".jpg", ".jpeg", ".gif", ".txt", ".log", ".yaml", ".yml",
                                  ".parquet", ".h5", ".hdf5", ".tfrecord", ".pkl")))})
            constants = {n.targets[0].id: ast.unparse(n.value)[:250] for n in tree.body
                         if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                         and n.targets[0].id.isupper()}
            functions = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            related = [p.name for p in (ROOT / "runs").iterdir()
                       if p.is_dir() and (dataset != "shared" and dataset in p.name)]
            relative = path.relative_to(ROOT).as_posix()
            rows.append(dict(path=relative, dataset=dataset,
                             purpose=(ast.get_docstring(tree) or name).split("\n\n")[0],
                             input_output_literals=strings, constants=constants,
                             functions=functions, imports=imports, related_saved_runs=related,
                             direct_run_path_literals=[value for value in strings if "runs/" in value],
                             disposition="Retained in place; historical preparation and diagnostics remain available.",
                             borrowed_functions=borrowed.get(relative, []),
                             replacement=("Selected functions borrowed unchanged: " + ", ".join(borrowed[relative])
                                          if relative in borrowed else "No full replacement; archival script retained"),
                             selected_experiment_command=commands.get(dataset),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (docs / "legacy_inventory.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    lines = ["# Legacy inventory", "", "All original scripts remain in place. No original run is overwritten.",
             "The JSON companion records imports, functions, constants and file literals; literals are evidence of inputs/outputs, not an inferred execution trace.",
             "The CSV companion inventories every saved research artifact by path, extension and size.", ""]
    for row in rows:
        lines += [f"## `{row['path']}`", "", row["purpose"], "",
                  f"Dataset: **{row['dataset']}**. {row['disposition']}",
                  f"Replacement: {row['replacement']}.",
                  f"Frozen selected experiment: `{row['selected_experiment_command']}`." if row['selected_experiment_command'] else "Shared/author script; see the final configurations in README.", "",
                  "Functions: " + ", ".join(f"`{f}`" for f in row["functions"]) + ".",
                  "Imports/reused functions: " + "; ".join(f"`{f}`" for f in row["imports"]),
                  "Constants: " + "; ".join(f"`{k}={v}`" for k, v in row["constants"].items()),
                  "Input/output file references: " + "; ".join(f"`{s}`" for s in row["input_output_literals"]),
                  "Direct run path literals: " + "; ".join(f"`{s}`" for s in row["direct_run_path_literals"]),
                  "Related saved experiments (dataset-level): " + ", ".join(row["related_saved_runs"]), ""]
    (docs / "legacy_inventory.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    count = 0
    with (docs / "legacy_artifacts.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "extension", "bytes"])
        for folder in ("runs", "report", "reports", "visualizations", "configs", "examples/data", "artifacts", "outputs", "logs"):
            for path in sorted((ROOT / folder).rglob("*")):
                if "__pycache__" in path.parts:
                    continue
                try:
                    if path.is_file():
                        writer.writerow([path.relative_to(ROOT).as_posix(), path.suffix, path.stat().st_size])
                        count += 1
                except OSError as error:
                    writer.writerow([path.relative_to(ROOT).as_posix(), "unresolved_link", str(error)])
    print(f"Inventoried {len(rows)} legacy scripts and {count} artifacts.")


if __name__ == "__main__":
    main()
