"""Inspect the 54 solids in the official FMB peg STEP assembly."""

from __future__ import annotations

import json
from pathlib import Path

from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.STEPControl import STEPControl_Reader
from OCP.BRep import BRep_Tool
from OCP.TopAbs import TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS


ROOT = Path(__file__).resolve().parents[1]
STEP = ROOT / "runs/sharerobot_fmb_episode_5201/peg_official.step"
OUT = STEP.with_name("peg_official_solids.json")


def main() -> None:
    reader = STEPControl_Reader()
    if reader.ReadFile(str(STEP)) != 1:
        raise RuntimeError(f"Could not read {STEP}")
    reader.TransferRoots()
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID)
    rows = []
    while explorer.More():
        shape = explorer.Current()
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        rows.append({
            "index": len(rows),
            "bounds_mm": [xmin, ymin, zmin, xmax, ymax, zmax],
            "size_mm": [xmax - xmin, ymax - ymin, zmax - zmin],
        })
        vertices = TopExp_Explorer(shape, TopAbs_VERTEX)
        points = set()
        while vertices.More():
            p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(vertices.Current()))
            points.add((round(p.X(), 5), round(p.Y(), 5), round(p.Z(), 5)))
            vertices.Next()
        rows[-1]["vertices_mm"] = sorted(points)
        explorer.Next()
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        print(row["index"], [round(v, 2) for v in row["size_mm"]],
              [round(v, 1) for v in row["bounds_mm"][:3]])
    print("solids", len(rows), "saved", OUT)


if __name__ == "__main__":
    main()
