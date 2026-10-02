"""Export top-plane perimeter and hole wires of the official FMB boards."""

from __future__ import annotations

import json

from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX, TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

from inspect_fmb_board_step import OUT, STEP


def bounds(shape):
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    return box.Get()


def main() -> None:
    reader = STEPControl_Reader()
    if reader.ReadFile(str(STEP)) != 1:
        raise RuntimeError(f"Could not read {STEP}")
    reader.TransferRoots()
    solids = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID)
    rows = []
    while solids.More():
        solid = solids.Current()
        row = {"solid_index": len(rows), "top_faces": []}
        faces = TopExp_Explorer(solid, TopAbs_FACE)
        while faces.More():
            face = faces.Current()
            box = bounds(face)
            if (abs(box[2]-48.5) < 1e-5 and abs(box[5]-48.5) < 1e-5) or (
                    abs(box[2]+1.5) < 1e-5 and abs(box[5]+1.5) < 1e-5):
                face_data = {"z_mm": box[2], "wires": []}
                wires = TopExp_Explorer(face, TopAbs_WIRE)
                while wires.More():
                    wire = wires.Current()
                    vertices = TopExp_Explorer(wire, TopAbs_VERTEX)
                    points = set()
                    while vertices.More():
                        p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(vertices.Current()))
                        points.add((round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6)))
                        vertices.Next()
                    coords = sorted(points)
                    box_wire = bounds(wire)
                    face_data["wires"].append({"center_vertex_mean_mm":
                        [sum(p[i] for p in coords)/len(coords) for i in range(3)],
                        "bounds_mm": list(box_wire),
                        "bbox_center_mm": [(box_wire[i]+box_wire[i+3])/2 for i in range(3)],
                        "vertices_mm": coords})
                    wires.Next()
                row["top_faces"].append(face_data)
            faces.Next()
        rows.append(row)
        solids.Next()
    target = OUT.with_name("peg_board_top_wires.json")
    target.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf8")
    for row in rows:
        print("solid", row["solid_index"], "flat faces", len(row["top_faces"]),
              "z,wires", [(f["z_mm"], len(f["wires"])) for f in row["top_faces"]])
        for face in row["top_faces"]:
            for i, wire in enumerate(face["wires"]):
                c = wire["bbox_center_mm"]
                b = wire["bounds_mm"]
                print(i, "bbox center", [round(x, 1) for x in c[:2]],
                      "size", [round(b[d+3]-b[d], 1) for d in range(2)],
                      "vertices", len(wire["vertices_mm"]))


if __name__ == "__main__":
    main()
