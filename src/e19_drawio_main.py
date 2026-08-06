#!/usr/bin/env python3
"""E19: the paper's main figure as native draw.io XML.

Visual-first three-panel narrative on the SAME mini floor plan:
  1  the map lies a little   (blue regions, red slivers at jambs)
  2  the optimizer finds it  (taut red dashed path clips the jamb;
                              3.2% -> argmin -> 62% amplifier)
  3  validate the answer     (green path through the door, covered by
                              a chain of certificate balls; result chip)
plus a two-card lineage footer.  <= 13 short labels, no paragraphs.

    python3 e19_drawio_main.py    # out/certgcs_main.drawio
"""
from __future__ import annotations

from xml.sax.saxutils import escape

CELLS = []
_id = [1]


def nid():
    _id[0] += 1
    return f"n{_id[0]}"


def vertex(x, y, w, h, style, value=""):
    i = nid()
    CELLS.append(
        f'<mxCell id="{i}" value="{escape(value)}" style="{style}" '
        f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" '
        f'width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i


def edge(points, style, value=""):
    i = nid()
    (x1, y1), (x2, y2) = points[0], points[-1]
    mids = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in
                   points[1:-1])
    arr = f'<Array as="points">{mids}</Array>' if mids else ""
    CELLS.append(
        f'<mxCell id="{i}" value="{escape(value)}" style="{style}" '
        f'edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
        f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
        f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/>{arr}'
        f"</mxGeometry></mxCell>")
    return i


S_TXT = ("text;html=1;align=left;verticalAlign=top;fontSize=%d;"
         "fontStyle=%d;fontColor=%s;")
S_PANEL = ("rounded=1;arcSize=4;fillColor=#FFFFFF;strokeColor=#CBD5E1;"
           "strokeWidth=1.5;")
S_WALL = "fillColor=#3F3F46;strokeColor=#18181B;strokeWidth=1;"
S_REGION = ("rounded=1;arcSize=8;fillColor=#93C5FD;opacity=22;"
            "strokeColor=#1D4ED8;strokeWidth=1;")
S_WEDGE = ("triangle;html=1;fillColor=#DC2626;strokeColor=none;"
           "direction=%s;")
S_BALL = ("ellipse;fillColor=#A7F3D0;opacity=45;strokeColor=#059669;"
          "strokeWidth=1.5;")
S_RED = ("endArrow=none;dashed=1;dashPattern=8 6;strokeColor=#DC2626;"
         "strokeWidth=3;html=1;rounded=1;")
S_GRN = ("endArrow=none;strokeColor=#059669;strokeWidth=3.5;html=1;"
         "rounded=1;")
S_DOT = "ellipse;fillColor=%s;strokeColor=none;"
S_CHIP = ("rounded=1;arcSize=18;fillColor=#E7F6EE;strokeColor=#047857;"
          "strokeWidth=1.5;fontSize=13;fontStyle=1;fontColor=#065F46;"
          "align=center;")
S_CARD = ("rounded=1;arcSize=10;fillColor=#FFF6DF;strokeColor=#B45309;"
          "strokeWidth=1.5;fontSize=13;fontColor=#7C4A03;align=left;"
          "verticalAlign=middle;spacingLeft=10;html=1;")


def minimap(ox, oy):
    """One floor-plan copy at panel origin; returns key coordinates."""
    # regions first (under everything)
    for x, y, w, h in ((0, 6, 195, 158), (0, 158, 205, 146),
                       (212, 16, 168, 148), (206, 172, 174, 124),
                       (148, 104, 118, 96)):
        vertex(ox + x, oy + y, w, h, S_REGION)
    # walls: vertical with a door, horizontal with a door
    vertex(ox + 182, oy + 0, 16, 118, S_WALL)      # V-top
    vertex(ox + 182, oy + 168, 16, 136, S_WALL)    # V-bottom
    vertex(ox + 198, oy + 148, 92, 14, S_WALL)     # H-left
    vertex(ox + 324, oy + 148, 56, 14, S_WALL)     # H-right
    # residual slivers at jambs / corners
    vertex(ox + 168, oy + 106, 14, 13, S_WEDGE % "west")
    vertex(ox + 198, oy + 162, 15, 12, S_WEDGE % "south")
    vertex(ox + 310, oy + 137, 13, 12, S_WEDGE % "north")
    return dict(start=(ox + 16, oy + 288), goal=(ox + 366, oy + 22),
                jamb=(ox + 176, oy + 112), door_v=(ox + 190, oy + 143))


def main():
    # ---------------- title ----------------
    vertex(20, 8, 1360, 30, S_TXT % (24, 1, "#1F2937"),
           "CertGCS — validate the answer, not the volume")

    P_W, P_H, P_Y = 440, 400, 60
    px = [30, 500, 970]

    # ================= panel 1 =================
    vertex(px[0], P_Y, P_W, P_H, S_PANEL)
    vertex(px[0] + 12, P_Y + 8, P_W - 24, 24,
           S_TXT % (16, 1, "#1F2937"), "① the map lies a little")
    m = minimap(px[0] + 30, P_Y + 60)
    # zoom bubble: the jamb sliver, drawn big in clear space
    edge([(m["jamb"][0] + 22, m["jamb"][1] - 14),
          (px[0] + 300, P_Y + 118)],
         "endArrow=none;strokeColor=#7F1D1D;strokeWidth=1.5;html=1;")
    vertex(px[0] + 286, P_Y + 46, 116, 116,
           "ellipse;fillColor=#FFFFFF;strokeColor=#7F1D1D;"
           "strokeWidth=2;")
    vertex(px[0] + 300, P_Y + 100, 40, 20, S_WALL)
    vertex(px[0] + 356, P_Y + 100, 34, 20, S_WALL)
    vertex(px[0] + 338, P_Y + 78, 24, 22, S_WEDGE % "south")
    vertex(px[0] + 296, P_Y + 126, 100, 20,
           S_TXT % (11, 0, "#7F1D1D"), "residual sliver")
    vertex(m["jamb"][0] - 20, m["jamb"][1] - 20, 46, 46,
           "ellipse;fillColor=none;strokeColor=#7F1D1D;strokeWidth=2;"
           "dashed=1;")
    vertex(px[0] + 40, P_Y + 356, 360, 26, S_TXT % (13, 0, "#B91C1C"),
           "ε > 0 residual — 3.2% of the volume, at walls and jambs")

    # ================= panel 2 =================
    vertex(px[1], P_Y, P_W, P_H, S_PANEL)
    vertex(px[1] + 12, P_Y + 8, P_W - 24, 24,
           S_TXT % (16, 1, "#1F2937"), "② the optimizer finds the lie")
    m = minimap(px[1] + 30, P_Y + 60)
    edge([m["start"], (px[1] + 120, P_Y + 250),
          (m["jamb"][0] + 6, m["jamb"][1] + 6),
          (px[1] + 268, P_Y + 128), m["goal"]], S_RED)
    vertex(m["start"][0] - 7, m["start"][1] - 7, 14, 14,
           S_DOT % "#16A34A")
    vertex(m["goal"][0] - 7, m["goal"][1] - 7, 14, 14,
           S_DOT % "#DC2626")
    # amplifier strip under the map
    vertex(px[1] + 74, P_Y + 344, 40, 40,
           "ellipse;fillColor=#DC2626;strokeColor=none;fontSize=11;"
           "fontColor=#FFFFFF;fontStyle=1;align=center;"
           "verticalAlign=middle;", "3.2%")
    edge([(px[1] + 120, P_Y + 364), (px[1] + 196, P_Y + 364)],
         "endArrow=block;strokeColor=#B91C1C;strokeWidth=3;html=1;"
         "fontSize=12;fontStyle=1;fontColor=#B91C1C;", "argmin")
    vertex(px[1] + 202, P_Y + 336, 212, 50,
           "rounded=1;arcSize=10;fillColor=#F3B0B0;strokeColor="
           "#B91C1C;strokeWidth=1.5;fontSize=13;fontStyle=1;"
           "fontColor=#7F1D1D;align=center;",
           "62% of task answers invalid")

    # ================= panel 3 =================
    vertex(px[2], P_Y, P_W, P_H, S_PANEL)
    vertex(px[2] + 12, P_Y + 8, P_W - 24, 24,
           S_TXT % (16, 1, "#1F2937"),
           "③ validate the answer, not the volume")
    m = minimap(px[2] + 30, P_Y + 60)
    gpath = [m["start"], (px[2] + 120, P_Y + 268),
             (px[2] + 196, P_Y + 216), m["door_v"],
             (px[2] + 268, P_Y + 128), m["goal"]]
    # certificate balls first (under the path), sizes adapt to walls
    balls = [(m["start"], 26), ((px[2] + 88, P_Y + 292), 30),
             ((px[2] + 138, P_Y + 252), 26),
             ((px[2] + 178, P_Y + 232), 20),
             ((px[2] + 200, P_Y + 200), 16),
             (m["door_v"], 14), ((px[2] + 232, P_Y + 166), 18),
             ((px[2] + 268, P_Y + 128), 24),
             ((px[2] + 320, P_Y + 76), 30), (m["goal"], 24)]
    for (cx, cy), r in balls:
        vertex(cx - r, cy - r, 2 * r, 2 * r, S_BALL)
    edge(gpath, S_GRN)
    vertex(m["start"][0] - 7, m["start"][1] - 7, 14, 14,
           S_DOT % "#16A34A")
    vertex(m["goal"][0] - 7, m["goal"][1] - 7, 14, 14,
           S_DOT % "#DC2626")
    vertex(px[2] + 34, P_Y + 306, 220, 22, S_TXT % (12, 0, "#047857"),
           "ball chain = continuous proof")
    vertex(px[2] + 118, P_Y + 362, 260, 30, S_CHIP,
           "0 / 29 invalid  ·  first answer 1.14 s  ·  +2.8%")

    # ================= lineage footer =================
    vertex(30, 480, 660, 56, S_CARD,
           "          one distance query, one ball exempt — "
           "safety certificates  (φ(q)/L)")
    vertex(44, 494, 28, 28, S_BALL)
    edge([(58, 508), (78, 496)],
         "endArrow=open;strokeColor=#059669;strokeWidth=2;html=1;")
    vertex(750, 480, 660, 56, S_CARD,
           "             check only the roads the route uses — "
           "lazy collision checking")
    for k, (c, wdt) in enumerate((("#CBD5E1", 2), ("#059669", 3),
                                  ("#CBD5E1", 2))):
        edge([(764, 496 + 12 * k), (806, 496 + 12 * k)],
             f"endArrow=none;strokeColor={c};strokeWidth={wdt};html=1;")

    xml = ('<mxfile host="app.diagrams.net"><diagram id="certgcs" '
           'name="CertGCS main figure"><mxGraphModel dx="1000" '
           'dy="600" grid="0" gridSize="10" guides="1" tooltips="1" '
           'connect="1" arrows="1" fold="1" page="1" pageScale="1" '
           'pageWidth="1440" pageHeight="560" math="0" shadow="0">'
           '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
           + "".join(CELLS) +
           "</root></mxGraphModel></diagram></mxfile>")
    out = "out/certgcs_main.drawio"
    with open(out, "w") as f:
        f.write(xml)
    import xml.dom.minidom as md
    md.parseString(xml)
    print(f"written {out} ({len(CELLS)} cells, well-formed XML)",
          flush=True)


if __name__ == "__main__":
    main()
