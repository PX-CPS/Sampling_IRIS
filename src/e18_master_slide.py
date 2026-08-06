#!/usr/bin/env python3
"""E18: the one-slide master figure, as an EDITABLE .pptx.

Three acts on one 16:9 slide, every element a native PowerPoint shape:
  I   the birth of the probabilistic map (IrisZo, eps>0 by construction)
  II  the amplification law: 3.2% volume -> 12.5% random -> 62% task,
      with the three mechanisms as cards
  III the certified layer pipeline, with the two lineages highlighted
      (safety certificates: one distance query frees a whole ball;
       lazy collision checking: only validate the edges the answer
       uses), the repair/blacklist loop, and the E16 closing numbers.

    python3 e18_master_slide.py    # out/certgcs_master_figure.pptx
"""
from __future__ import annotations

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

C_TXT = "1F2937"
C_BLUE_F, C_BLUE_L = "E8F1FD", "1D4ED8"
C_RED_F1, C_RED_F2, C_RED_F3 = "FDECEC", "F9D2D2", "F3B0B0"
C_RED_L = "B91C1C"
C_GRN_F, C_GRN_L = "E7F6EE", "047857"
C_AMB_F, C_AMB_L = "FFF6DF", "B45309"
C_GRY_F, C_GRY_L = "F4F4F5", "52525B"


def _arrow(conn, dashed=False):
    ln = conn.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"),
                          {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    if dashed:
        d = ln.makeelement(qn("a:prstDash"), {"val": "dash"})
        ln.insert(0, d)


def arrow(shapes, x1, y1, x2, y2, color=C_TXT, w=1.6, dashed=False):
    c = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1),
                             Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor.from_string(color)
    c.line.width = Pt(w)
    _arrow(c, dashed)
    return c


def box(shapes, x, y, w, h, fill, line, lines, shape=None, lw=1.2,
        align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, radius=None):
    sp = shapes.add_shape(shape or MSO_SHAPE.ROUNDED_RECTANGLE,
                          Inches(x), Inches(y), Inches(w), Inches(h))
    if radius is not None:
        try:
            sp.adjustments[0] = radius
        except Exception:
            pass
    sp.fill.solid()
    sp.fill.fore_color.rgb = RGBColor.from_string(fill)
    sp.line.color.rgb = RGBColor.from_string(line)
    sp.line.width = Pt(lw)
    sp.shadow.inherit = False
    tf = sp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top",
              "margin_bottom"):
        setattr(tf, m, Pt(4))
    first = True
    for txt, size, bold, color in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = RGBColor.from_string(color)
    return sp


def text(shapes, x, y, w, h, lines, align=PP_ALIGN.LEFT):
    sp = shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sp.text_frame
    tf.word_wrap = True
    first = True
    for txt, size, bold, color in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = RGBColor.from_string(color)
    return sp


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    sh = s.shapes

    # ---------------- title ----------------
    text(sh, 0.15, 0.04, 13.0, 0.5, [
        ("CertGCS at a glance — why planner outputs must be "
         "validated, and why validation is cheap", 19, True,
         C_TXT)])

    # ---------------- ACT I ----------------
    box(sh, 0.15, 0.56, 3.5, 0.30, C_GRY_F, C_GRY_L,
        [("① Birth of the probabilistic map (offline)", 11.5, True,
          C_TXT)],
        shape=MSO_SHAPE.RECTANGLE, lw=0.75)
    box(sh, 0.15, 0.92, 3.5, 0.78, C_BLUE_F, C_BLUE_L, [
        ("IrisZo convex decomposition", 10.5, True, C_TXT),
        ("point oracle + hyperplane cuts + Bernoulli (ε,δ) gate", 9,
         False, C_TXT),
        ("non-convex C-obstacles ⇒ every cut is only local", 9,
         False, C_RED_L)])
    arrow(sh, 1.9, 1.72, 1.9, 1.94)
    box(sh, 0.15, 1.96, 3.5, 0.86, C_BLUE_F, C_BLUE_L, [
        ("Region map: 97% of the volume truly free", 10, True,
         C_TXT),
        ("+ 3.2% residual pockets — ε>0 by construction", 9, False,
         C_RED_L),
        ("pockets live on boundary shells / door jambs", 9, False,
         C_TXT)])
    # wedge glyph
    tri = sh.add_shape(MSO_SHAPE.RIGHT_TRIANGLE, Inches(3.28),
                       Inches(2.44), Inches(0.28), Inches(0.30))
    tri.fill.solid()
    tri.fill.fore_color.rgb = RGBColor.from_string("DC2626")
    tri.line.fill.background()
    tri.rotation = 180

    # ---------------- ACT II ----------------
    box(sh, 3.85, 0.56, 9.33, 0.30, C_GRY_F, C_GRY_L,
        [("② The amplification law — the optimizer is a search "
          "engine for model errors, not a sampler", 11.5, True,
          C_TXT)], shape=MSO_SHAPE.RECTANGLE, lw=0.75)
    stat = [("3.2%", "volume error",
             "uniform measure: darts into the regions", C_RED_F1),
            ("12.5%", "random-query answers invalid",
             "2/16 benchmark optima, two machines", C_RED_F2),
            ("62%", "task-query answers invalid",
             "18/29, deepest −91 mm", C_RED_F3)]
    xs = [3.85, 7.13, 10.41]
    for (v, t1, t2, f), x in zip(stat, xs):
        box(sh, x, 0.92, 2.77, 0.80, f, C_RED_L, [
            (v + "  " + t1, 11.5, True, C_RED_L),
            (t2, 9, False, C_TXT)], align=PP_ALIGN.CENTER,
            anchor=MSO_ANCHOR.MIDDLE)
    arrow(sh, 6.63, 1.32, 7.12, 1.32, color=C_RED_L, w=2.2)
    arrow(sh, 9.91, 1.32, 10.40, 1.32, color=C_RED_L, w=2.2)
    text(sh, 6.56, 0.98, 0.62, 0.3, [("×4", 11, True, C_RED_L)],
         align=PP_ALIGN.CENTER)
    text(sh, 9.84, 0.98, 0.62, 0.3, [("×5", 11, True, C_RED_L)],
         align=PP_ALIGN.CENTER)
    mech = [("① Taut paths patrol the boundary",
             "a shortest path hugs walls and cuts corners — exactly "
             "where the pockets live"),
            ("② The argmin harvests model surplus",
             "if the through-pocket shortcut is shorter, EVERY modeled"
             " optimum takes it; the exacter the solver, the surer"),
            ("③ Tasks live in the dirty real estate",
             "grasp poses hug the shelves = where the generator "
             "truncated deepest; task corridors follow them")]
    for (t1, t2), x in zip(mech, xs):
        box(sh, x, 1.80, 2.77, 0.72, "FFFFFF", C_RED_L, [
            (t1, 9.5, True, C_RED_L), (t2, 8.5, False, C_TXT)], lw=0.9)
    box(sh, 3.85, 2.58, 9.33, 0.30, C_GRY_F, C_RED_L, [
        ("Measure mismatch: audits sample the uniform measure, the "
         "planner samples the argmin measure — the only aligned test "
         "is a test of the argmin itself (↓ our layer)", 10, True,
         C_RED_L)],
        shape=MSO_SHAPE.RECTANGLE, lw=0.9)

    # ---------------- ACT III ----------------
    box(sh, 0.15, 2.96, 13.03, 0.30, C_GRY_F, C_GRY_L,
        [("③ The certified layer (online) — the necessary thing, "
          "done the cheap way", 11.5, True, C_TXT)],
        shape=MSO_SHAPE.RECTANGLE, lw=0.75)
    flow = [
        ("Region map", "from ① — 97% clean\n= why checking is cheap",
         C_BLUE_F, C_BLUE_L),
        ("Interface sampling\n+ informed pruning",
         "admissible LB keys; pruned-\nsubgraph LB stays valid",
         C_BLUE_F, C_BLUE_L),
        ("Search → candidate", "anytime incumbent;\nmonotone improvement",
         C_BLUE_F, C_BLUE_L),
        ("Validate ONLY\nthe candidate",
         "lineage 2, lazy: never the\ngraph — just the answer",
         C_AMB_F, C_AMB_L),
        ("Certificate-ball chain\ncovers the whole path",
         "lineage 1: one distance\nquery frees a whole ball",
         C_AMB_F, C_AMB_L),
        ("SOCP polish →\nre-validate",
         "convex optimum in corridor;\npolish is an argmin too",
         C_BLUE_F, C_BLUE_L),
        ("Certified path + ε-cert\nor honest no-path",
         "LB ≤ c* ≤ certified UB;\ndegrades gracefully",
         C_GRN_F, C_GRN_L),
    ]
    fx, fw, fy, fh, gap = 0.15, 1.72, 3.34, 1.00, 0.165
    for i, (t1, t2, f, l) in enumerate(flow):
        x = fx + i * (fw + gap)
        box(sh, x, fy, fw, fh, f, l, [
            (t1, 9.5, True, C_TXT if f != C_GRN_F else "065F46"),
            (t2, 8, False, C_TXT)], lw=1.4 if f == C_AMB_F else 1.2)
        if i:
            arrow(sh, x - gap + 0.005, fy + fh / 2, x - 0.005,
                  fy + fh / 2, w=1.8)
    arrow(sh, 1.9, 2.82, 1.9, 3.33)          # Act I -> pipeline

    # repair / blacklist loop
    bx4 = fx + 4 * (fw + gap)
    box(sh, 3.2, 4.60, 5.4, 0.62, "FFFFFF", C_RED_L, [
        ("dirty segment → in-region repair (convexity keeps via "
         "points legal) → unfixable: blacklist the edge", 9.5, True,
         C_RED_L),
        ("the lazy loop: delete → re-search → re-validate, until "
         "certified or honestly infeasible", 8.5, False, C_TXT)],
        lw=1.1)
    arrow(sh, bx4 + fw / 2, fy + fh, 8.1, 4.60, color=C_RED_L)
    arrow(sh, 3.35, 4.60, 4.45, fy + fh + 0.02, color=C_RED_L,
          dashed=True)

    # lineage cards
    box(sh, 0.15, 5.32, 6.42, 1.30, C_AMB_F, C_AMB_L, [
        ("Lineage 2 · lazy collision checking — LazyPRM (Bohlin & "
         "Kavraki '00) / LazySP (Dellin & Srinivasa '16)", 9, True,
         C_AMB_L),
        ("\u201cNavigate first, repair later — only the roads the "
         "route uses.\u201d", 9.5, True, C_TXT),
        ("Most edges are never used by any shortest path → assume all"
         " valid, search, then check only the returned path's edges; "
         "delete failures, re-search.", 8, False, C_TXT),
        ("Our twist: the skeleton is the IRIS adjacency graph (no PRM"
         " to build), and what gets validated is the argmin of convex"
         " optimization — the antidote to the amplification law.", 8,
         False, C_AMB_L)], lw=1.4)
    box(sh, 6.76, 5.32, 6.42, 1.30, C_AMB_F, C_AMB_L, [
        ("Lineage 1 · safety certificates — Bialkowski/Karaman/"
         "Frazzoli (WAFR'12/IJRR'16) → Safe Bubble Cover (ISRR'24)",
         9, True, C_AMB_L),
        ("\u201cOne distance query, one ball exempt\u201d: clearance"
         " φ(q) ⇒ the ball of radius φ(q)/L is provably safe "
         "(L = robot Lipschitz constant).", 9.5, True, C_TXT),
        ("Overlapping balls chained along the segment = a continuous "
         "proof: no resolution parameter, no loophole; long strides "
         "in the open, fine steps near obstacles.", 8, False, C_TXT),
        ("Measured: ~200 calls × 2.3 ms per path, certificates in "
         "0.02–0.06 s — cheap because the map is 97% clean (the "
         "dividend of ①).", 8, False, C_AMB_L)], lw=1.4)
    # ---------------- results strip ----------------
    box(sh, 0.15, 6.74, 13.03, 0.68, C_GRN_F, C_GRN_L, [
        ("E16 head-to-head (29 task queries, same library & machine):"
         " reference delivers 21/29 invalid (18 probe-confirmed) vs "
         "ours 0 invalid (15 certified + 14 honest no-paths) · first "
         "certified answer 1.14 s < 1.59 s for the unvalidated one · "
         "cost ratio 1.0000 on clean pairs · price of truth only "
         "where the reference is wrong: median +14%", 9, True,
         "065F46"),
        ("All three routes to a stricter library fail (strict gate: "
         "47 regions, 0 portals / C-IRIS: 7 islands / margins: "
         "contamination flat, connectivity lost first) — volume-level"
         " certainty is unaffordable; the path-level certificate does"
         " the necessary thing, the cheap way.", 8, False, C_TXT)],
        lw=1.4)

    out = "out/certgcs_master_figure.pptx"
    prs.save(out)
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
