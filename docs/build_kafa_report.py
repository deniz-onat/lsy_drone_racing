"""Generate an explainable PDF report on the KaFa v9.* and v10 MPCC controllers.

Run:  python3 docs/build_kafa_report.py
Output: docs/KaFa_v9_v10_explained.pdf
"""

from __future__ import annotations

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.graphics.shapes import Drawing, Group, Line, Polygon, Rect, String
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = os.path.join(os.path.dirname(__file__), "KaFa_v9_v10_explained.pdf")

# ----------------------------------------------------------------------------- palette
INK = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#0f4c81")
ACCENT2 = colors.HexColor("#2a9d8f")
LIGHT = colors.HexColor("#eef3f8")
CODEBG = colors.HexColor("#f4f4f0")
MUTED = colors.HexColor("#5a5a6e")
BOXBG = colors.HexColor("#fff8e7")
BOXBORDER = colors.HexColor("#e0a800")

# ----------------------------------------------------------------------------- styles
ss = getSampleStyleSheet()


def style(name, **kw):
    return ParagraphStyle(name, parent=ss["Normal"], **kw)


S_TITLE = style("t", fontName="Helvetica-Bold", fontSize=26, leading=30, textColor=ACCENT)
S_SUB = style("s", fontName="Helvetica", fontSize=13, leading=18, textColor=MUTED)
S_H1 = style("h1", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=ACCENT,
             spaceBefore=18, spaceAfter=6)
S_H2 = style("h2", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=INK,
             spaceBefore=12, spaceAfter=4)
S_BODY = style("b", fontName="Helvetica", fontSize=10.2, leading=15, textColor=INK,
               alignment=TA_JUSTIFY, spaceAfter=6)
S_BULLET = style("bu", fontName="Helvetica", fontSize=10.2, leading=14.5, textColor=INK,
                 alignment=TA_LEFT)
S_CODE = style("c", fontName="Courier", fontSize=8.6, leading=11.5, textColor=colors.HexColor("#22313f"))
S_CODECAP = style("cc", fontName="Helvetica-Oblique", fontSize=8.6, leading=11, textColor=MUTED,
                  spaceBefore=2, spaceAfter=8)
S_BOXH = style("bh", fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=colors.HexColor("#7a5a00"))
S_BOX = style("bx", fontName="Helvetica", fontSize=9.8, leading=13.5, textColor=INK, alignment=TA_JUSTIFY)
S_CAP = style("cap", fontName="Helvetica-Oblique", fontSize=9, leading=12, textColor=MUTED,
              alignment=TA_CENTER, spaceBefore=3)
S_TH = style("th", fontName="Helvetica-Bold", fontSize=8.8, leading=11, textColor=colors.white)
S_TD = style("td", fontName="Helvetica", fontSize=8.8, leading=11, textColor=INK)
S_TDB = style("tdb", fontName="Helvetica-Bold", fontSize=8.8, leading=11, textColor=INK)
S_FOOT = style("ft", fontName="Helvetica", fontSize=8, textColor=MUTED, alignment=TA_CENTER)

DIAG_W = 16.0 * cm  # usable content width for drawings

E = []  # story


# ----------------------------------------------------------------------------- diagram primitives
def _wrap(text, max_chars):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars or not cur:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def dbox(d, x, y, w, h, text, fill, stroke=None, tcolor=colors.white, fs=9, bold=True,
         max_chars=22):
    """Draw a rounded box with centred, auto-wrapped, multi-line text."""
    stroke = stroke or fill
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=fill, strokeColor=stroke, strokeWidth=1))
    lines = []
    for seg in text.split("\n"):
        lines += _wrap(seg, max_chars) if seg else [""]
    fn = "Helvetica-Bold" if bold else "Helvetica"
    total = len(lines) * (fs + 2)
    ty = y + h / 2 + total / 2 - fs
    for ln in lines:
        d.add(String(x + w / 2, ty, ln, fontName=fn, fontSize=fs, fillColor=tcolor,
                     textAnchor="middle"))
        ty -= fs + 2


def _arrowhead(d, x, y, ang, color, size=6):
    import math
    a = math.radians(ang)
    p1 = (x, y)
    p2 = (x - size * math.cos(a - 0.4), y - size * math.sin(a - 0.4))
    p3 = (x - size * math.cos(a + 0.4), y - size * math.sin(a + 0.4))
    d.add(Polygon([p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]], fillColor=color, strokeColor=color))


def darrow(d, x1, y1, x2, y2, color=INK, w=1.4, dash=None, label=None, lcolor=MUTED):
    import math
    ln = Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=w)
    if dash:
        ln.strokeDashArray = dash
    d.add(ln)
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    _arrowhead(d, x2, y2, ang, color)
    if label:
        d.add(String((x1 + x2) / 2, (y1 + y2) / 2 + 4, label, fontName="Helvetica",
                     fontSize=7.5, fillColor=lcolor, textAnchor="middle"))


def diagram(height, draw_fn, caption=None):
    """Append a Drawing (width DIAG_W) built by draw_fn(d), with an optional caption."""
    d = Drawing(DIAG_W, height)
    d.add(Rect(0, 0, DIAG_W, height, fillColor=colors.white, strokeColor=colors.white))
    draw_fn(d)
    E.append(d)
    if caption:
        E.append(Paragraph(caption, S_CAP))
    gap(8)


# ----------------------------------------------------------------------------- the diagrams
C_PLAN = colors.HexColor("#0f4c81")
C_MPCC = colors.HexColor("#2a9d8f")
C_ATT = colors.HexColor("#6a4c93")
C_DRONE = colors.HexColor("#1a1a2e")
C_NEW = colors.HexColor("#e07a00")
C_GREY = colors.HexColor("#8a93a0")
W = DIAG_W


def draw_pipeline(d):
    """The per-step control loop: observation -> planner -> MPCC -> attitude -> drone -> back."""
    y = 78
    bw, bh = 3.05 * cm, 1.5 * cm
    xs = [0.15 * cm, 3.55 * cm, 7.1 * cm, 10.65 * cm, 13.0 * cm]
    dbox(d, xs[0], y, bw, bh, "Observation\n(drone pose,\ngates, obstacles)", C_GREY, fs=8.5)
    dbox(d, xs[1], y, bw, bh, "v8 Planner\ngate-aware spline\n(the PATH)", C_PLAN, fs=8.5)
    dbox(d, xs[2], y, bw, bh, "MPCC\nplan accelerations\n(the SPEED)", C_MPCC, fs=8.5)
    dbox(d, xs[3], y, 2.0 * cm, bh, "to attitude\nm·(a+g)", C_ATT, fs=8.5)
    dbox(d, xs[4], y, 2.85 * cm, bh, "Drone\n[roll,pitch,\nyaw,thrust]", C_DRONE, fs=8.5)
    for i in range(4):
        x1 = xs[i] + (2.0 * cm if i == 3 else bw)
        darrow(d, x1, y + bh / 2, xs[i + 1], y + bh / 2, color=INK)
    # feedback loop drone -> observation
    darrow(d, xs[4] + 1.4 * cm, y, xs[4] + 1.4 * cm, 20, color=C_GREY, w=1.2)
    darrow(d, xs[4] + 1.4 * cm, 20, xs[0] + 1.5 * cm, 20, color=C_GREY, w=1.2)
    darrow(d, xs[0] + 1.5 * cm, 20, xs[0] + 1.5 * cm, y, color=C_GREY, w=1.2,
           label="50 Hz feedback — re-plan every step")
    d.add(String(xs[1] + bw / 2, y + bh + 6, "WHERE to fly", fontName="Helvetica-Oblique",
                 fontSize=8, fillColor=C_PLAN, textAnchor="middle"))
    d.add(String(xs[2] + bw / 2, y + bh + 6, "HOW FAST to fly", fontName="Helvetica-Oblique",
                 fontSize=8, fillColor=C_MPCC, textAnchor="middle"))


def draw_geometry(d):
    """Contouring vs lag error: path, drone point, reference point, perpendicular & tangent."""
    import math
    from reportlab.graphics.shapes import Circle
    RED = colors.HexColor("#c1121f")
    # gentle path arc sitting in the upper-middle band
    pts = []
    for i in range(60):
        t = i / 59.0
        x = 1.2 * cm + t * 13.0 * cm
        yy = 112 + 14 * math.sin(t * 2.0 - 0.3)
        pts.append((x, yy))
    for i in range(len(pts) - 1):
        d.add(Line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1],
                   strokeColor=C_PLAN, strokeWidth=2.4))
    d.add(String(pts[-1][0], pts[-1][1] + 7, "path", fontName="Helvetica-Oblique",
                 fontSize=8.5, fillColor=C_PLAN, textAnchor="end"))
    # reference foot point on the path + unit tangent
    ri = 30
    rx, ry = pts[ri]
    tx, ty = pts[ri + 3][0] - pts[ri - 3][0], pts[ri + 3][1] - pts[ri - 3][1]
    tn = math.hypot(tx, ty)
    tx, ty = tx / tn, ty / tn
    nx, ny = -ty, tx  # unit normal
    # drone: lag back along tangent and offset sideways along the normal
    LAG, CON = 78.0, 52.0
    px, py = rx - LAG * tx - CON * nx, ry - LAG * ty - CON * ny
    fpx, fpy = px + LAG * tx, py + LAG * ty  # foot of perpendicular (lag end / contour start)
    # tangent guide (dashed) through the reference
    darrow(d, rx - 95 * tx, ry - 95 * ty, rx + 34 * tx, ry + 34 * ty, color=C_PLAN, w=0.9,
           dash=(3, 3))
    d.add(String(rx + 38 * tx, ry + 34 * ty - 2, "tangent", fontName="Helvetica-Oblique",
                 fontSize=8, fillColor=C_PLAN, textAnchor="start"))
    # lag leg (along tangent)
    darrow(d, px, py, fpx, fpy, color=C_NEW, w=1.8)
    d.add(String((px + fpx) / 2 + 14, (py + fpy) / 2 - 20, "lag  (small W_LAG → drives speed)",
                 fontName="Helvetica-Bold", fontSize=8.5, fillColor=C_NEW, textAnchor="middle"))
    # contour leg (perpendicular)
    darrow(d, fpx, fpy, rx, ry, color=RED, w=1.8)
    d.add(String(fpx - 6, (fpy + ry) / 2, "contour", fontName="Helvetica-Bold", fontSize=8.5,
                 fillColor=RED, textAnchor="end"))
    d.add(String(fpx - 6, (fpy + ry) / 2 - 11, "(big W_CONTOUR)", fontName="Helvetica",
                 fontSize=7.5, fillColor=RED, textAnchor="end"))
    # markers
    d.add(Rect(rx - 4, ry - 4, 8, 8, fillColor=C_MPCC, strokeColor=C_MPCC))
    d.add(String(rx + 8, ry + 6, "reference", fontName="Helvetica-Bold", fontSize=8.5,
                 fillColor=C_MPCC, textAnchor="start"))
    d.add(String(rx + 8, ry - 5, "(receding target)", fontName="Helvetica", fontSize=7.5,
                 fillColor=C_MPCC, textAnchor="start"))
    d.add(Circle(px, py, 5.5, fillColor=C_DRONE, strokeColor=C_DRONE))
    d.add(String(px - 9, py - 3, "drone", fontName="Helvetica-Bold", fontSize=8.5,
                 fillColor=C_DRONE, textAnchor="end"))


def draw_stall(d):
    """v9 stall vs v9.1 governor, two mini panels."""
    import math
    from reportlab.graphics.shapes import Circle
    RED = colors.HexColor("#c1121f")

    def curve(x0):
        pts = []
        for i in range(40):
            t = i / 39.0
            x = x0 + t * 6.2 * cm
            yy = 52 + 18 * math.sin(t * 3.0)
            pts.append((x, yy))
        return pts

    # ---- titles (two lines each, well clear of the markers) ----
    d.add(String(0.2 * cm, 140, "v9 — the stall", fontName="Helvetica-Bold", fontSize=9.5,
                 fillColor=RED, textAnchor="start"))
    d.add(String(0.2 * cm, 127, "projection freezes → reference frozen → stable hover",
                 fontName="Helvetica", fontSize=8, fillColor=MUTED, textAnchor="start"))
    d.add(String(8.8 * cm, 140, "v9.1 — the governor", fontName="Helvetica-Bold", fontSize=9.5,
                 fillColor=C_NEW, textAnchor="start"))
    d.add(String(8.8 * cm, 127, "ref creeps forward → lag grows → drone pulled out",
                 fontName="Helvetica", fontSize=8, fillColor=MUTED, textAnchor="start"))
    # divider
    d.add(Line(8.35 * cm, 18, 8.35 * cm, 118, strokeColor=colors.HexColor("#cfd8e0"),
               strokeWidth=0.8, strokeDashArray=(3, 3)))

    # ---- panel A: v9 stall ----
    pa = curve(0.2 * cm)
    for i in range(len(pa) - 1):
        d.add(Line(pa[i][0], pa[i][1], pa[i + 1][0], pa[i + 1][1], strokeColor=C_GREY, strokeWidth=2))
    dr, rf = pa[20], pa[26]
    d.add(Circle(dr[0], dr[1] + 22, 5.5, fillColor=C_DRONE, strokeColor=C_DRONE))  # overshot, hovering
    d.add(String(dr[0], dr[1] + 33, "drone hovers", fontName="Helvetica", fontSize=7.5,
                 fillColor=C_DRONE, textAnchor="middle"))
    d.add(Rect(rf[0] - 4, rf[1] - 4, 8, 8, fillColor=C_MPCC, strokeColor=C_MPCC))
    d.add(String(rf[0] + 7, rf[1] - 2, "frozen ref", fontName="Helvetica", fontSize=7.5,
                 fillColor=C_MPCC, textAnchor="start"))
    d.add(String(dr[0], dr[1] - 14, "(no forward pull)", fontName="Helvetica-Oblique",
                 fontSize=7.5, fillColor=RED, textAnchor="middle"))

    # ---- panel B: v9.1 governor ----
    pb = curve(8.7 * cm)
    for i in range(len(pb) - 1):
        d.add(Line(pb[i][0], pb[i][1], pb[i + 1][0], pb[i + 1][1], strokeColor=C_PLAN, strokeWidth=2))
    drb, rfb = pb[20], pb[30]
    d.add(Circle(drb[0], drb[1] + 22, 5.5, fillColor=C_DRONE, strokeColor=C_DRONE))
    d.add(String(drb[0], drb[1] + 33, "drone", fontName="Helvetica", fontSize=7.5,
                 fillColor=C_DRONE, textAnchor="middle"))
    d.add(Rect(rfb[0] - 4, rfb[1] - 4, 8, 8, fillColor=C_NEW, strokeColor=C_NEW))
    d.add(String(rfb[0] + 7, rfb[1] - 2, "creeping ref", fontName="Helvetica", fontSize=7.5,
                 fillColor=C_NEW, textAnchor="start"))
    darrow(d, drb[0] + 6, drb[1] + 20, rfb[0] - 6, rfb[1] + 2, color=C_NEW, w=1.6)
    d.add(String((drb[0] + rfb[0]) / 2, drb[1] - 14, "growing lag pull", fontName="Helvetica-Oblique",
                 fontSize=7.5, fillColor=C_NEW, textAnchor="middle"))


def draw_speedprofile(d):
    """v9.2: curvature speed profile — track with corners, speed arrows slow in corners."""
    import math
    pts = []
    for i in range(80):
        t = i / 79.0
        x = 0.6 * cm + t * 14.4 * cm
        yy = 60 + 34 * math.sin(t * 6.3)  # several corners
        pts.append((x, yy))
    # curvature ~ |second derivative|; color/space arrows by local straightness
    for i in range(len(pts) - 1):
        d.add(Line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1],
                   strokeColor=C_PLAN, strokeWidth=2.2))
    # place speed markers: big arrow on straights (extrema of slope) small near peaks (corners)
    for i in range(6, len(pts) - 6, 8):
        x, yy = pts[i]
        # second difference as curvature proxy
        d2 = abs(pts[i - 6][1] - 2 * yy + pts[i + 6][1])
        fast = d2 < 6
        col = C_MPCC if fast else colors.HexColor("#c1121f")
        L = 26 if fast else 11
        tx = pts[i + 2][0] - pts[i - 2][0]
        tyv = pts[i + 2][1] - pts[i - 2][1]
        tn = math.hypot(tx, tyv)
        tx, tyv = tx / tn, tyv / tn
        darrow(d, x, yy, x + L * tx, yy + L * tyv, color=col, w=2.2)
    d.add(String(0.6 * cm, 110, "fast on straights (near V_CAP)", fontName="Helvetica-Bold",
                 fontSize=8.5, fillColor=C_MPCC, textAnchor="start"))
    d.add(String(W - 0.2 * cm, 110, "slow into corners (v = sqrt(a_lat / kappa))",
                 fontName="Helvetica-Bold", fontSize=8.5, fillColor=colors.HexColor("#c1121f"),
                 textAnchor="end"))


def draw_v10_compare(d):
    """v9.x 'chase an external reference' vs v10 'progress is a state inside the optimiser'."""
    bh = 1.15 * cm
    # top row: v9.x
    d.add(String(0.1 * cm, 145, "v9.x — speed is an INPUT (someone moves the reference)",
                 fontName="Helvetica-Bold", fontSize=9, fillColor=C_GREY, textAnchor="start"))
    dbox(d, 0.1 * cm, 110, 3.4 * cm, bh, "external projection\n+ governor", C_GREY, fs=8)
    dbox(d, 4.0 * cm, 110, 3.4 * cm, bh, "moves reference\nat V_REF / profile", C_GREY, fs=8)
    dbox(d, 7.9 * cm, 110, 3.6 * cm, bh, "MPCC chases it\n(point mass)", C_MPCC, fs=8)
    dbox(d, 12.0 * cm, 110, 3.8 * cm, bh, "attitude -> drone", C_ATT, fs=8)
    darrow(d, 3.5 * cm, 110 + bh / 2, 4.0 * cm, 110 + bh / 2)
    darrow(d, 7.4 * cm, 110 + bh / 2, 7.9 * cm, 110 + bh / 2)
    darrow(d, 11.5 * cm, 110 + bh / 2, 12.0 * cm, 110 + bh / 2)
    # bottom row: v10
    d.add(String(0.1 * cm, 90, "v10 — speed is an OUTPUT (the optimiser decides it)",
                 fontName="Helvetica-Bold", fontSize=9, fillColor=C_NEW, textAnchor="start"))
    dbox(d, 0.1 * cm, 18, 5.0 * cm, 2.4 * cm,
         "Time-optimal MPCC\nstates: pos, vel, THETA, v_theta\ncost: ... - MU*v_theta\n"
         "(progress is rewarded)", C_NEW, fs=8, max_chars=30)
    dbox(d, 5.6 * cm, 30, 4.5 * cm, 1.5 * cm,
         "curvature cap\nv_theta <= v_curv(theta)\nbrakes corners", C_MPCC, fs=8, max_chars=26)
    dbox(d, 11.5 * cm, 30, 4.3 * cm, 1.5 * cm, "attitude -> drone", C_ATT, fs=8)
    darrow(d, 5.1 * cm, 42, 5.6 * cm, 42)
    darrow(d, 10.1 * cm, 42, 11.5 * cm, 42)
    d.add(String(0.1 * cm, 6, "no external reference, no stall governor — progress is anchored to "
                 "the drone, so it cannot freeze", fontName="Helvetica-Oblique", fontSize=7.5,
                 fillColor=MUTED, textAnchor="start"))


def draw_lineage(d):
    """v9 -> v9.1 -> v9.2 -> v10 evolution with the delta on each arrow."""
    bw, bh = 3.0 * cm, 1.7 * cm
    y = 55
    xs = [0.1 * cm, 4.35 * cm, 8.6 * cm, 12.9 * cm]
    dbox(d, xs[0], y, bw, bh, "v9\nMPCC chasing a\nfixed-speed ref", C_GREY, fs=8.5)
    dbox(d, xs[1], y, bw, bh, "v9.1\n+ progress\ngovernor", C_PLAN, fs=8.5)
    dbox(d, xs[2], y, bw, bh, "v9.2\n+ curvature\nspeed profile", C_MPCC, fs=8.5)
    dbox(d, xs[3], y, bw, bh, "v10\ntime-optimal\n(acados RTI)", C_NEW, fs=8.5)
    deltas = ["fixes the\nmid-track stall", "slow corners,\nfast straights",
              "speed becomes\nan output"]
    for i in range(3):
        x1 = xs[i] + bw
        x2 = xs[i + 1]
        darrow(d, x1, y + bh / 2, x2, y + bh / 2, color=INK)
        for j, ln in enumerate(deltas[i].split("\n")):
            d.add(String((x1 + x2) / 2, y + bh + 6 - j * 9, ln, fontName="Helvetica",
                         fontSize=7, fillColor=MUTED, textAnchor="middle"))
    d.add(String(W / 2, 18, "Each version is a thin subclass of the previous one — only the marked "
                 "delta changes.", fontName="Helvetica-Oblique", fontSize=8, fillColor=MUTED,
                 textAnchor="middle"))


def P(txt, st=S_BODY):
    E.append(Paragraph(txt, st))


def H1(txt):
    E.append(Spacer(1, 2))
    E.append(Paragraph(txt, S_H1))
    E.append(HRFlowable(width="100%", thickness=1.1, color=ACCENT, spaceAfter=6))


def H2(txt):
    E.append(Paragraph(txt, S_H2))


def gap(h=6):
    E.append(Spacer(1, h))


def bullets(items, st=S_BULLET):
    li = [ListItem(Paragraph(t, st), leftIndent=6, value="•") for t in items]
    E.append(ListFlowable(li, bulletType="bullet", bulletColor=ACCENT, leftIndent=14,
                          bulletFontSize=8, spaceAfter=6))


def code(lines, caption=None):
    txt = "<br/>".join(
        l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
        for l in lines
    )
    t = Table([[Paragraph(txt, S_CODE)]], colWidths=[16.0 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODEBG),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d6d6cc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT2),
    ]))
    E.append(t)
    if caption:
        E.append(Paragraph(caption, S_CODECAP))
    else:
        gap(8)


def callout(title, body):
    inner = [[Paragraph(title, S_BOXH)], [Paragraph(body, S_BOX)]]
    t = Table(inner, colWidths=[16.0 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BOXBG),
        ("BOX", (0, 0), (-1, -1), 0.8, BOXBORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (0, 0), 8),
        ("TOPPADDING", (0, 1), (0, 1), 0),
        ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ("BOTTOMPADDING", (0, 1), (0, 1), 9),
        ("LINEBEFORE", (0, 0), (0, -1), 3, BOXBORDER),
    ]))
    E.append(t)
    gap(8)


def table(data, col_w, header=True, zebra=True):
    rows = []
    for r, row in enumerate(data):
        cells = []
        for c, val in enumerate(row):
            if r == 0 and header:
                cells.append(Paragraph(val, S_TH))
            elif c == 0:
                cells.append(Paragraph(val, S_TDB))
            else:
                cells.append(Paragraph(val, S_TD))
        rows.append(cells)
    t = Table(rows, colWidths=col_w, repeatRows=1 if header else 0)
    sty = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cfd8e0")),
    ]
    if header:
        sty += [("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("LINEBELOW", (0, 0), (-1, 0), 1.0, ACCENT)]
    if zebra:
        for r in range(1, len(data)):
            if r % 2 == 0:
                sty.append(("BACKGROUND", (0, r), (-1, r), LIGHT))
    t.setStyle(TableStyle(sty))
    E.append(t)
    gap(8)


# ============================================================================= COVER
E.append(Spacer(1, 3.2 * cm))
P("KaFa MPCC Controllers", S_TITLE)
gap(4)
P("How <b>v9</b>, <b>v9.1</b>, <b>v9.2</b> and <b>v10</b> Work", S_SUB)
gap(2)
P("An explainable walkthrough of a model-predictive contouring controller for autonomous "
  "drone racing — from a fixed-speed point-mass tracker to a real-time time-optimal planner.",
  style("ld", fontName="Helvetica", fontSize=11, leading=16, textColor=INK))
gap(20)
E.append(HRFlowable(width="100%", thickness=2, color=ACCENT2))
gap(10)
table(
    [["Version", "One-line idea"],
     ["v9", "Point-mass MPCC chasing a reference that recedes at a fixed speed."],
     ["v9.1", "v9 + a progress governor that kills the mid-track stall."],
     ["v9.2", "v9.1 + a curvature speed profile: slow into corners, fast on straights."],
     ["v10", "Time-optimal MPCC: speed becomes an output of the optimiser (real-time, acados)."]],
    col_w=[2.6 * cm, 13.4 * cm],
)
gap(6)
P("Project: <font name='Courier'>lsy_drone_racing</font> &nbsp;·&nbsp; track: level2 &nbsp;·&nbsp; "
  "control mode: attitude (roll, pitch, yaw, collective thrust)",
  style("meta", fontName="Helvetica", fontSize=9.5, leading=14, textColor=MUTED))
E.append(PageBreak())

# ============================================================================= 1. BIG PICTURE
H1("1 · The shared backbone (what every version reuses)")
P("All four controllers are the same machine with a different brain bolted into one slot. "
  "Before explaining what changes, it helps to see what stays fixed across v9 → v10.")

H2("A two-phase flight")
P("Each controller runs a tiny state machine:")
code([
    "TAKEOFF   ->  v8 vertical climb, tracked by a cascaded PID",
    "NAVIGATE  ->  v8 gate-aware path, tracked by the MPCC  (this is where the versions differ)",
], "The phase machine in KaFa_1500_v9.py. Takeoff is identical everywhere; only the NAVIGATE tracker changes.")

H2("The path comes from v8, not from the MPCC")
P("None of these versions invent <i>where</i> to fly. A planner inherited from v8 builds a "
  "smooth cubic spline through the centre of every remaining gate (with virtual \"gate-post "
  "funnels\" so the curve threads each opening instead of clipping a frame bar, and obstacle "
  "push-off). The spline is rebuilt whenever a gate is passed or an observed gate/obstacle pose "
  "shifts. The MPCC's only job is to <b>fly that fixed geometry as fast as it safely can</b>. "
  "This separation is the central design choice: <i>geometry is planned offline-style; speed is "
  "decided online by the controller.</i>")

H2("What is MPCC, in one paragraph?")
P("<b>Model-Predictive Contouring Control</b>. \"Model-predictive\" means: at every control step "
  "(50 Hz) the controller simulates a short horizon of the future (~0.7–0.9 s) using a simple "
  "physics model, picks the sequence of accelerations that minimises a cost over that horizon, "
  "applies only the <i>first</i> one, then re-plans from scratch next step. \"Contouring\" means "
  "the cost is written in path coordinates: a <b>contouring error</b> (how far the drone is "
  "<i>sideways</i> off the path) and a <b>lag error</b> (how far it is <i>behind</i> the point it "
  "should be chasing). Penalise sideways error hard, penalise lag gently, and the drone hugs the "
  "path while being pulled forward along it.")

diagram(160, draw_geometry,
        "Figure 1 — The two MPCC errors, in path coordinates. The error from the drone to its "
        "receding reference is split into a <b>lag</b> part (along the path, lightly penalised — "
        "this is what drives forward speed) and a <b>contour</b> part (sideways, heavily "
        "penalised — this keeps the drone on the line).")

callout("The model is a point mass",
        "The MPCC does not model the drone's full rigid-body dynamics. It treats the drone as a "
        "point with mass, whose control input is a 3-D acceleration. The optimiser plans "
        "world-frame accelerations; a separate step converts the first acceleration into an "
        "<font name='Courier'>[roll, pitch, yaw, thrust]</font> attitude command via "
        "<font name='Courier'>_vector_to_attitude</font>. The thrust vector the drone must "
        "produce is simply <b>m·(a + g)</b> — the planned acceleration plus gravity, scaled by "
        "mass. Tilt and thrust limits in the optimiser are what keep this point-mass plan "
        "trackable by the real attitude loop.")

H2("Speed is squeezed out of the actuator limits")
P("The recurring philosophy across the whole family: <b>don't hand-tune a cruise speed.</b> "
  "Instead, give the optimiser the drone's real limits — a maximum thrust magnitude, a maximum "
  "tilt angle, a minimum vertical thrust (so it never free-falls), a velocity cap — and let the "
  "fastest feasible motion fall out. That is what lets one controller generalise to any track "
  "geometry the planner hands it. The versions differ mainly in <i>how the forward pull is "
  "generated</i> and <i>how corners are slowed down</i>.")

diagram(118, draw_pipeline,
        "Figure 2 — The shared per-step control loop (runs at 50 Hz). The v8 planner decides "
        "<b>where</b> to fly (the path); the MPCC decides <b>how fast</b> (accelerations); the "
        "result is converted to an attitude command. Everything re-plans from the latest "
        "observation every step. The four versions only change the MPCC box.")

E.append(PageBreak())

# ============================================================================= 2. v9
H1("2 · v9 — the baseline MPCC")
P("v9 replaced v8's hand-tuned speed caps and cascaded-PID path tracker with the MPCC. It keeps "
  "v8's planner and vertical takeoff and changes only the NAVIGATE tracker.")

H2("How the reference moves")
P("v9 builds a <b>receding reference</b>. Each step it finds the closest point on the spline "
  "ahead of the drone (a forward-windowed nearest-point projection), then samples a string of "
  "look-ahead targets spaced a fixed arc-distance apart: spacing = <b>V_REF · step_dt</b>. So the "
  "reference marches down the path at a constant speed V_REF and the drone chases it. Faster "
  "V_REF ⇒ the reference pulls away harder ⇒ the drone flies faster — until the actuator limits "
  "or gate precision break.")

H2("The optimisation, in plain terms")
P("Over a horizon of <b>N = 14</b> steps of 0.05 s (~0.7 s look-ahead), the optimiser chooses "
  "accelerations to minimise:")
code([
    "cost =  W_CONTOUR * (sideways distance from path)^2      # stay on the line  (=14.0, big)",
    "      + W_LAG      * (distance behind the reference)^2    # chase forward     (= 1.0, small)",
    "      + W_ACCEL    * (acceleration)^2                     # be smooth         (= 0.02)",
    "",
    "subject to, at every step:",
    "   |thrust|^2  <=  a_max^2            # can't pull more than max thrust",
    "   thrust_z    >=  A_Z_MIN  (=7.0)    # always keep lift, never free-fall",
    "   |thrust_xy| <=  TILT_RATIO*thrust_z   # tilt <= ~29 deg  (TILT_RATIO=0.55)",
    "   |velocity|  <=  V_MAX               # hard speed cap",
    "   point-mass kinematics linking pos, vel, acc",
], "The v9 cost and constraints (KaFa_v9/mpcc.py). Contouring is weighted ~14x the lag term, so "
   "the drone prioritises staying on the line over catching up.")
P("This is a small nonlinear program. v9 builds it once with CasADi and re-solves it every "
  "control step with IPOPT, warm-started from last step's plan. No special build environment is "
  "needed — that simplicity is v9's selling point.")

H2("The takeoff ramp")
P("Right after the takeoff hand-off the drone sits slightly off the path. If the reference "
  "immediately recedes at full V_REF, the MPCC lunges at the first gate and can sag toward the "
  "floor. So V_REF is ramped in from a fraction (RAMP_START) up to full over RAMP_S seconds, "
  "letting the drone settle onto the line first.")

H2("Documented results (level2)")
table(
    [["Setting", "Avg lap (successful)", "Finish rate", "Note"],
     ["V_REF = 1.8", "9.55 s", "91 %", "the safe, reliable tune"],
     ["V_REF = 2.0", "8.66 s", "63 %", "faster but overshoots gates"]],
    col_w=[3.6 * cm, 4.2 * cm, 2.6 * cm, 5.6 * cm],
)
P("This is the central tension of the whole family in one table: pushing the recede rate up buys "
  "lap time but loses gates. v9.1 and v9.2 are different attempts to break that trade-off.")

callout("v9's fatal flaw: the mid-track stall",
        "v9's reference is driven <i>purely</i> by the drone's own nearest-point projection. At a "
        "sharp turn or a loop the drone can overshoot the path; the projection then stops "
        "advancing; the receding reference freezes about one horizon ahead; and the MPCC settles "
        "into a perfectly stable hover it never escapes (replans only happen on a gate pass). On "
        "twisty tracks this killed runs outright. Fixing it is the entire point of v9.1.")

diagram(150, draw_stall,
        "Figure 3 — The stall and its fix. <b>Left (v9):</b> the drone overshoots a sharp turn, "
        "its nearest-point projection stops advancing, the reference freezes a horizon ahead, and "
        "it hovers forever. <b>Right (v9.1):</b> the governor forces the reference to creep "
        "forward regardless, growing the lag error until the drone is pulled back onto the path.")

E.append(PageBreak())

# ============================================================================= 3. v9.1
H1("3 · v9.1 — the stall fix and a robustness pass")
P("v9.1 is a thin subclass of v9. It keeps the planner, takeoff and the MPCC structure, and "
  "changes three things: a <b>progress governor</b> (the headline fix), two <b>solver "
  "robustness fixes</b>, and a <b>retune</b> of the speed levers.")

H2("The progress governor — guaranteeing forward motion")
P("The stall happened because reference progress could freeze. The governor makes progress "
  "<i>always</i> creep forward, but in a bounded way so a genuinely blocked drone doesn't get a "
  "runaway reference:")
code([
    "target = max(prev_progress, projection)        # never go backward; catch the projection",
    "target = max(target, prev_progress + creep)    # but ALWAYS creep forward a little",
    "target = min(target, projection + max_lead)    # yet stay bounded ahead of a stuck drone",
], "The three lines of _govern() (KaFa_1500_v9_1.py). creep = MIN_PROGRESS_RATE·dt (0.5 curve-s "
   "per second); max_lead = MAX_LEAD_T (0.30 s).")
P("In normal flight the projection advances faster than the creep, so the middle line never "
  "binds and behaviour is identical to v9. The governor only \"wakes up\" when the projection "
  "freezes: then progress creeps ahead until the reference leads the drone by max_lead, which "
  "grows the lag error and <b>pulls the drone forward out of the stall</b>. The lead cap means a "
  "drone pinned against a pole gets a steady, bounded tug rather than a reference rocketing away.")
P("A <b>stall watchdog</b> sits on top: if the drone stays slower than V_STALL (0.25 m/s) for "
  "T_STALL (0.2 s), it scales both the creep and the lead by STALL_BOOST (2x) so the pull is "
  "strong enough to break a hard stall.")

H2("Two solver robustness fixes")
bullets([
    "<b>Don't constrain the initial velocity.</b> The velocity cap V_MAX is now applied only to "
    "the <i>future</i> velocities the solver decides, not to <font name='Courier'>vel[0]</font>, "
    "which is a measurement. In v9, entering a turn already above V_MAX made the whole problem "
    "infeasible, so the solver threw and fell back to a stale plan exactly when it mattered most.",
    "<b>Better failure fallback.</b> On a solve failure v9.1 shifts the previous plan one step "
    "and <i>holds</i> the last command, instead of wrapping the already-used first command back "
    "to the end of the horizon.",
])

H2("The retune — where the extra speed actually comes from")
P("A parameter sweep produced a key, slightly counter-intuitive finding: <b>raising V_REF does "
  "not help</b> — it just makes the point-mass model overshoot the gates. Speed is bought "
  "elsewhere:")
table(
    [["Knob", "v9", "v9.1", "Why"],
     ["HORIZON", "14 (~0.7 s)", "18 (~0.9 s)", "more anticipation to carry speed through turns"],
     ["W_LAG", "1.0", "1.5", "the real speed lever: close the lag harder, up toward V_MAX"],
     ["TILT_RATIO", "0.55 (~29°)", "0.85 (~40°)", "corner authority — not cruise speed"],
     ["V_REF", "1.8", "1.8", "deliberately unchanged"]],
    col_w=[3.0 * cm, 3.0 * cm, 3.3 * cm, 6.7 * cm],
)
callout("Subtle but important: tilt is a corner knob, not a speed knob",
        "Sweeps showed lap time is essentially flat across tilt 0.75–1.05 and V_MAX 2.4–3.6 at a "
        "fixed recede rate — at cruise the drone never saturates tilt or V_MAX (it uses ~6 m/s² "
        "of ~18 m/s² available). What the extra tilt (0.85 ≈ 40°) buys is the <i>corners</i>: it "
        "removes slow near-stall grinds through the tight gate-2/3 turns, halving run-to-run lap "
        "spread, at no finish-rate cost. Above ~1.0 the point-mass plan commands tilt the real "
        "attitude loop can't track, and gates start being overshot.")
P("Net effect: ~6 % faster on level2 than v9 with no finish-rate cost, and noticeably more "
  "robust on twisty tracks (the stall is gone).")

E.append(PageBreak())

# ============================================================================= 4. v9.2
H1("4 · v9.2 — a curvature-aware speed profile")
P("v9.2 is a thin subclass of v9.1 that changes exactly one thing: <b>how fast the reference "
  "recedes</b>. v9/v9.1 use a single constant rate (V_REF), which forces a compromise — a rate "
  "safe through the tight gate turns is needlessly slow on the straights. v9.2 makes the recede "
  "rate <b>depend on the local path curvature</b>.")

H2("The idea: a friction circle")
P("A drone (like a car) can only turn so hard before it slides off its line — the limit is set "
  "by how much <i>lateral</i> acceleration it can produce, A_LAT_MAX. On a curve of curvature κ, "
  "lateral acceleration is v²·κ, so the fastest safe speed through that curve is:")
code([
    "v_curv(s)  =  sqrt( A_LAT_MAX / kappa(s) )    clamped to [V_MIN, V_CAP]",
    "",
    "tight corner (large kappa)  ->  low  v_curv   (self-braking)",
    "straight     (kappa ~ 0)    ->  high v_curv  ->  clamped at V_CAP",
], "The curvature speed cap (KaFa_v9_1/speed_profile.py). Curvature comes from the spline's "
   "analytic 1st/2nd derivatives — nothing numerical is hand-rolled.")
P("A raw curvature cap would demand impossible instant speed changes, so two passes smooth it "
  "into something a real drone can follow:")
bullets([
    "<b>Backward pass</b> — walk the path from the end backward, forcing the speed to start "
    "braking <i>early enough</i> to hit each corner's cap in time (bounded by longitudinal "
    "acceleration).",
    "<b>Forward pass</b> — walk forward, forcing acceleration out of corners to respect the same "
    "longitudinal limit.",
])
P("This is a textbook <b>time-optimal path parameterisation</b>: the well-known way to find the "
  "fastest speed profile along a fixed path under an acceleration budget. The result is a "
  "look-ahead that is densely spaced (slow) into corners and widely spaced (fast) on straights.")

diagram(122, draw_speedprofile,
        "Figure 4 — The curvature-aware speed profile. Arrow length is the local reference speed: "
        "long on the straights (clamped at V_CAP) and short through the corners, where the "
        "friction-circle cap v = √(A_LAT_MAX / κ) forces self-braking. v9/v9.1 would fly the "
        "whole path at one blunt safe rate instead.")

H2("How it plugs in")
P("v9.2 keeps v9.1's MPCC untouched — its V_MAX / tilt / thrust limits remain the safety net. "
  "It only swaps the look-ahead spacing from \"constant arc-step\" to \"arc-steps spaced by the "
  "profiled speed.\" Two supporting tweaks:")
bullets([
    "<b>Raise the MPCC's hard cap</b> to V_MAX = 3.0 (above the profile's V_CAP = 2.4) so the "
    "tracker has headroom to actually reach the faster reference instead of clipping it.",
    "<b>Gentler takeoff ramp</b> (RAMP_START 0.12, RAMP_S 1.4 s vs v9.1's 0.2 / 1.1 s): the "
    "faster straight speed re-exposes the gate-0 hand-off transient, so the look-ahead is eased "
    "in more slowly. This single change took finish rate from ~75 % back to 100 % on hard seeds.",
])

H2("Why these specific numbers")
table(
    [["Knob", "Value", "Reasoning"],
     ["V_CAP", "2.4 m/s", "straight-line recede cap; main speed lever. 2.8 is ~3% faster but "
                          "the gate-0 transient starts to bite (finish ~92%)."],
     ["A_LAT_MAX", "8.0 m/s²", "≈ tilt_ratio·g at 40° tilt — the physical cornering ceiling. "
                              "12+ corners too fast and overshoots gate-2/3 (finish 40–50%)."],
     ["V_MIN", "1.3 m/s", "floor so the reference can't crawl to a stop at a spline cusp."]],
    col_w=[2.8 * cm, 2.4 * cm, 10.8 * cm],
)
P("Result: 100 % finish on level2 at a lower lap time than v9.1 — the straights run near V_CAP "
  "while the sharp gate-2/3 reversal self-brakes, instead of everything flying at one blunt safe "
  "rate.")

E.append(PageBreak())

# ============================================================================= 5. v10
H1("5 · v10 — a time-optimal MPCC")
P("v9.x always poses the problem as <i>\"chase a reference that someone else moves at a chosen "
  "speed.\"</i> v10 changes the question entirely: it lets the <b>optimiser itself decide how "
  "fast to traverse the path</b>, by making progress a state and paying the optimiser to make "
  "progress. This is a genuinely different and more powerful formulation.")

H2("Progress becomes a state")
P("v10 adds two new state variables to the point mass:")
bullets([
    "<b>θ</b> (\"theta\") — how far along the path the drone has progressed, in arc length.",
    "<b>v<sub>θ</sub></b> — the <i>rate</i> of progress (how fast it is advancing along the path).",
])
P("The drone now has 8 states: <font name='Courier'>[pos(3), vel(3), θ, v_θ]</font>, and the "
  "controls are <font name='Courier'>[acceleration(3), dv_θ]</font> — it even chooses how to "
  "change its own progress rate. The cost adds one decisive term:")
code([
    "cost =  W_CONTOUR*(sideways err)^2 + W_LAG*(lag)^2     # stay on the path (as before)",
    "      - MU * v_theta                                   # *** reward making progress ***",
    "      + W_ACCEL*|acc|^2 + R_DV*(dv_theta)^2            # smoothness",
], "The v10 cost (KaFa_v10/mpcc.py). The minus sign on MU·v_theta means the optimiser is PAID to "
   "go faster — so traversal speed is an OUTPUT of the optimisation, not an input.")
P("Because faster progress lowers the cost, the optimiser pushes v<sub>θ</sub> up until "
  "<i>something stops it</i> — and the thing that stops it is the same friction-circle curvature "
  "limit from v9.2, now used as a per-step bound:")
code([
    "v_theta  <=  v_curv(theta)        # progress rate capped by what the corner can hold",
    "|vel|    <=  v_curv(theta)        # actual speed capped too (a SOFT constraint)",
], "The curvature cap (a soft/slack constraint so a momentary overspeed doesn't make the problem "
   "infeasible). The optimiser rides the cap: full speed on straights, auto-braked into corners.")

diagram(165, draw_v10_compare,
        "Figure 5 — The conceptual leap. In v9.x (top) something outside the solver moves a "
        "reference and the MPCC chases it. In v10 (bottom) progress θ and its rate v_θ are states "
        "inside the optimiser, and rewarding v_θ makes speed an output — so the external "
        "projection and the stall governor simply disappear.")

callout("Why this is cleaner than v9.x",
        "Because progress is now a <i>state anchored to the drone</i>, it physically cannot "
        "freeze ahead of the drone. That means v10 deletes v9.1's external nearest-point "
        "projection-driven reference <b>and its entire 5-knob stall governor</b> — the stall is "
        "structurally impossible rather than patched. The corner-braking that v9.2 bolted on as "
        "a separate reference-shaping pass is now intrinsic: the optimiser brakes corners because "
        "the curvature cap is part of the same problem it is already solving.")

H2("The hard part: making it real-time")
P("This richer problem is far more expensive to solve. A first version using IPOPT that embedded "
  "the spline reference inside the solver cost <b>~1.3 seconds per solve</b> — completely "
  "unusable at 50 Hz (the 20 ms budget). v10 ships only because of two engineering moves:")
bullets([
    "<b>acados SQP-RTI.</b> Instead of solving the nonlinear program to convergence each step, "
    "it does <i>one</i> real-time iteration — a single quadratic-program (QP) solve per control "
    "step, warm-started from the last. The C solver is code-generated and compiled once per "
    "process and cached.",
    "<b>Linearise the reference out of the solver (Liniger's MPCC).</b> The spline is not "
    "evaluated inside the solver. Each step, the path point, unit tangent and curvature speed at "
    "the <i>predicted</i> progress of every horizon stage are computed in NumPy "
    "(<font name='Courier'>ArcPath</font>) and passed in as per-stage parameters. The reference "
    "is approximated as a straight line p_ref(θ) ≈ point + tangent·(θ − θ̄) around that point.",
])
P("Together these bring the solve down to a <b>few milliseconds</b> — real-time. The trade-off: "
  "v10 requires the acados environment (run under <font name='Courier'>pixi run</font>), whereas "
  "v9.x runs anywhere CasADi/IPOPT does.")

callout("Compute time is not score time",
        "The competition scores <i>flight</i> time and finish rate, not solver speed. So the "
        "1.3 s IPOPT version and the 1.3 ms acados version would score identically <b>if</b> the "
        "slow one could run — but it can't be flown or rendered live. acados isn't about a better "
        "lap; it's about the controller being runnable at all in real time.")

H2("Documented results (level2)")
table(
    [["Setting", "Lap", "Finish", "Note"],
     ["V_MAX = 3.0", "~8.7 s", "~94 %", "reliability-first default"],
     ["V_MAX = 3.5+", "few % faster", "~80 %", "re-exposes the gate-0 takeoff transient"]],
    col_w=[3.4 * cm, 3.2 * cm, 2.4 * cm, 7.0 * cm],
)
P("v10 needs <i>more</i> takeoff easing than v9.2 (RAMP_START 0.08, RAMP_S 2.0 s) because it "
  "carries more speed into gate 0. With that ramp it reaches 100 % finish at full speed in the "
  "hard-seed tests.")

E.append(PageBreak())

# ============================================================================= 6. COMPARISON
H1("6 · Side-by-side and the real speed ceiling")

H2("The lineage at a glance")
diagram(95, draw_lineage,
        "Figure 6 — The evolution. Each version inherits the last and changes exactly one thing.")
table(
    [["", "v9", "v9.1", "v9.2", "v10"],
     ["Forward drive", "constant V_REF reference", "constant V_REF + governor",
      "curvature speed profile", "−μ·v_θ (progress reward)"],
     ["Speed is...", "an input (V_REF)", "an input (V_REF)", "an input (shaped profile)",
      "an OUTPUT of the optimiser"],
     ["Corner braking", "none (one safe rate)", "none (one safe rate)",
      "friction-circle profile", "friction-circle cap, in-solver"],
     ["Stall handling", "stalls (bug)", "5-knob governor", "inherits governor",
      "impossible by design"],
     ["Solver", "CasADi + IPOPT", "CasADi + IPOPT", "CasADi + IPOPT", "acados SQP-RTI (C)"],
     ["Needs special env", "no", "no", "no", "yes (acados / pixi)"],
     ["level2 result", "9.55 s / 91%", "~6% faster, robust", "100% finish, faster still",
      "~8.7 s / ~94%"]],
    col_w=[3.0 * cm, 3.25 * cm, 3.25 * cm, 3.25 * cm, 3.25 * cm],
)

H2("What actually limits the lap time")
P("A recurring, slightly surprising lesson runs through every version's tuning notes: on this "
  "short track the binding constraint is <b>gate-passing precision under gate randomisation "
  "(±0.15 m)</b>, not actuator authority. The drone almost never runs out of thrust or tilt at "
  "cruise. So:")
bullets([
    "The genuine speed knob is the <b>reference recede rate</b> (V_REF in v9/v9.1, the V_CAP "
    "profile in v9.2, the progress reward against the curvature cap in v10) — <i>not</i> tilt "
    "or V_MAX.",
    "Pushing that rate up overshoots gates: constant V_REF 1.8→~90% finish, 2.0→~67%, 2.4→~40%.",
    "The two stubborn failure modes everywhere are (a) the <b>takeoff → gate-0 hand-off "
    "transient</b> (cured by the ramp, which each faster version needs more of) and (b) the "
    "<b>gate-2/3 reversal overshoot</b> (cured by corner braking — the whole point of v9.2/v10).",
    "Because every version hits the same gate-precision ceiling, v10's reliable lap (~8.7 s) is "
    "<b>competitive with, but not dramatically faster than</b>, v9.2. v10's real win is "
    "structural cleanliness (no governor, corner-braking built in, progress can't freeze), not a "
    "huge lap-time jump on this particular track.",
])

callout("The one-sentence summary",
        "v9 makes a point mass chase a fixed-speed reference; v9.1 stops that reference from "
        "freezing; v9.2 makes the reference slow into corners and sprint on straights; and v10 "
        "throws out the \"chase a reference\" framing entirely and lets a real-time optimiser "
        "decide the speed itself — the cleanest formulation, bounded by the same gate-precision "
        "reality that caps all of them.")

gap(10)
E.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cfd8e0")))
gap(4)
P("Generated from the source in <font name='Courier'>lsy_drone_racing/control/</font> "
  "(KaFa_v9, KaFa_v9_1, KaFa_v9_2, KaFa_v10) and their cockpit/settings files.", S_FOOT)


# ============================================================================= build
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"KaFa MPCC Controllers — v9 to v10   ·   page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=2.4 * cm, rightMargin=2.4 * cm, topMargin=2.0 * cm, bottomMargin=2.0 * cm,
    title="KaFa MPCC Controllers: v9 to v10 explained", author="generated",
)
doc.build(E, onFirstPage=lambda c, d: None, onLaterPages=_footer)
print("wrote", OUT)
