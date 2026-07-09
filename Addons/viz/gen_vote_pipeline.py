#!/usr/bin/env python
"""Emit vote_pipeline.svg (house style: Arial, #9aa0a6 arrows, rounded rects, embedded
thumbnails) AND a matching vote_pipeline.png from one shared layout spec."""
import base64, os
from PIL import Image, ImageDraw, ImageFont

FIG = r"c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures"
W, H = 1740, 760
ARROW = "#9aa0a6"
INK = "#202124"
BLUE = "#4a72b0"
GREY = "#8a8f96"

# ---- layout: name -> dict(x,y,w,h, kind, text, thumb, dashed, fill, edge, tcol) ----
TH = 210, 158
N = {}
def box(name, x, y, w, h, text="", kind="proc", thumb=None, dashed=False,
        fill="#f5f6f7", edge="#5f6368", tcol=INK, fs=15):
    N[name] = dict(x=x, y=y, w=w, h=h, text=text, kind=kind, thumb=thumb,
                   dashed=dashed, fill=fill, edge=edge, tcol=tcol, fs=fs)

# inputs + extractors (thumbnails)
box("frame", 40, 300, *TH, kind="thumb", thumb="pipe_frame.png", text="frame pair (t, t−δ)")
box("dist",  300, 66, *TH, kind="thumb", thumb="pipe_districts.png", text="DINOv2 districts")
box("flow",  300, 300, *TH, kind="thumb", thumb="pipe_flow.png", text="RAFT dense flow")
box("depth", 300, 534, *TH, kind="thumb", thumb="pipe_depth.png", text="up-to-scale depth")
box("votes", 610, 300, *TH, kind="thumb", thumb="pipe_votes.png", text="per-district votes (f_k, Z_k, p_k)")
# branch A (ablated, dashed grey)
box("fit",   905, 150, 250, 78, kind="proc", dashed=True, fill="#fbfbfb", edge=GREY, tcol="#6b7075",
    text="egomotion fit\n→ robust trust")
box("trust", 1200, 150, 250, 78, kind="proc", dashed=True, fill="#fbfbfb", edge=GREY, tcol="#6b7075",
    text="per-ray motion trust\n(alternative consumer, ablated)", fs=13)
# branch B (locked)
box("stats", 905, 470, 250, 78, kind="proc", text="vote statistics\nq10,  dis3")
box("dec", 1235, 462, 150, 96, kind="diamond", text="still?", fill="#eef2f8", edge=BLUE)
# outputs
box("freeze", 1470, 300, 250, 62, kind="out", fill="#eaf0f9", edge=BLUE,
    text="freeze: copy previous\npose, skip solve", fs=14)
box("repin", 1470, 386, 250, 62, kind="out", fill="#eaf0f9", edge=BLUE,
    text="re-pin frozen keyframes\nafter every BA cycle", fs=14)
box("track", 1470, 556, 250, 56, kind="out", fill="#f0f1f3", edge="#5f6368",
    text="track: normal pose solve", fs=14)

def cx(n): return N[n]["x"] + N[n]["w"] / 2
def cy(n): return N[n]["y"] + N[n]["h"] / 2
def R(n):  return N[n]["x"] + N[n]["w"], cy(n)      # right-center
def L(n):  return N[n]["x"], cy(n)                   # left-center
def Tp(n): return cx(n), N[n]["y"]
def Bp(n): return cx(n), N[n]["y"] + N[n]["h"]

# arrows: (src_pt, dst_pt, label, labelpos)
A = [
    (R("frame"), L("dist"), "", None), (R("frame"), L("flow"), "", None),
    (R("frame"), L("depth"), "", None),
    (R("dist"), L("votes"), "", None), (R("flow"), L("votes"), "", None),
    (R("depth"), L("votes"), "", None),
    (R("votes"), L("fit"), "", None), (R("votes"), L("stats"), "", None),
    (R("fit"), L("trust"), "", None), (R("stats"), L("dec"), "", None),
    ((cx("dec"), N["dec"]["y"]+8), L("freeze"), "yes", "u"),
    ((cx("dec"), N["dec"]["y"]+N["dec"]["h"]-8), L("track"), "no", "d"),
    (Bp("freeze"), Tp("repin"), "", None),
]

# locked-config grouping rect
lk = [N[k] for k in ("stats", "dec", "freeze", "repin", "track")]
LKx0 = min(b["x"] for b in lk) - 16; LKy0 = min(b["y"] for b in lk) - 16
LKx1 = max(b["x"]+b["w"] for b in lk) + 16; LKy1 = max(b["y"]+b["h"] for b in lk) + 16

# ============================ SVG ============================
def b64(path):
    with open(os.path.join(FIG, path), "rb") as f:
        return base64.b64encode(f.read()).decode()

def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def svg_text(x, y, lines, fs, col, anchor="middle", weight="400"):
    dy0 = -(len(lines)-1)*fs*0.6
    out = f'<text x="{x:.0f}" y="{y+dy0:.0f}" font-size="{fs}" fill="{col}" text-anchor="{anchor}" font-weight="{weight}">'
    for i, ln in enumerate(lines):
        out += f'<tspan x="{x:.0f}" dy="{0 if i==0 else fs*1.15:.1f}">{esc(ln)}</tspan>'
    return out + "</text>"

s = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
     f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Arial, Helvetica, sans-serif">',
     '<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7.2" refY="3" orient="auto" '
     f'markerUnits="strokeWidth"><path d="M0,0 L7,3 L0,6 Z" fill="{ARROW}"/></marker></defs>',
     f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
# locked box
s.append(f'<rect x="{LKx0}" y="{LKy0}" width="{LKx1-LKx0}" height="{LKy1-LKy0}" rx="14" '
         f'fill="none" stroke="{BLUE}" stroke-width="1.5" stroke-dasharray="7 5" opacity="0.8"/>')
s.append(svg_text((LKx0+LKx1)/2, LKy1+22, ["locked configuration"], 15, BLUE, weight="700"))
# arrows
for (x1, y1), (x2, y2), lab, lp in A:
    s.append(f'<path d="M{x1:.0f},{y1:.0f} L{x2-6:.0f},{y2:.0f}" fill="none" stroke="{ARROW}" '
             f'stroke-width="2" marker-end="url(#ar)"/>')
    if lab:
        mx, my = (x1+x2)/2, (y1+y2)/2 + (-8 if lp == "u" else 16)
        s.append(svg_text(mx, my, [lab], 13, "#5f6368"))
# nodes
for n, b in N.items():
    if b["kind"] == "thumb":
        s.append(f'<image x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" height="{b["h"]}" '
                 f'xlink:href="data:image/png;base64,{b64(b["thumb"])}"/>')
        s.append(f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" height="{b["h"]}" rx="6" '
                 f'fill="none" stroke="#c8ccd0" stroke-width="1"/>')
        s.append(svg_text(cx(n), b["y"]+b["h"]+18, b["text"].split("\n"), 14, INK))
    elif b["kind"] == "diamond":
        px = f'{cx(n):.0f},{b["y"]:.0f} {b["x"]+b["w"]:.0f},{cy(n):.0f} {cx(n):.0f},{b["y"]+b["h"]:.0f} {b["x"]:.0f},{cy(n):.0f}'
        s.append(f'<polygon points="{px}" fill="{b["fill"]}" stroke="{b["edge"]}" stroke-width="1.5"/>')
        s.append(svg_text(cx(n), cy(n)-b["fs"]*0.3, b["text"].split("\n"), b["fs"], b["tcol"], weight="600"))
    else:
        dash = ' stroke-dasharray="6 4"' if b["dashed"] else ''
        s.append(f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" height="{b["h"]}" rx="10" '
                 f'fill="{b["fill"]}" stroke="{b["edge"]}" stroke-width="1.5"{dash}/>')
        s.append(svg_text(cx(n), cy(n)-b["fs"]*0.25, b["text"].split("\n"), b["fs"], b["tcol"], weight="500"))
s.append('</svg>')
open(os.path.join(FIG, "vote_pipeline.svg"), "w", encoding="utf-8").write("\n".join(s))
print("wrote vote_pipeline.svg", os.path.getsize(os.path.join(FIG, "vote_pipeline.svg")), "bytes")

# ============================ PNG (PIL, 2x) ============================
S = 2
img = Image.new("RGB", (W*S, H*S), "white")
d = ImageDraw.Draw(img)
FONTP = r"C:\Windows\Fonts\arial.ttf"; FONTB = r"C:\Windows\Fonts\arialbd.ttf"
def font(sz, bold=False): return ImageFont.truetype(FONTB if bold else FONTP, sz*S)
def hexrgb(h): h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))
def dtext(x, y, lines, fs, col, bold=False, anchor="mm"):
    f = font(fs, bold); col = hexrgb(col)
    tot = len(lines)*fs*1.15*S
    yy = y*S - tot/2 + fs*0.6*S
    for ln in lines:
        d.text((x*S, yy), ln, font=f, fill=col, anchor="mm"); yy += fs*1.15*S
def dashed_rect(x0, y0, x1, y1, col, wdt, rad=0):
    col = hexrgb(col); step = 11*S
    pts = []
    for xx in range(int(x0*S), int(x1*S), step): pts.append(((xx, y0*S), (min(xx+6*S, x1*S), y0*S)))
    for xx in range(int(x0*S), int(x1*S), step): pts.append(((xx, y1*S), (min(xx+6*S, x1*S), y1*S)))
    for yy in range(int(y0*S), int(y1*S), step): pts.append(((x0*S, yy), (x0*S, min(yy+6*S, y1*S))))
    for yy in range(int(y0*S), int(y1*S), step): pts.append(((x1*S, yy), (x1*S, min(yy+6*S, y1*S))))
    for a, b in pts: d.line([a, b], fill=col, width=wdt*S)
def arrow(x1, y1, x2, y2, lab=None, lp=None):
    import math
    col = hexrgb(ARROW)
    ang = math.atan2(y2-y1, x2-x1); x2e = x2-6
    d.line([(x1*S, y1*S), (x2e*S, y2*S)], fill=col, width=2*S)
    hl = 9*S
    for da in (2.6, -2.6):
        d.line([(x2e*S, y2*S), (x2e*S-hl*math.cos(ang+da), y2*S-hl*math.sin(ang+da))], fill=col, width=2*S)
    d.polygon([(x2e*S, y2*S), (x2e*S-hl*math.cos(ang+0.4), y2*S-hl*math.sin(ang+0.4)),
               (x2e*S-hl*math.cos(ang-0.4), y2*S-hl*math.sin(ang-0.4))], fill=col)
    if lab:
        mx, my = (x1+x2)/2, (y1+y2)/2 + (-10 if lp=="u" else 14)
        dtext(mx, my, [lab], 13, "#5f6368")
# locked box
dashed_rect(LKx0, LKy0, LKx1, LKy1, BLUE, 2)
dtext((LKx0+LKx1)/2, LKy1+16, ["locked configuration"], 15, BLUE, bold=True)
# arrows
for (x1, y1), (x2, y2), lab, lp in A: arrow(x1, y1, x2, y2, lab, lp)
# nodes
for n, b in N.items():
    x0, y0, w, h = b["x"], b["y"], b["w"], b["h"]
    if b["kind"] == "thumb":
        th = Image.open(os.path.join(FIG, b["thumb"])).resize((w*S, h*S))
        img.paste(th, (x0*S, y0*S))
        d.rounded_rectangle([x0*S, y0*S, (x0+w)*S, (y0+h)*S], radius=6*S, outline=hexrgb("#c8ccd0"), width=S)
        dtext(cx(n), y0+h+15, b["text"].split("\n"), 14, INK)
    elif b["kind"] == "diamond":
        pts = [(cx(n)*S, y0*S), ((x0+w)*S, cy(n)*S), (cx(n)*S, (y0+h)*S), (x0*S, cy(n)*S)]
        d.polygon(pts, fill=hexrgb(b["fill"]), outline=hexrgb(b["edge"]))
        d.line(pts+[pts[0]], fill=hexrgb(b["edge"]), width=int(1.5*S))
        dtext(cx(n), cy(n), b["text"].split("\n"), b["fs"], b["tcol"], bold=True)
    else:
        d.rounded_rectangle([x0*S, y0*S, (x0+w)*S, (y0+h)*S], radius=10*S, fill=hexrgb(b["fill"]),
                            outline=hexrgb(b["edge"]), width=int(1.5*S))
        if b["dashed"]:
            dashed_rect(x0, y0, x0+w, y0+h, b["edge"], 1)
        dtext(cx(n), cy(n), b["text"].split("\n"), b["fs"], b["tcol"])
img.save(os.path.join(FIG, "vote_pipeline.png"))
print("wrote vote_pipeline.png", img.size)
