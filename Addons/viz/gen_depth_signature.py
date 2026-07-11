#!/usr/bin/env python
"""figures/depth_signature.png (+ .svg) — the schematic behind eq:vote-fit.
Three panels, one per motion type, each showing the flow a NEAR district (solid) and a FAR
district (hollow) observe: a camera TURN is depth-blind (equal vectors), a lateral SLIDE is
depth-scaled (near moves more, as 1/Z), and forward motion is a depth-scaled radial ZOOM.
Fitting the three simultaneously separates the motion types and cancels the depth scale.
House style: Arial, #9aa0a6 arrows, one shared layout spec emitting SVG and PNG."""
import os, math, base64
from PIL import Image, ImageDraw, ImageFont

FIG = r"c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures"
W, H = 1500, 560
INK = "#202124"; MUT = "#5f6368"; ARROW = "#9aa0a6"
BLUE = "#4a72b0"; WARM = "#c2571a"
PW = 440; PH = 400; PX = [40, 530, 1020]; PY = 90

# districts: (x, y, r, near?) in panel-local coords; flow vectors per panel computed below
NEAR = [(120, 300, 46), (300, 260, 40)]
FAR = [(150, 120, 24), (330, 150, 20)]
CX, CY = PW // 2, PH // 2   # principal point for the zoom panel

def flows(panel):
    out = []
    for (x, y, r) in NEAR + FAR:
        near = (x, y, r) in NEAR
        if panel == 0:      # turn: depth-blind, uniform
            v = (60, 0)
        elif panel == 1:    # slide: depth-scaled, near moves more
            v = (78 if near else 26, 0)
        else:               # zoom: radial, depth-scaled
            dx, dy = x - CX, y - CY
            n = math.hypot(dx, dy) or 1
            m = 66 if near else 24
            v = (dx / n * m, dy / n * m)
        out.append(((x, y, r), near, v))
    return out

TITLES = [("camera turn  (u)", "depth-blind: every district shifts equally"),
          ("lateral slide  (v)", "depth-scaled: near districts move more, as 1/Z"),
          ("forward motion  (d)", "depth-scaled radial: the field expands from the centre")]

# ---------------- PNG ----------------
S = 2
img = Image.new("RGB", (W * S, H * S), "white")
d = ImageDraw.Draw(img)
F = lambda sz, b=False: ImageFont.truetype(r"C:\Windows\Fonts\arial" + ("bd" if b else "") + ".ttf", sz * S)
hx = lambda h: tuple(int(h.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

d.text((W // 2 * S, 26 * S), "the depth signature of each motion type", font=F(19, True), fill=hx(INK), anchor="mm")
for p in range(3):
    ox = PX[p]
    d.rounded_rectangle([ox * S, PY * S, (ox + PW) * S, (PY + PH) * S], radius=12 * S,
                        outline=hx("#c8ccd0"), width=S, fill=(250, 250, 251))
    d.text(((ox + PW // 2) * S, (PY + 26) * S), TITLES[p][0], font=F(15, True), fill=hx(INK), anchor="mm")
    d.text(((ox + PW // 2) * S, (PY + PH + 22) * S), TITLES[p][1], font=F(12), fill=hx(MUT), anchor="mm")
    if p == 2:
        d.ellipse([(ox + CX - 4) * S, (PY + CY - 4) * S, (ox + CX + 4) * S, (PY + CY + 4) * S], fill=hx(MUT))
        d.text(((ox + CX) * S, (PY + CY + 16) * S), "principal point", font=F(10), fill=hx(MUT), anchor="mm")
    for (x, y, r), near, (vx, vy) in flows(p):
        cx, cy = ox + x, PY + y
        col = hx(BLUE) if near else hx(WARM)
        if near:
            d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], fill=col)
        else:
            d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], outline=col, width=3 * S)
        ang = math.atan2(vy, vx)
        m = math.hypot(vx, vy)
        x1e, y1e = cx + (r + 5) * math.cos(ang), cy + (r + 5) * math.sin(ang)
        x2, y2 = x1e + m * math.cos(ang), y1e + m * math.sin(ang)
        d.line([x1e * S, y1e * S, x2 * S, y2 * S],
               fill=hx(ARROW), width=4 * S)
        hl = 12 * S
        d.polygon([(x2 * S, y2 * S),
                   (x2 * S - hl * math.cos(ang + 0.45), y2 * S - hl * math.sin(ang + 0.45)),
                   (x2 * S - hl * math.cos(ang - 0.45), y2 * S - hl * math.sin(ang - 0.45))], fill=hx(ARROW))
# legend
lx, ly = 40, 58
d.ellipse([lx * S, (ly - 8) * S, (lx + 16) * S, (ly + 8) * S], fill=hx(BLUE))
d.text(((lx + 26) * S, ly * S), "near district (small Z)", font=F(12), fill=hx(INK), anchor="lm")
d.ellipse([(lx + 210) * S, (ly - 8) * S, (lx + 226) * S, (ly + 8) * S], outline=hx(WARM), width=3 * S)
d.text(((lx + 236) * S, ly * S), "far district (large Z)", font=F(12), fill=hx(INK), anchor="lm")
d.text(((lx + 430) * S, ly * S), "arrows: observed optical flow f_k", font=F(12), fill=hx(MUT), anchor="lm")
img.save(os.path.join(FIG, "depth_signature.png"))
print("wrote depth_signature.png", img.size)

# ---------------- SVG (same spec) ----------------
def esc(s): return s.replace("&", "&amp;")
s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
     f'font-family="Arial, Helvetica, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     f'<defs><marker id="ar" markerWidth="10" markerHeight="10" refX="7.5" refY="3" orient="auto" '
     f'markerUnits="strokeWidth"><path d="M0,0 L7,3 L0,6 Z" fill="{ARROW}"/></marker></defs>',
     f'<text x="{W//2}" y="30" font-size="19" font-weight="700" fill="{INK}" text-anchor="middle">the depth signature of each motion type</text>']
for p in range(3):
    ox = PX[p]
    s.append(f'<rect x="{ox}" y="{PY}" width="{PW}" height="{PH}" rx="12" fill="#fafafb" stroke="#c8ccd0"/>')
    s.append(f'<text x="{ox+PW//2}" y="{PY+30}" font-size="15" font-weight="700" fill="{INK}" text-anchor="middle">{esc(TITLES[p][0])}</text>')
    s.append(f'<text x="{ox+PW//2}" y="{PY+PH+26}" font-size="12" fill="{MUT}" text-anchor="middle">{esc(TITLES[p][1])}</text>')
    if p == 2:
        s.append(f'<circle cx="{ox+CX}" cy="{PY+CY}" r="4" fill="{MUT}"/>')
        s.append(f'<text x="{ox+CX}" y="{PY+CY+20}" font-size="10" fill="{MUT}" text-anchor="middle">principal point</text>')
    for (x, y, r), near, (vx, vy) in flows(p):
        cx, cy = ox + x, PY + y
        col = BLUE if near else WARM
        fill = col if near else 'none'
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{col}" stroke-width="3"/>')
        ang = math.atan2(vy, vx)
        m = math.hypot(vx, vy)
        x1e, y1e = cx + (r + 5) * math.cos(ang), cy + (r + 5) * math.sin(ang)
        s.append(f'<path d="M{x1e:.0f},{y1e:.0f} '
                 f'L{x1e + m*math.cos(ang):.0f},{y1e + m*math.sin(ang):.0f}" stroke="{ARROW}" stroke-width="4" fill="none" marker-end="url(#ar)"/>')
ly = 58
s.append(f'<circle cx="48" cy="{ly}" r="8" fill="{BLUE}"/><text x="66" y="{ly+4}" font-size="12" fill="{INK}">near district (small Z)</text>')
s.append(f'<circle cx="258" cy="{ly}" r="8" fill="none" stroke="{WARM}" stroke-width="3"/><text x="276" y="{ly+4}" font-size="12" fill="{INK}">far district (large Z)</text>')
s.append(f'<text x="470" y="{ly+4}" font-size="12" fill="{MUT}">arrows: observed optical flow f_k</text>')
s.append('</svg>')
open(os.path.join(FIG, "depth_signature.svg"), "w", encoding="utf-8").write("\n".join(s))
print("wrote depth_signature.svg")
