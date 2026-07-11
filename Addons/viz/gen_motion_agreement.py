#!/usr/bin/env python
"""figures/motion_agreement.png — the gate's evidence, shown on two real frames.
Row 1: camera moving (C1 f115) - the district votes agree with one rigid story -> track.
Row 2: camera parked, instrument working tissue (C1 f312) - the epipolar dissent -> freeze.
Columns: input frame | semantic districts | votes coloured by epipolar agreement.
Districts/flow/Sampson computed with the deployed machinery (DINOv2-S/14-reg from the local
hub cache, dense flow, F-matrix RANSAC), ref frame delta=8 as in the gate."""
import os, math
import numpy as np
import cv2
import torch
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont

FIG = r"c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures"
BASE = r"C:\Users\benli\OneDrive\Desktop\Results\drive-download-20260711T053215Z-2-001\rect_bestbase_bdds_final_20260709"
INK = (32, 33, 36); MUT = (95, 99, 104); BLUE = (74, 114, 176); WARM = (194, 87, 26)
BAR = 26
PAL = np.array([[230,25,75],[60,180,75],[255,225,25],[0,130,200],[245,130,48],[145,30,180],
 [70,240,240],[240,50,230],[210,245,60],[250,190,190],[0,128,128],[170,110,40]],np.uint8)

def tile(cap, fr):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, img = cap.read(); assert ok, fr
    t = img[0:360, 0:480].copy(); t[:BAR] = t[BAR]
    return t

CASES = [("camera still", "C1_001_calm_v3gate_s0", 50, "quiet minority:\neven the quietest\ndistricts are still"),
         ("camera moving", "C2_001_calm_v3gate_s0", 110, "agreement: one rigid\nstory explains\nevery district"),
         ("scene deforming", "E3_005_calm_v3gate_s0", 128, "dissent: the majority\nrejects every\nrigid story")]
frames = {}
for _t, vidname, cur, _r in CASES:
    cap = cv2.VideoCapture(os.path.join(BASE, vidname, "panels.mp4"))
    frames[(vidname, cur)] = tile(cap, cur)
    frames[(vidname, cur - 8)] = tile(cap, cur - 8)
    cap.release()

hub = r"C:\Users\benli\.cache\torch\hub\facebookresearch_dinov2_main"
dino = torch.hub.load(hub, 'dinov2_vits14_reg', source='local', verbose=False).eval()

def districts(bgr, K=12):
    im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    H, W = im.shape[:2]; gh, gw = (H // 14) * 14, (W // 14) * 14
    im = cv2.resize(im, (gw, gh))
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    t = torch.from_numpy((im - mean) / std).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        tok = dino.forward_features(t)['x_norm_patchtokens'][0].numpy()
    X = tok.reshape(-1, tok.shape[-1]); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    lab = KMeans(K, n_init=4, random_state=0).fit_predict(X).reshape(gh // 14, gw // 14).astype(np.uint8)
    return cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)

def sampson(F, p1, p2):
    x1 = np.hstack([p1, np.ones((len(p1), 1))]); x2 = np.hstack([p2, np.ones((len(p2), 1))])
    Fx1 = x1 @ F.T; Ftx2 = x2 @ F
    d = np.sum(x2 * Fx1, 1) ** 2 / (Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2 + 1e-12)
    return np.sqrt(d)

dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
rows = []
for title, vidname, cur, reason in CASES:
    cv2.setRNGSeed(0)                       # findFundamentalMat RANSAC is stochastic otherwise
    a, b = frames[(vidname, cur - 8)], frames[(vidname, cur)]
    flow = dis.calc(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    lab = districts(b)
    H, W = flow.shape[:2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    p1 = np.stack([uu, vv], -1).reshape(-1, 2); p2 = p1 + flow.reshape(-1, 2)
    idx = np.linspace(0, len(p1) - 1, 4000).astype(np.int64)
    Fm, _ = cv2.findFundamentalMat(p1[idx], p2[idx], cv2.FM_RANSAC, 1.0, 0.999)
    samp = sampson(Fm.astype(np.float64), p1, p2).reshape(H, W)
    # per-district votes
    votes = []
    for k in range(12):
        m = lab == k
        if m.sum() < 400: continue
        ys, xs = np.where(m)
        fk = np.median(flow[m], axis=0); sk = float(np.median(samp[m]))
        votes.append((int(xs.mean()), int(ys.mean()), fk, sk, k))
    mags = sorted(np.hypot(v[2][0], v[2][1]) for v in votes)
    q10 = float(np.percentile([np.hypot(v[2][0], v[2][1]) for v in votes], 10))
    dis3 = float(np.mean([v[3] > 3.0 for v in votes]))
    # panels
    edges = cv2.Canny((lab * 20).astype(np.uint8), 1, 1)
    dcol = PAL[lab % 12][:, :, ::-1]
    dpan = cv2.addWeighted(b, 0.45, dcol, 0.55, 0); dpan[edges > 0] = (255, 255, 255)
    vpan = (b.astype(np.float32) * 0.45).astype(np.uint8); vpan[edges > 0] = (100, 100, 100)
    for (cx, cy, fk, sk, k) in votes:
        col = (26, 87, 194)[::-1] if sk > 3.0 else (176, 114, 74)   # BGR: WARM dissent / BLUE agree
        col = (26, 87, 194) if sk > 3.0 else (176, 114, 74)
        v = fk * 6.0
        cv2.circle(vpan, (cx, cy), 5, col, -1)
        cv2.arrowedLine(vpan, (cx, cy), (int(cx + v[0]), int(cy + v[1])), col, 2, tipLength=0.35)
    rows.append((title, reason, b, dpan, vpan, q10, dis3))

# ---------------- assemble with PIL ----------------
TW, TH = 480, 334   # crop the dead label strip
PADL, PADT, GAP = 210, 56, 8
W = PADL + 3 * TW + 2 * GAP + 20
H = PADT + 3 * TH + 2 * GAP + 60
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
F = lambda sz, b=False: ImageFont.truetype(r"C:\Windows\Fonts\arial" + ("bd" if b else "") + ".ttf", sz)
heads = ["input frame", "semantic districts (k-means on DINO patch tokens)", "district votes and epipolar agreement"]
for c, htxt in enumerate(heads):
    d.text((PADL + c * (TW + GAP) + TW // 2, PADT - 26), htxt, font=F(15, True), fill=INK, anchor="mm")
for r, (title, reason, b, dpan, vpan, q10, dis3) in enumerate(rows):
    y = PADT + r * (TH + GAP)
    for c, pan in enumerate((b, dpan, vpan)):
        p = Image.fromarray(cv2.cvtColor(pan[BAR:, :], cv2.COLOR_BGR2RGB)).resize((TW, TH))
        img.paste(p, (PADL + c * (TW + GAP), y))
    d.text((16, y + 40), title, font=F(16, True), fill=INK)
    d.text((16, y + 72), f"q10 = {q10:.1f} px\ndis3 = {dis3:.2f}", font=F(13), fill=MUT)
    verdict = "track" if not (q10 < 2.5 or dis3 > 0.5) else "freeze"
    vcol = tuple(BLUE) if verdict == "track" else tuple(WARM)
    d.text((16, y + 128), reason, font=F(12), fill=MUT)
    d.text((16, y + 200), f"-> {verdict}", font=F(17, True), fill=vcol)
d.text((PADL, H - 34), "arrow = district median flow (x6); ", font=F(12), fill=MUT)
d.text((PADL + 232, H - 34), "blue = consistent with one rigid story, ", font=F(12), fill=tuple(BLUE))
d.text((PADL + 480, H - 34), "orange = dissents (Sampson > 3 px)", font=F(12), fill=tuple(WARM))
img.save(os.path.join(FIG, "motion_agreement.png"))
print("wrote motion_agreement.png", img.size)
