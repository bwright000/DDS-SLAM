#!/usr/bin/env python3
"""Idempotently apply the v1 flow_map (mapping catch-up) edits to EndoGSLAM scripts/main.py.

All edits are PARITY-SAFE: with the flow_map config block absent or enable=False, every edit is a no-op
(get_loss takes the base branch; `_fm_gate is None` -> `_fm_w`/`_fm_mw` None; `_fm_holdout_k=0`; the
gs_flow_gate import lives inside `if enable`). Anchors verified against base 6338c643. Errors loudly on
drift. Run AFTER apply_patches.py (the CRCD loader patches).

  python Addons/gs/apply_patches_flowmap.py <ENDOGSLAM_DIR>
"""
import os
import sys

ENDO = sys.argv[1] if len(sys.argv) > 1 else "/content/EndoGSLAM"
M = "scripts/main.py"

PATCHES = [
    # --- Edit A: get_loss signature — append ONE default-None kwarg (existing positional calls unaffected) ---
    (
        M, "map_weight=None):",
        "visualize_tracking_loss=False, tracking_iteration=None):",
        "visualize_tracking_loss=False, tracking_iteration=None, map_weight=None):",
    ),
    # --- Edit B: mapping depth residual — per-pixel DEPTH up-weight (geometry path; NOT mean-preserving) ---
    (
        M, "# FLOWMAP depth up-weight",
        "        else:\n"
        "            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()",
        "        else:\n"
        "            # FLOWMAP depth up-weight (v1 mapping catch-up): per-pixel up-weight of the DEPTH\n"
        "            # (geometry) residual where the scene moved -> map re-fits the moved surface harder.\n"
        "            # NOT mean-preserving: w floored at 1 (static-pixel gradient never reduced); w==None -> base.\n"
        "            depth_resid = torch.abs(curr_data['depth'] - depth)\n"
        "            if mapping and (map_weight is not None):\n"
        "                assert map_weight.shape == depth.shape, (map_weight.shape, depth.shape)\n"
        "                depth_resid = depth_resid * map_weight\n"
        "            losses['depth'] = depth_resid[mask].mean()",
    ),
    # --- gate-build: build the GSFlowGate ONCE before the frame loop (import isolated inside `if enable`) ---
    (
        M, "# FLOWMAP gate-build",
        "    # Iterate over Scan\n"
        "    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):",
        "    # FLOWMAP gate-build: mapping-catchup gate ONCE (only if enabled; import isolated for parity)\n"
        "    import os as _os_fm, time as _time_fm\n"
        "    _fm_cfg = config.get('flow_map', {})\n"
        "    _fm_gate = None\n"
        "    if _fm_cfg.get('enable', False):\n"
        "        from Addons.motion.gs_flow_gate import GSFlowGate\n"
        "        _fm_gate = GSFlowGate(config, device)\n"
        "        _K_fm = intrinsics.detach().cpu().numpy() if hasattr(intrinsics, 'detach') else np.asarray(intrinsics)\n"
        "        _fm_gate.set_intrinsics(_K_fm[:3, :3])\n"
        "        print(f'[flow_map] ENABLED lam={_fm_gate.lam} deadband={_fm_gate.deadband} depth_db={_fm_gate.depth_db} ref_stride={_fm_gate.ref_stride} uniform={_fm_gate.uniform} holdout={_fm_cfg.get(\"eval_holdout_every\",0)}', flush=True)\n"
        "    _fm_holdout_k = int(_fm_cfg.get('eval_holdout_every', 0))\n"
        "    _fm_holdout_idx = []\n"
        "    _fm_loop_t0 = _time_fm.time()\n"
        "    # Iterate over Scan\n"
        "    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):",
    ),
    # --- heartbeat: one clean line per frame (gated on FM_HEARTBEAT; tqdm is suppressed in the runbook) ---
    (
        M, "FM_HB frame",
        "        print() # always show global iteration",
        "        print() # always show global iteration\n"
        "        if _os_fm.environ.get('FM_HEARTBEAT'):\n"
        "            _fm_el = _time_fm.time() - _fm_loop_t0\n"
        "            _fm_eta = _fm_el / max(time_idx - checkpoint_time_idx + 1, 1) * (num_frames - 1 - time_idx)\n"
        "            print(f'FM_HB frame {time_idx+1}/{num_frames} \\u00b7 {_fm_el:.0f}s \\u00b7 eta {_fm_eta:.0f}s', flush=True)",
    ),
    # --- per-frame weight (once/frame; feeds the gate buffer EVERY frame) + held-out skip of mapping ---
    (
        M, "# FLOWMAP per-frame mapping-catchup weight",
        "        # Densification & KeyFrame-based Mapping\n"
        "        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:",
        "        # FLOWMAP per-frame mapping-catchup weight (once/frame; feeds the gate buffer every frame)\n"
        "        _fm_w = None\n"
        "        if _fm_gate is not None:\n"
        "            with torch.no_grad():\n"
        "                _fm_w2c = torch.eye(4).cuda().float()\n"
        "                _fm_w2c[:3, :3] = build_rotation(F.normalize(params['cam_unnorm_rots'][..., time_idx].detach()))\n"
        "                _fm_w2c[:3, 3] = params['cam_trans'][..., time_idx].detach()\n"
        "                _fm_img = np.ascontiguousarray((color.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()[:, :, ::-1])\n"
        "                _fm_wm = _fm_gate.step(_fm_img, depth[0].detach().cpu().numpy(), _fm_w2c.cpu().numpy(), time_idx)\n"
        "            _fm_w = torch.from_numpy(_fm_wm).to(device=depth.device, dtype=depth.dtype).unsqueeze(0)\n"
        "        _fm_is_holdout = (_fm_holdout_k > 0 and time_idx > 0 and (time_idx % _fm_holdout_k == _fm_holdout_k - 1))\n"
        "        if _fm_is_holdout:\n"
        "            _fm_holdout_idx.append(time_idx)\n"
        "        # Densification & KeyFrame-based Mapping\n"
        "        if (time_idx == 0 or (time_idx+1) % config['map_every'] == 0) and not _fm_is_holdout:",
    ),
    # --- call-site: attach the weight ONLY for the current-frame mapping iter (iter_time_idx == time_idx) ---
    (
        M, "_fm_mw = _fm_w if",
        "                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],",
        "                _fm_mw = _fm_w if (_fm_w is not None and iter_time_idx == time_idx) else None\n"
        "                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],",
    ),
    (
        M, "map_weight=_fm_mw)",
        "config['mapping']['ignore_outlier_depth_loss'], mapping=True)",
        "config['mapping']['ignore_outlier_depth_loss'], mapping=True, map_weight=_fm_mw)",
    ),
]


def main():
    for rel, marker, old, new in PATCHES:
        path = os.path.join(ENDO, rel)
        if not os.path.isfile(path):
            sys.exit(f"[flowmap-patch] MISSING {path} — wrong ENDOGSLAM_DIR?")
        s = open(path, encoding="utf-8").read()
        if marker in s:
            print(f"[flowmap-patch] {rel} :: {marker[:32]} already applied")
            continue
        if old not in s:
            sys.exit(f"[flowmap-patch] ANCHOR NOT FOUND in {rel} (drift from base 6338c643):\n  {old.splitlines()[0]}")
        open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
        print(f"[flowmap-patch] {rel} :: {marker[:32]} PATCHED")
    print("[flowmap-patch] all v1 flow_map edits applied.")


if __name__ == "__main__":
    main()
