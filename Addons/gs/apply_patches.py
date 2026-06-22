#!/usr/bin/env python3
"""Idempotently apply the 3 in-place EndoGSLAM edits needed by the CRCD adapter.

We patch (not vendor) the authors' files so our repo never redistributes EndoGSLAM code
(it has no LICENSE — see GS_MIGRATION caution C1). Anchored to the pinned base commit
6338c643; if an anchor is missing the script errors loudly (= EndoGSLAM version drift).

Usage:  python apply_patches.py <ENDOGSLAM_DIR>
"""
import os
import sys

ENDO = sys.argv[1] if len(sys.argv) > 1 else "/content/EndoGSLAM"

# (file, idempotency-marker, old, new)
PATCHES = [
    # 1. register the loader in the package __init__
    (
        "datasets/gradslam_datasets/__init__.py",
        "from .crcd import CRCDDataset",
        "from .endoslam import EndoSLAMDataset\nfrom .c3vd import C3VDDataset",
        "from .endoslam import EndoSLAMDataset\nfrom .c3vd import C3VDDataset\nfrom .crcd import CRCDDataset",
    ),
    # 2. import CRCDDataset in main.py
    (
        "scripts/main.py",
        "    CRCDDataset\n)",
        "    load_dataset_config,\n    EndoSLAMDataset,\n    C3VDDataset\n)",
        "    load_dataset_config,\n    EndoSLAMDataset,\n    C3VDDataset,\n    CRCDDataset\n)",
    ),
    # 3. register 'crcd' in the get_dataset factory
    (
        "scripts/main.py",
        '["crcd"]',
        '    elif config_dict["dataset_name"].lower() in ["c3vd"]:\n'
        '        return C3VDDataset(config_dict, basedir, sequence, **kwargs)\n'
        "    else:\n"
        '        raise ValueError(f"Unknown dataset name {config_dict[\'dataset_name\']}")',
        '    elif config_dict["dataset_name"].lower() in ["c3vd"]:\n'
        '        return C3VDDataset(config_dict, basedir, sequence, **kwargs)\n'
        '    elif config_dict["dataset_name"].lower() in ["crcd"]:\n'
        '        return CRCDDataset(config_dict, basedir, sequence, **kwargs)\n'
        "    else:\n"
        '        raise ValueError(f"Unknown dataset name {config_dict[\'dataset_name\']}")',
    ),
    # 4. fix the always-true depth-extension dispatch so MoGe-2 .npy depth loads via np.load
    (
        "datasets/gradslam_datasets/basedataset.py",
        "_dext = os.path.splitext(depth_path)[1].lower()",
        '''        if ".png" or '.jpg' in depth_path:
            # if 'Pixelwise' in depth_path: # NOTE: we use this to identify unitycam endoslam dataset
            #     depth = np.asarray(PIL.Image.open(depth_path).convert("L"), dtype=np.float64)
            #     # depth = cv2.blur(depth, (10, 10))
            #     depth = cv2.GaussianBlur(depth, (21, 21), 10)
            #     depth = 1.0 / (depth + 1e-10) + 0.2
            # else:
                # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(imageio.imread(depth_path), dtype=np.float64)
        elif '.npy' in depth_path:
            depth = np.load(depth_path).astype(np.float64)
        elif '.tiff' in depth_path:
            depth = np.array(PIL.Image.open(depth_path), dtype=np.float64)
        else:
            raise ValueError("Depth image format not supported.")''',
        '''        # FIX (GS migration): the original dispatch `if ".png" or '.jpg' in depth_path:` is ALWAYS true
        # (the string literal ".png" is truthy), so .npy depth never reached np.load. Catch .npy first;
        # everything else keeps the original imageio path (handles .png/.jpg AND C3VD's .tiff unchanged).
        _dext = os.path.splitext(depth_path)[1].lower()
        if _dext == '.npy':
            depth = np.load(depth_path).astype(np.float64)
        else:
            depth = np.asarray(imageio.imread(depth_path), dtype=np.float64)''',
    ),
    # 5. neutralise the final cosmetic plot_video (IndexError on an empty keyframes dir; runs AFTER
    #    SLAM+eval+params.npz are saved, so it only adds a scary traceback to every run).
    (
        "scripts/main.py",
        "#plotvideo-off",
        "    plot_video(os.path.join(results_dir, 'eval', 'plots'), os.path.join('./experiments/', experiment.group_name, experiment.scene_name, 'keyframes'))",
        "    pass  #plotvideo-off cosmetic keyframe-plot video (IndexError on empty dir); SLAM+eval already saved",
    ),
    # 6. eval_save renders/saves NOTHING when train_or_test='all' (visall=False + every frame is 'train'
    #    -> the `continue` skips render+metric+plot). Flip it so the eval pass renders every frame at the
    #    estimated pose, saving color/depth + the 6-panel plots (our visual) and computing psnr/depth_l1.
    (
        "utils/eval_helpers.py",
        "visall = True",
        "        visall = False # NOTE: for debug",
        "        visall = True  # PATCHED (GS migration): render+save ALL frames (train_or_test='all')",
    ),
]


def main():
    for rel, marker, old, new in PATCHES:
        path = os.path.join(ENDO, rel)
        if not os.path.isfile(path):
            sys.exit(f"[patch] MISSING {path} — is ENDOGSLAM_DIR correct?")
        s = open(path, encoding="utf-8").read()
        if marker in s:
            print(f"[patch] {rel}: already applied")
            continue
        if old not in s:
            sys.exit(
                f"[patch] ANCHOR NOT FOUND in {rel} — EndoGSLAM differs from pinned base 6338c643.\n"
                f"        Re-pin or re-derive the patch. Expected snippet:\n        {old.splitlines()[0]}"
            )
        open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
        print(f"[patch] {rel}: PATCHED")
    print("[patch] all EndoGSLAM edits applied.")


if __name__ == "__main__":
    main()
