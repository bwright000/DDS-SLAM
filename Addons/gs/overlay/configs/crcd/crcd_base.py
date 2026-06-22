import os
import yaml

# GS migration Phase-0 run config: UNMODIFIED EndoGSLAM on CRCD with MoGe-2 depth.
# This is a verbatim copy of configs/c3vd/c3vd_base.py with ONLY the `data` block re-pointed at CRCD
# (+ resolution). Tracking/mapping/pruning are left EXACTLY as the surgical base ships, because Phase-0 is
# the "does the base even drive on our road" gate — no contributions, no tuning. See
# DDS-SLAM/Addons/docs/GS_MIGRATION_HANDOVER_20260622.md §5/§14.

scenes = ["C1_001"]   # add more CRCD snippets later; one snippet = the Phase-0 gate

primary_device = "cuda:0"
seed = int(os.environ.get("SEED", 0))     # n=3: run with SEED=0,1,2 -> separate output dirs (below)
try:
    scene_name = scenes[int(os.environ["SCENE_NUM"])]
except KeyError:
    scene_name = "C1_001"

map_every = 1
keyframe_every = 8

# --- overnight config-sweep knobs (env-driven; ALL defaults = base => bit-identical baseline) ---
_TAG  = os.environ.get("RUN_TAG", "base")            # arm name -> run-dir suffix
_LRT  = float(os.environ.get("LR_TRANS_MULT", 1.0))  # cam_trans LR x mult  (#0 proven lever: lower => less jitter)
_LRR  = float(os.environ.get("LR_ROT_MULT", 1.0))    # cam_rot   LR x mult
_SIL  = float(os.environ.get("SIL_THRES", 0.99))     # tracking silhouette mask threshold
_FWD  = bool(int(os.environ.get("FWD_PROP", 1)))     # const-velocity init (0=off; CRCD is static-heavy)
_DENS = bool(int(os.environ.get("DENSIFY", 0)))      # GS-gradient densification (coverage; costs runtime/VRAM)
tracking_iters = int(os.environ.get("TRK_ITERS", 15))
mapping_iters  = int(os.environ.get("MAP_ITERS", 25))

group_name = "CRCD_base"
run_name = f"{scene_name}_{_TAG}_s{seed}"            # experiments/CRCD_base/C1_001_<tag>_s<seed>

# Native rectified res from the staging-generated data yaml (matches SGS's "native res" convention so
# the GS-migration number is comparable to SGS c1_001=3.31mm). DOWNSAMPLE=2 halves it if a long snippet
# OOMs the T4 (SGS ran c1_001=360 frames native fine; OOM only hit >~450 frames @1280x720).
_DS = int(os.environ.get("DOWNSAMPLE", 1))
try:
    _cam = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "data", "crcd.yaml"), encoding="utf-8"))["camera_params"]
    _H, _W = int(_cam["image_height"]) // _DS, int(_cam["image_width"]) // _DS
except Exception:
    _H, _W = 720 // _DS, 1280 // _DS

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,
    keyframe_every=keyframe_every,
    distance_keyframe_selection=True,
    distance_current_frame_prob=0.1,
    mapping_window_size=-1,
    report_global_progress_every=2000,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=True,        # Phase-0: keep final params so the Sim3-ATE wrapper can read the trajectory
    checkpoint_interval=int(1e10),
    data=dict(
        basedir="./data/CRCD",                      # stage CRCD-Published <seq>/ under here on Colab
        gradslam_data_cfg="./configs/data/crcd.yaml",  # CRCD intrinsics + file globs (Brick 1)
        sequence=scene_name,                        # e.g. C1_001 (CRCD-Published 360-frame snippet)
        desired_image_height=_H,                    # native rectified res / DOWNSAMPLE (default native, == SGS)
        desired_image_width=_W,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,                              # -1 => all 360 frames
        train_or_test="all",                        # 🚨 NOT 'train' — we need every frame for SLAM + 360-ATE
    ),
    tracking=dict(                                  # base values; sweep knobs override via env (defaults = base)
        use_gt_poses=False,                         # true online SLAM: estimate pose, GT only for ATE
        forward_prop=_FWD,                          # constant-velocity init (FWD_PROP=0 to disable)
        num_iters=tracking_iters,                   # TRK_ITERS
        use_sil_for_loss=True,
        sil_thres=_SIL,                             # SIL_THRES
        use_l1=True,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.0, rgb_colors=0.0, unnorm_rotations=0.0,
            logit_opacities=0.0, log_scales=0.0,
            cam_unnorm_rots=0.002 * _LRR, cam_trans=0.005 * _LRT,   # LR_ROT_MULT / LR_TRANS_MULT
        ),
    ),
    mapping=dict(                                   # ← UNMODIFIED from c3vd_base.py
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=1.0, depth=1.0),
        lrs=dict(
            means3D=0.0001, rgb_colors=0.0025, unnorm_rotations=0.001,
            logit_opacities=0.05, log_scales=0.001,
            cam_unnorm_rots=0.000, cam_trans=0.000,
        ),
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0, remove_big_after=0, stop_after=20, prune_every=20,
            removal_opacity_threshold=0.005, final_removal_opacity_threshold=0.005,
            reset_opacities=False, reset_opacities_every=int(1e10),
        ),
        use_gaussian_splatting_densification=_DENS,   # DENSIFY=1 to enable GS-gradient clone/split
        densify_dict=dict(
            start_after=500, remove_big_after=3000, stop_after=5000, densify_every=100,
            grad_thresh=0.0002, num_to_split_into=2,
            removal_opacity_threshold=0.005, final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),
    ),
    viz=dict(
        render_mode='color',
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=False,
        viz_w=320, viz_h=320,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=30,
        enter_interactive_post_online=False,        # headless Colab
        gaussian_simplification=True,
    ),
)
