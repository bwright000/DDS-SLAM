# Full 3D Reconstruction Plan — GT vs DDS-SLAM on Sem SuPer Dataset

## Context

DDS-SLAM produces a trained neural SDF (saved as `checkpoint150.pt`) and estimated camera poses, but **never actually extracts a mesh** (the `#TODO: Evaluation of reconstruction` at `ddsslam.py:623`). Meanwhile, the pre-generated depth maps (Depth Anything V2 / Monodepth2) + estimated poses can produce a direct depth-based 3D reconstruction for comparison. The goal is a standalone script that produces both reconstructions as `.ply` files and optionally visualises them.

## Approach

Create `Addons/reconstruct_3d.py` — a single script with two reconstruction paths:

### Path A: Depth-Based (GT) Reconstruction
Back-project depth maps + RGB images using camera intrinsics and DDS-SLAM estimated poses into a fused 3D representation.

1. **Load** estimated poses from `est_c2w_data.txt` (reuse pattern from `visualize_run.py:load_poses_from_txt`)
2. **Load** depth `.npy` files + RGB images for each frame
3. **For each frame**: un-project pixels → 3D points using intrinsics (fx, fy, cx, cy) and the c2w pose, with RGB colour per point
4. **TSDF Fusion** (Open3D): Integrate depth frames into a TSDF volume, then extract mesh via marching cubes — this gives a watertight, noise-filtered mesh
5. **Fallback**: If Open3D unavailable, produce a coloured point cloud `.ply` via trimesh
6. **Optional semantic colouring**: Use `seg/png_masks/` to colour vertices by tissue class instead of RGB

### Path B: DDS-SLAM Neural SDF Mesh Extraction
Extract the implicit surface from the trained checkpoint.

1. **Load config** from `config.json` saved alongside the checkpoint
2. **Reconstruct `JointEncoding` model** with the config and bounding box
3. **Load checkpoint** weights via `model.load_state_dict(ckpt['model'])`
4. **Call `extract_mesh()`** from `utils.py` with:
   - `query_fn=model.query_sdf`
   - `color_func=model.query_color` (optional, for per-vertex colour)
   - `voxel_size` from config `mesh.voxel_final` (0.03)
   - Bounding box from config `mapping.bound`
5. **Save** as `.ply`

### Output
- `output/DDS-SLAM-Results/trail3_depth_anything/demo/mesh_gt_depth.ply` — depth-based reconstruction
- `output/DDS-SLAM-Results/trail3_depth_anything/demo/mesh_ddsslam.ply` — neural SDF reconstruction
- Optional: combined Rerun visualisation via existing `visualize_run.py --mesh` flag

## Key Files

| File | Role |
|------|------|
| `Addons/reconstruct_3d.py` | **NEW** — main script |
| `utils.py:extract_mesh()` (L64-153) | Reuse for neural SDF mesh extraction |
| `model/scene_rep.py:JointEncoding` | Model class to reconstruct from checkpoint |
| `model/scene_rep.py:query_sdf` (L176-200) | SDF query function for marching cubes |
| `configs/Super/trail3.yaml` + `Super.yaml` | Config for model reconstruction |
| `config.py:load_config()` | Config loader |
| `Addons/visualize_run.py:load_poses_from_txt()` | Pose loading utility |

## Script Interface

```
python Addons/reconstruct_3d.py \
  --config configs/Super/trail3.yaml \
  --checkpoint output/DDS-SLAM-Results/trail3_depth_anything/demo/checkpoint150.pt \
  --datadir data/v2_data/trial_3 \
  --depth_dir output/DDS-SLAM-Results/depth_maps_depth_anything \
  --posefile output/DDS-SLAM-Results/trail3_depth_anything/demo/est_c2w_data.txt \
  --output_dir output/DDS-SLAM-Results/trail3_depth_anything/demo \
  --mode both                  # "gt", "neural", or "both"
  --semantic                   # optional: use semantic mask colouring
  --voxel_size 0.03            # marching cubes voxel size
  --tsdf_voxel 0.004           # TSDF voxel size for depth fusion
  --skip 1                     # use every N-th frame for GT reconstruction
```

## Implementation Detail

### Depth-based reconstruction (Open3D TSDF)

```python
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=tsdf_voxel,
    sdf_trunc=0.02,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)
for i, (rgb_path, depth_path, pose) in enumerate(frames):
    color = o3d.io.read_image(rgb_path)
    depth_np = np.load(depth_path) / depth_scale  # to meters
    depth_o3d = o3d.geometry.Image(depth_np.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth_o3d, ...)

    # Convert c2w (OpenGL) → w2c (OpenCV) for Open3D
    c2w_cv = pose @ GL_TO_CV
    w2c = np.linalg.inv(c2w_cv)

    volume.integrate(rgbd, intrinsic, o3d.core.Tensor(w2c))

mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh("mesh_gt_depth.ply", mesh)
```

### Neural SDF reconstruction

```python
cfg = config.load_config(args.config)
bound = torch.tensor(cfg['mapping']['bound']).float()
model = JointEncoding(cfg, bound).to(device)
ckpt = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(ckpt['model'])
model.eval()

mesh = extract_mesh(
    query_fn=model.query_sdf,
    config=cfg,
    bounding_box=bound.to(device),
    marching_cube_bound=torch.tensor(cfg['mapping']['marching_cubes_bound']).float().to(device),
    color_func=model.query_color if hasattr(model, 'query_color') else None,
    voxel_size=cfg['mesh']['voxel_final'],
    mesh_savepath="mesh_ddsslam.ply"
)
```

### Semantic colouring (optional overlay)

```python
SEMANTIC_COLORS = {
    0: [128, 128, 128],  # background
    1: [255, 0, 0],      # tissue class 1
    2: [0, 255, 0],      # tissue class 2
    ...
}
# For each point, lookup nearest semantic label from masks
```

## Verification

1. **Run with `--mode gt`**: Produces `mesh_gt_depth.ply` — open in MeshLab to verify geometry
2. **Run with `--mode neural`**: Produces `mesh_ddsslam.ply` — open in MeshLab to verify the neural surface
3. **Run with `--mode both`**: Produces both, prints vertex/face counts for each
4. **Visualise**: Use existing `visualize_run.py --mesh mesh_ddsslam.ply` to overlay mesh on camera trajectory in Rerun
5. **Compare**: Load both .ply files in MeshLab side-by-side to compare GT depth-based vs neural SDF reconstruction quality

## Dependencies

- `open3d` — for TSDF fusion (depth-based path). Script will gracefully fall back to point cloud if unavailable
- `torch` + `tinycudann` — for neural SDF extraction (GPU recommended)
- `trimesh` — already in project, for mesh I/O
- `numpy`, `cv2` — already in project

## Alternative Approaches Considered

### AMB3R (CVPR 2026) & MoGe (CVPR 2025)
Both were evaluated as potential alternatives:
- **AMB3R**: Feed-forward multi-view RGB → point cloud. No mesh output, no dynamic scene handling, not trained on medical data. Could serve as an alternative depth source but doesn't replace the reconstruction pipeline.
- **MoGe/MoGe-2**: Single-image → point map/depth. Essentially a monocular depth estimator like Depth Anything V2. Could be added to `generate_depth.py` as another depth backbone option but doesn't change the reconstruction approach.

**Conclusion**: TSDF fusion + neural SDF marching cubes remains the correct approach for this dataset. These alternatives could be explored as depth source replacements in future work.
