# Artifact Naming Convention (locked 2026-06-08)

Replaces the old `variant_a/b/c` scheme. **Go-forward only** — existing
`variant_*` depth dirs and Drive checkpoints are left in place (configs and
runbooks still reference them); apply this to all NEW artifacts.

## Format

    Model_Changes_YYYYMMDD

- Components separated by underscore `_`.
- Within a component, join multiple words with hyphens: `c1-001`, `back4000`, `paperfaith-hash19`.
- `Model` = the thing that produced the artifact (SLAM method, or depth model).
- `Changes` = the descriptive delta — what makes this run/artifact distinct. No more `variant_a`; say what changed.
- `YYYYMMDD` = run date.

## Where things live

Depth maps live **as an element of the dataset**, beside `rgb/` and `seg/`:

    <dataset>/<snippet>/
        rgb/
        seg/
        depth/
            MoGe2_c1-001_20260608/          <- a named depth source
            RAFTStereo_c1-001_20260608/

`depth_subdir` in the config points at the chosen named folder.

## Examples

| Artifact | Name |
|---|---|
| Depth source (dir under `depth/`) | `MoGe2_c1-001_20260608/`, `RAFTStereo_p2-1-back4000_20260608/`, `Stereo_trail3_20260608/` |
| Model checkpoint | `DDS-SLAM_crcd-c1-001-mogedepth_20260608.pt` |
| Run output / `exp_name` | `DDS-SLAM_semsup-paperfaith-hash19_20260608` |
| 6-panel video | `DDS-SLAM_semsup-paperfaith_20260608_6panel.mp4` |

For depth artifacts the "Model" slot is the **depth generator** (MoGe2,
RAFTStereo, Stereo, AF-SfM); for SLAM artifacts it is the **SLAM method**
(DDS-SLAM, SNI-SLAM, Co-SLAM).
