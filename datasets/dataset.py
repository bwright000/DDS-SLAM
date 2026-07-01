import glob
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from .utils import get_camera_rays, alphanum_key, as_intrinsics_matrix
import matplotlib.pyplot as plt
import re

def compute_edge(rgb_data, depth_data, instance=None, UseInstance=False):

    if not UseInstance :
        #edges0 = cv2.Canny(depth_data.astype(np.uint8), 8, 20)
        edges0 = cv2.adaptiveThreshold(depth_data.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        mask = (depth_data == 0).astype(np.uint8)
        kernel = np.ones((11, 11), np.uint8)
        kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        rgb_image = cv2.dilate(rgb_data, kernel_e, iterations=10)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        remaining_mask = cv2.bitwise_not(dilated_mask)
        remaining_mask[remaining_mask == 254] = 0
        edges_depth = edges0 #* remaining_mask
        edges_rgb = cv2.Canny(rgb_image, 60, 180)#*dilated_mask
        edges = edges_rgb + edges_depth
    else:
        edges = cv2.Canny(instance, 50, 150)
    edges = np.where(edges == 255, 0, 1).astype(np.uint8)
    dist_transform = cv2.distanceTransform(edges, cv2.DIST_L2, 0, dstType=cv2.CV_32F)
    edge_data = np.exp(-dist_transform / 10)

    return edge_data

def compute_edge_semantic(semantic_data, depth_data, instance=None, UseInstance=False):
    if not UseInstance :
        edges0 = cv2.adaptiveThreshold(depth_data.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        mask = (depth_data == 0).astype(np.uint8)                      
        kernel = np.ones((11, 11), np.uint8)                           
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)          
        remaining_mask = cv2.bitwise_not(dilated_mask)                
        remaining_mask[remaining_mask == 254] = 0                     
        edges_depth = edges0 #* remaining_mask                           
        edges_semantic = cv2.Canny(semantic_data, 1, 1)#*dilated_mask    
        edges = edges_semantic                         
    else:
        edges = cv2.Canny(instance, 50, 150)
    edges = np.where(edges == 255, 0, 1).astype(np.uint8)
    dist_transform = cv2.distanceTransform(edges, cv2.DIST_L2, 0, dstType=cv2.CV_32F)
    edge_data = np.exp(-dist_transform / 10)

    return edge_data

def create_border_data(depth_data, border_width=10):
    border_data = torch.zeros_like(depth_data, dtype=torch.float32)
    border_data[:border_width, :] = 1 
    border_data[-border_width:, :] = 1 
    border_data[:, :border_width] = 1 
    border_data[:, -border_width:] = 1 
    return border_data

def get_dataset(config):
    '''
    Get the dataset class from the config file.
    '''
    if config['dataset'] == 'stereomis':
        dataset = StereoMISDataset
        
    elif config['dataset'] == 'super':
        dataset = SuperDataset
    

    
    return dataset(config, 
                   config['data']['datadir'], 
                   trainskip=config['data']['trainskip'], 
                   downsample_factor=config['data']['downsample'], 
                   sc_factor=config['data']['sc_factor'])

class BaseDataset(Dataset):
    def __init__(self, cfg):
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.H, self.W = cfg['cam']['H']//cfg['data']['downsample'],\
            cfg['cam']['W']//cfg['data']['downsample']

        self.fx, self.fy =  cfg['cam']['fx']//cfg['data']['downsample'],\
             cfg['cam']['fy']//cfg['data']['downsample']
        self.cx, self.cy = cfg['cam']['cx']//cfg['data']['downsample'],\
             cfg['cam']['cy']//cfg['data']['downsample']
        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_edge'] if 'crop_edge' in cfg['cam'] else 0
        self.ignore_w = cfg['tracking']['ignore_edge_W']
        self.ignore_h = cfg['tracking']['ignore_edge_H']

        self.total_pixels = (self.H - self.crop_size*2) * (self.W - self.crop_size*2)
        self.num_rays_to_save = int(self.total_pixels * cfg['mapping']['n_pixels'])
        
    
    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def _attach_dino(self, ret, index, edge):
        """Inc-1 v2 (WildGS-faithful, mode:'dino'): attach the COMPACT DINO patch-grid [gh,gw,C]. The
        ray sites bilinear-SAMPLE it on-demand at their pixel coords (ddsslam.sample_dino_grid) ->
        ~13ms vs a 944ms/472MB full-frame upsample, and nothing per-pixel is stored. The grid covers
        the full image FOV; sites normalise by (H,W) and add any crop offset (crop_edge==0 in our
        configs). No-op (no key added) when dino is off -> base/geo bit-identical. Shared by
        StereoMISDataset + SuperDataset."""
        if getattr(self, 'dino_paths', None) is None:
            return ret
        grid = np.load(self.dino_paths[index]).astype(np.float32)           # [gh,gw,C] (fp16 on disk)
        ret["dino_grid"] = torch.from_numpy(grid)
        return ret

    def _attach_deform(self, ret, index):
        """ARM-2 Stage-1 (deformation_sup_weight>0): attach the baked Δx* deformation-target grid
        [gh,gw,3] + trust [gh,gw,1] for the field teacher. current_frame_mapping bilinear-samples them
        at the ray pixels (ddsslam.sample_dino_grid, same as dino). None => no key => base bit-identical.
        Shared by StereoMISDataset + SuperDataset."""
        if getattr(self, 'deform_paths', None) is None:
            return ret
        npz = np.load(self.deform_paths[index])
        # B (DEPTH-ANCHOR): scale the baked Δx* targets so the field's render-applied warp lands INSIDE the
        # metric render band. The field learns vox_motion->Δx* faithfully, but |Δx*|max=0.084 = 84% of the
        # ±0.1 band -> over-warps the metric SDF gates -> distortion (why teacher+B2 stalled at 25.9 in the
        # campaign). deform_target_scale<1 shrinks the field's magnitude toward metric so B2 can co-adapt the
        # map to a band-fitting warp. Single anchor point (consistent across current_frame_mapping + replay).
        # Default 1.0 = base byte-identical.
        _ts = float(self.config.get('training', {}).get('deform_target_scale', 1.0))
        _dx = npz['dx'].astype(np.float32)
        ret["deform_dx"] = torch.from_numpy(_dx if _ts == 1.0 else _dx * _ts)            # [gh,gw,3]
        ret["deform_trust"] = torch.from_numpy(npz['trust'].astype(np.float32))[..., None]  # [gh,gw,1]
        return ret

    def _attach_seg(self, ret, index, H, W, edge):
        """ARM-1 #4 (uncertainty.whatkind_weight>0): attach a per-frame CANONICAL seg-class map [H,W]
        for the what-kind CE prior. Modelled on _attach_deform: gated on the flag, None => no key =>
        base bit-identical. Reads the RAW mask (cv2.IMREAD_UNCHANGED -> the uint class ids, NOT the
        BGR-truncated copy __getitem__ uses for the Canny edge field), canonicalises per-dataset, and
        applies the SAME resize(NEAREST)->downsample->crop pipeline as edge_semantic/depth so the
        per-ray gather at [indice_h, indice_w] aligns 1:1. Canonical labels: {0=bg, 1=tissue, 2=tool}.

          CRCD masks: uint16 {0=bg, 1=Liver, 2=Gallbladder, 3=Tool} -> {0:0, 1:1, 2:1, 3:2}
                      (Liver+Gallbladder -> tissue, per the seg-policy memory).
        Shared by StereoMISDataset + SuperDataset (each passes its own seg_label_paths)."""
        if getattr(self, 'seg_label_paths', None) is None:
            return ret
        raw = cv2.imread(self.seg_label_paths[index], cv2.IMREAD_UNCHANGED)
        if raw is None:
            return ret
        if raw.ndim == 3:                       # collapse any accidental multi-channel to the first plane
            raw = raw[..., 0]
        raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_NEAREST)
        # canonicalise CRCD class ids {0 bg, 1 liver, 2 gallbladder, 3 tool} -> {0 bg, 1 tissue, 2 tool}
        seg = np.zeros_like(raw, dtype=np.int64)
        seg[(raw == 1) | (raw == 2)] = 1        # tissue
        seg[raw == 3] = 2                        # tool
        if self.downsample_factor > 1:
            seg = cv2.resize(seg.astype(np.int32), (W // self.downsample_factor, H // self.downsample_factor),
                             interpolation=cv2.INTER_NEAREST).astype(np.int64)
        if edge > 0:
            seg = seg[edge:-edge, edge:-edge]
        ret["seg"] = torch.from_numpy(seg.astype(np.int64))
        return ret

class StereoMISDataset(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1, 
                 downsample_factor=1, translation=0.0, 
                 sc_factor=1., crop=0):
        super(StereoMISDataset, self).__init__(cfg)

        self.config = cfg
        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.img_files = sorted(glob.glob(f'{self.basedir}/video_frames/*l.png'))[-4000:]
        self.depth_paths = sorted(glob.glob(f'{self.basedir}/depth/*.png'))[-4000:]

        self.semantic_paths = sorted(
           glob.glob(os.path.join(
           self.basedir, 'masks', '*.png')))[-2000:]

        # Inc-1 v2 (mode:'dino'): per-frame DINO feature grids, globbed from THIS run's basedir and
        # sliced with the SAME [-4000:] as img_files so frame i <-> dino_paths[i]. CRCD and StereoMIS
        # are DIFFERENT datasets that merely share this loader class (crcd.yaml dataset:'stereomis'),
        # so the path is per-run, never hardcoded. None => no 'dino' key => base/geo bit-identical.
        self.dino_paths = None
        _unc = self.config.get('uncertainty', {})
        if _unc.get('enable', False) and _unc.get('mode', 'geo') == 'dino':
            _sub = _unc.get('dino_subdir', 'dino')
            self.dino_paths = sorted(glob.glob(f'{self.basedir}/{_sub}/*_dino.npy'))[-4000:]
            assert len(self.dino_paths) == len(self.img_files), \
                f"DINO features {len(self.dino_paths)} != frames {len(self.img_files)} in {self.basedir}/{_sub}"

        # ARM-2 Stage-1: baked Δx* deformation targets (deformation_sup_weight>0). Same [-4000:] slice as
        # img_files so frame i <-> deform_paths[i]. None => no key => base bit-identical.
        self.deform_paths = None
        if self.config['training'].get('deformation_sup_weight', 0) > 0:
            _dsub = self.config.get('data', {}).get('deform_subdir', 'deform')
            self.deform_paths = sorted(glob.glob(f'{self.basedir}/{_dsub}/*_deform.npz'))[-4000:]
            assert len(self.deform_paths) == len(self.img_files), \
                f"deform targets {len(self.deform_paths)} != frames {len(self.img_files)} in {self.basedir}/{_dsub}"

        # ARM-1 #4: per-frame RAW seg masks for the what-kind CE prior (whatkind_weight>0). Reuses the
        # existing semantic_paths (same masks the edge field is built from) but read via
        # cv2.IMREAD_UNCHANGED in _attach_seg to recover the class ids. Align one mask per frame using
        # the SAME ratio rule as __getitem__ (1:1 CRCD vs half-rate StereoMIS) so seg_label_paths[index]
        # pairs correctly. None => no 'seg' key => base/per-pixel-dino bit-identical.
        self.seg_label_paths = None
        if self.config.get('uncertainty', {}).get('whatkind_weight', 0) > 0 or self.config.get('training', {}).get('tool_mask', False) or self.config.get('training', {}).get('tool_mask_track', False) or self.config.get('flow_track', {}).get('mode', '') == 'solve_pnp':
            if len(self.semantic_paths) >= len(self.img_files):
                self.seg_label_paths = [self.semantic_paths[min(i, len(self.semantic_paths) - 1)]
                                        for i in range(len(self.img_files))]
            else:
                self.seg_label_paths = [self.semantic_paths[min(i // 2, len(self.semantic_paths) - 1)]
                                        for i in range(len(self.img_files))]

        self.load_poses(self.basedir)

        self.rays_d = None
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)

        if self.config['cam']['crop_edge'] > 0:
            self.H -= self.config['cam']['crop_edge']*2
            self.W -= self.config['cam']['crop_edge']*2
            self.cx -= self.config['cam']['crop_edge']
            self.cy -= self.config['cam']['crop_edge']
   
    def __len__(self):
        return self.num_frames
  
    def __getitem__(self, index):
        color_path = self.img_files[index]
        depth_path = self.depth_paths[index]
        # Masks may be 1:1 (CRCD / SAM3-style, one label per frame) or half-rate
        # (StereoMIS native, one mask per two frames). Pick pairing by ratio.
        if len(self.semantic_paths) >= len(self.img_files):
            semantic_path = self.semantic_paths[min(index, len(self.semantic_paths) - 1)]
        else:
            semantic_path = self.semantic_paths[min(index // 2, len(self.semantic_paths) - 1)]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth_data0 = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            raise NotImplementedError()

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        semantic_data = cv2.imread(semantic_path)
        semantic_data = cv2.resize(semantic_data, (W, H))
        edge_data_semantic = compute_edge_semantic(semantic_data, depth_data0)

        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            self.fx = self.fx // self.downsample_factor
            self.fy = self.fy // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)
            edge_data_semantic = cv2.resize(edge_data_semantic, (W, H), interpolation=cv2.INTER_AREA)
        
        edge = self.config['cam']['crop_edge']
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
            edge_data_semantic = edge_data_semantic[edge:-edge, edge:-edge]

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))
        #edge_data = torch.from_numpy(edge_data.astype(np.float32))
        edge_data_semantic = torch.from_numpy(edge_data_semantic.astype(np.float32))
        border_data = create_border_data(depth_data)

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w":  self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "edge_semantic": edge_data_semantic,
            "border": border_data,
            "direction": self.rays_d
        }
        ret = self._attach_dino(ret, index, edge)
        ret = self._attach_deform(ret, index)
        ret = self._attach_seg(ret, index, H, W, edge)

        return ret

    def load_poses(self, basedir):
        """Load poses for StereoMIS dataset.

        self.poses: identity poses used for tracking initialization (first frame
        pose and constant velocity model). DDS-SLAM estimates all poses from
        rendering loss — it does NOT use GT poses during tracking.

        self.gt_poses: real GT poses from groundtruth.txt (TUM format) used ONLY
        for ATE evaluation after the run completes. If groundtruth.txt is not
        found, gt_poses falls back to the same identity poses.
        """
        # Identity poses for tracking (matches original DDS-SLAM code)
        self.poses = []
        for i in range(len(self.img_files)):
            c2w = np.eye(4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w[:3, 3] *= self.sc_factor
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

        # Load GT poses for evaluation
        gt_file = os.path.join(basedir, 'groundtruth.txt')
        self.gt_poses = None

        if os.path.isfile(gt_file):
            data = np.loadtxt(gt_file, comments='#')
            if data.ndim == 1:
                data = data.reshape(1, -1)
            print(f"Loaded {len(data)} GT poses from {gt_file}")
            self.gt_poses = []

            for img_path in self.img_files:
                basename = os.path.splitext(os.path.basename(img_path))[0]
                frame_num = int(re.sub(r'[^0-9]', '', basename))
                gt_idx = frame_num - 1  # filenames are 1-indexed

                if 0 <= gt_idx < len(data):
                    tx, ty, tz = data[gt_idx, 1:4]
                    qx, qy, qz, qw = data[gt_idx, 4:8]
                    r = Rotation.from_quat([qx, qy, qz, qw])
                    c2w = np.eye(4, dtype=np.float32)
                    c2w[:3, :3] = r.as_matrix()
                    c2w[:3, 3] = [tx, ty, tz]
                    c2w[:3, 3] *= self.sc_factor
                else:
                    c2w = np.eye(4, dtype=np.float32)
                    c2w[:3, 1] *= -1
                    c2w[:3, 2] *= -1
                c2w = torch.from_numpy(c2w).float()
                self.gt_poses.append(c2w)

            first_frame = int(re.sub(r'[^0-9]', '', os.path.basename(self.img_files[0])))
            last_frame = int(re.sub(r'[^0-9]', '', os.path.basename(self.img_files[-1])))
            print(f"GT poses loaded for frames {first_frame}-{last_frame} (for evaluation only)")
        else:
            print(f"No groundtruth.txt found, ATE evaluation will use identity poses")

class SuperDataset(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1, 
                 downsample_factor=1, translation=0.0, 
                 sc_factor=1., crop=0):
        super(SuperDataset, self).__init__(cfg)

        self.config = cfg
        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop

        self.img_files = sorted(glob.glob(f'{self.basedir}/rgb/*left.png'))
        # depth_subdir lets us A/B different depth sources (Monodepth2 mono /
        # stereo / AF-SfM / etc.) without duplicating rgb+seg. Defaults to 'rgb'
        # for back-compat with the upstream Semantic-SuPer layout where depth
        # .npys live alongside the RGB pngs.
        depth_subdir = cfg.get('data', {}).get('depth_subdir', 'rgb')
        self.depth_paths = sorted(
            glob.glob(f'{self.basedir}/{depth_subdir}/*left_depth.npy'))
        self.semantic_paths=sorted(glob.glob(f'{self.basedir}/seg/png_masks/*left.png'))

        # Inc-1 v2 (mode:'dino'): per-frame DINO grids (SemSup; NO [-4000:] slice, unlike StereoMIS).
        # None => no 'dino' key => base/geo bit-identical.
        self.dino_paths = None
        _unc = self.config.get('uncertainty', {})
        if _unc.get('enable', False) and _unc.get('mode', 'geo') == 'dino':
            _sub = _unc.get('dino_subdir', 'dino')
            self.dino_paths = sorted(glob.glob(f'{self.basedir}/{_sub}/*_dino.npy'))
            assert len(self.dino_paths) == len(self.img_files), \
                f"DINO features {len(self.dino_paths)} != frames {len(self.img_files)} in {self.basedir}/{_sub}"

        # ARM-2 Stage-1: baked Δx* deformation targets (deformation_sup_weight>0). NO [-4000:] slice
        # (SemSup, like dino above). None => no key => base bit-identical.
        self.deform_paths = None
        if self.config['training'].get('deformation_sup_weight', 0) > 0:
            _dsub = self.config.get('data', {}).get('deform_subdir', 'deform')
            self.deform_paths = sorted(glob.glob(f'{self.basedir}/{_dsub}/*_deform.npz'))
            assert len(self.deform_paths) == len(self.img_files), \
                f"deform targets {len(self.deform_paths)} != frames {len(self.img_files)} in {self.basedir}/{_dsub}"
            print(f'[teacher] deform targets ATTACHED: {len(self.deform_paths)} from {self.basedir}/{_dsub} '
                  f'(deformation_sup_weight={self.config["training"]["deformation_sup_weight"]})')

        # ARM-1 #4: per-frame RAW seg masks for the what-kind CE prior (whatkind_weight>0). SemSup is
        # 1:1 (one mask per frame). Read via cv2.IMREAD_UNCHANGED in _attach_seg. None => no 'seg' key
        # => base/per-pixel-dino bit-identical.
        self.seg_label_paths = None
        if self.config.get('uncertainty', {}).get('whatkind_weight', 0) > 0 or self.config.get('training', {}).get('tool_mask', False) or self.config.get('training', {}).get('tool_mask_track', False):
            self.seg_label_paths = [self.semantic_paths[min(i, len(self.semantic_paths) - 1)]
                                    for i in range(len(self.img_files))]

        self.load_poses(os.path.join(self.basedir, 'pose'))
        
        self.rays_d = None
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)

        if self.config['cam']['crop_edge'] > 0:
            self.H -= self.config['cam']['crop_edge']*2
            self.W -= self.config['cam']['crop_edge']*2
            self.cx -= self.config['cam']['crop_edge']
            self.cy -= self.config['cam']['crop_edge']
   
    def __len__(self):
        return self.num_frames
  
    def __getitem__(self, index):
        color_path = self.img_files[index]
        depth_path = self.depth_paths[index]
        semantic_path = self.semantic_paths[index]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth_data0 = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        elif '.npy' in depth_path:
            depth_data = np.load(depth_path)
            depth_data = depth_data.reshape(depth_data.shape[-2], depth_data.shape[-1])

            depth_data0 = depth_data
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            raise NotImplementedError()

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))
        # instance_data = cv2.resize(instance_data, (W, H))
        if '.png' in semantic_path:
            semantic_data = cv2.imread(semantic_path)
            semantic_data = cv2.resize(semantic_data, (W, H))
        elif '.npy' in semantic_path:
            semantic_data = np.load(semantic_path)#.reshape(3,semantic_data.shape[-2], semantic_data.shape[-1])
            semantic_data = semantic_data.reshape(semantic_data.shape[-2], semantic_data.shape[-1],3)
            print(semantic_data)
            semantic_data = cv2.cvtColor(semantic_data, cv2.COLOR_BGR2GRAY)

        edge_data_semantic = compute_edge_semantic(semantic_data, depth_data0)


        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor


        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            self.fx = self.fx // self.downsample_factor
            self.fy = self.fy // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)
            edge_data_semantic = cv2.resize(edge_data_semantic, (W, H), interpolation=cv2.INTER_AREA)
        
        edge = self.config['cam']['crop_edge']
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
            edge_data_semantic = edge_data_semantic[edge:-edge, edge:-edge]

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))
        edge_data_semantic = torch.from_numpy(edge_data_semantic.astype(np.float32))
        border_data = create_border_data(depth_data)

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w":  self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "edge_semantic": edge_data_semantic,
            "border": border_data,
            "direction": self.rays_d
        }
        ret = self._attach_dino(ret, index, edge)
        ret = self._attach_deform(ret, index)
        ret = self._attach_seg(ret, index, H, W, edge)

        return ret

    def load_poses(self, path):
        self.poses = []
        for i in range(len(self.img_files)):
            c2w=np.eye(4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w[:3, 3] *= self.sc_factor
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)
