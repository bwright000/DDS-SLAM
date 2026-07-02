import os
import sys
#os.environ['TCNN_CUDA_ARCHITECTURES'] = '86'
#import wandb
# Package imports
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse
import shutil
import json
import cv2

from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
import config
from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from datasets.dataset import get_dataset
from utils import coordinates, extract_mesh, colormap_image
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion

# Debug logger lives at Addons/diagnostics/ — add to path before import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Addons', 'diagnostics'))
from debug_logger import DebugLogger

import matplotlib.pyplot as plt

save_rendering_result=True


def sample_dino_grid(grid, hh, ww, H, W):
    """Inc-1 v2 (WildGS-faithful): bilinear-sample a COMPACT DINO patch-grid [gh,gw,C] at pixel
    coords (hh,ww) -> per-ray feature [N,C]. Mirrors WildGS's store-grid-then-upsample, but samples
    ONLY at the rays we use (~13ms vs a 944ms/472MB full-frame upsample). hh/ww are pixel rows/cols
    in [0,H)x[0,W); the grid covers the full image FOV. Returns on the grid's device (CPU here)."""
    gh, gw, C = grid.shape
    g = grid.float().permute(2, 0, 1).unsqueeze(0)                  # [1,C,gh,gw]
    hh = torch.as_tensor(hh, dtype=torch.float32)
    ww = torch.as_tensor(ww, dtype=torch.float32)
    gy = (hh / max(H - 1, 1)) * 2 - 1
    gx = (ww / max(W - 1, 1)) * 2 - 1
    coords = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)        # [1,N,1,2] (grid_sample x=cols,y=rows)
    out = F.grid_sample(g, coords, mode='bilinear', align_corners=True)  # [1,C,N,1]
    return out.squeeze(0).squeeze(-1).permute(1, 0).contiguous()    # [N,C]


class DDSSLAM():
    def __init__(self, config):
        self.config = config
        # Determinism fix (2026-06-14): seed_everything was DEFINED but NEVER CALLED -> every run was
        # nondeterministic, which on the bistable/underdetermined deformation field flipped DEAD<->LIVE
        # across runs of the SAME config (battery-5 LIVE 0.12 vs battery-6 DEAD 5.9e-9). Seed it.
        # (NOTE: reproducible != robust; robustness still needs the well-posing redesign below.)
        self.seed_everything(int(self.config.get('seed', 0)))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        
        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.keyframeDatabase = self.create_kf_database(config)
        self.model = JointEncoding(config, self.bounding_box).to(self.device)

        # ARM-2 Stage-2: causal teacher REPLAY buffer (one entry per processed frame: surface pts Xk + time
        # + baked Δx*/trust). Online by construction -- only frames already seen. Fixes the frame-k-only
        # teacher's recency-bias/time-agnostic collapse. field_optimizer set in create_optimizer.
        self.deform_replay = []
        self.field_optimizer = None

        # flow-as-sensor (flow_track.enable): load RAFT ONCE, in fork_rng so the model RNG above is
        # UNTOUCHED (the flow arm's init == base's init -> clean A/B); OFF => never imported/loaded =>
        # base byte-identical. RAFT lives here (orchestrator), NOT in JointEncoding -> the parity gate
        # (which snapshots the model) is unaffected.
        self.flow_track_on = bool(self.config.get('flow_track', {}).get('enable', False))
        self._flow_buf = None
        self._trust_map = None   # [H,W] depth-supervisor per-pixel trust weight (for the trust/ diagnostic dump)
        if self.flow_track_on:
            from collections import deque
            from Addons.motion.flow_track import load_raft
            _ft = self.config['flow_track']
            self._dino = None
            with torch.random.fork_rng(devices=(list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [])):
                self._raft, self._raft_tf = load_raft(self.device, bool(_ft.get('raft_small', False)))
                if _ft.get('agreement', False) or _ft.get('residual', 'sampson') == 'rigid' or _ft.get('mode', '') in ('depth_pool', 'solve_pnp'):   # agreement gate, L0 region-pool, and the new depth_pool/solve_pnp modes all need DINO (fork_rng -> parity-safe)
                    from Addons.motion.flow_track import load_dino
                    self._dino = load_dino(self.device)
            self._flow_buf = deque(maxlen=int(_ft.get('ref_stride', 8)))
            self._gate_fixed_pose = {}   # frame_id -> the pose the gate FROZE (for freeze_ba: re-apply after BA)
            print(f"[flow_track] ON: RAFT-{'small' if _ft.get('raft_small') else 'large'} "
                  f"ref_stride={_ft.get('ref_stride', 8)} gate={_ft.get('gate', False)} "
                  f"agreement={_ft.get('agreement', False)} freeze_ba={_ft.get('freeze_ba', False)}")

        # E0 MAPPING-ROUTING (flow-as-sensor, the INVERSE of tracking): the same per-region camera-vs-scene
        # signal, used to ROUTE the deformation field in the MAPPER -- the field warps moving (tissue) regions
        # and is OFF on static/camera regions (-> they stay sharp). Independent of flow_track (the tracking
        # gate); loads its OWN RAFT/DINO inside fork_rng (parity-safe) unless flow_track already loaded them.
        # Default-off -> base byte-identical.
        self.map_route_on = bool(self.config.get('map_route', {}).get('enable', False))
        self._map_route_buf = None
        self._route_map = None
        if self.map_route_on:
            from collections import deque
            from Addons.motion.flow_track import load_raft, load_dino
            _mr = self.config['map_route']
            with torch.random.fork_rng(devices=(list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [])):
                if getattr(self, '_raft', None) is None:
                    self._raft, self._raft_tf = load_raft(self.device, bool(_mr.get('raft_small', False)))
                if getattr(self, '_dino', None) is None:
                    self._dino = load_dino(self.device)
            self._map_route_buf = deque(maxlen=int(_mr.get('ref_stride', 8)))
            print(f"[map_route] ON: field routed to moving regions in current_frame_mapping+render "
                  f"ref_stride={_mr.get('ref_stride', 8)} deadband={_mr.get('deadband', 3.0)} n_groups={_mr.get('n_groups', 12)}")

        _dbg_dir = os.path.join(config['data']['output'], config['data']['exp_name'], 'debug')
        self.debug_logger = DebugLogger(_dbg_dir)

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
    def get_pose_representation(self):
        '''
        Get the pose representation axis-angle or quaternion
        '''
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
            print('Using axis-angle as rotation representation, identity init would cause inf')
        
        elif self.config['training']['rot_rep'] == "quat":
            print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError
        
    def create_pose_data(self):
        '''
        Create the pose data
        '''
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
        self.load_gt_pose() 
    
    def create_bounds(self):
        '''
        Get the pre-defined bounds for the scene
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(self.device)

    def create_kf_database(self, config):
        '''
        Create the keyframe database
        '''
        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)  
        print('#kf:', num_kf)
        print('#Pixels to save:', self.dataset.num_rays_to_save)
        return KeyFrameDatabase(config, 
                                self.dataset.H, 
                                self.dataset.W, 
                                num_kf, 
                                self.dataset.num_rays_to_save, 
                                self.device)
    
    def load_gt_pose(self):
        '''
        Load ground truth poses for evaluation.
        - pose_gt: real GT from groundtruth.txt if available, else identity
        - pose_gt_identity: always identity (matches paper's evaluation method)
        '''
        # Real GT (from groundtruth.txt or identity fallback)
        self.pose_gt = {}
        gt_source = getattr(self.dataset, 'gt_poses', None) or self.dataset.poses
        for i, pose in enumerate(gt_source):
            self.pose_gt[i] = pose

        # Identity GT (paper's method — measures drift from starting pose)
        self.pose_gt_identity = {}
        for i, pose in enumerate(self.dataset.poses):
            self.pose_gt_identity[i] = pose
 
    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
    
    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
                     'model': self.model.state_dict()}
        torch.save(save_dict, save_path)
        print('Save the checkpoint')

    def load_ckpt(self, load_path):
        '''
        Load the model parameters and the estimated pose
        '''
        dict = torch.load(load_path)
        self.model.load_state_dict(dict['model'])
        self.est_c2w_data = dict['pose']
        self.est_c2w_data_rel = dict['pose_rel']

    def select_samples(self, H, W, samples):
        '''
        randomly select samples from the image
        '''
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)
        return indice

    def _snapshot_ret(self, ret):
        '''
        Extract scalar diagnostics from a tracking-iteration `ret` dict so we
        can keep them around without pinning the autograd graph. Used to log
        the BEST-iter loss components (matches the pose stored in est_c2w_data).
        '''
        def _f(x):
            if x is None:
                return None
            try:
                return float(x.item())
            except AttributeError:
                return float(x)
        snap = {
            'rgb_loss':           _f(ret.get('rgb_loss')),
            'depth_loss':         _f(ret.get('depth_loss')),
            'sdf_loss':           _f(ret.get('sdf_loss')),
            'fs_loss':            _f(ret.get('fs_loss')),
            'edge_semantic_loss': _f(ret.get('edge_semantic_loss')),
            'psnr':               _f(ret.get('psnr')),
            'sdf_stats':          dict(ret['sdf_stats']) if ret.get('sdf_stats') else None,
        }
        return snap

    def get_loss_from_ret(
            self,
            ret,
            rgb=True,
            sdf=True,
            depth=True,
            fs=True,
            edge=False,
            edge_semantic=True,
            smooth=False,
    ):
        '''
        Get the training loss
        '''
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
        if fs:
            loss +=  self.config['training']['fs_weight'] * ret["fs_loss"]
        if edge:
            loss += self.config['training']['rgb_weight'] * 0.5 * ret["edge_loss"]
        if edge_semantic:
            loss += self.config['training']['rgb_weight'] * 0.1 * ret["edge_semantic_loss"]

        # Deformation-magnitude regularizer (||dx||^2) — wires the missing stabilizer the
        # authors named (time_smoothness_weight etc.) but never connected. Default 0 = off
        # (regression-safe). With wd=0 on time_net, a swept weight should settle the field
        # at a stable, useful nonzero deformation (vs denormal collapse at wd=1e-6 / divergence at wd=0).
        if self.config['training'].get('deformation_reg_weight', 0) > 0 and ret.get('def_reg') is not None:
            loss += self.config['training']['deformation_reg_weight'] * ret['def_reg']

        # ARM-2 Inc-1: self-supervised aleatoric NLL on the photometric residual,
        # trains the uncertainty head. Mirrors the def_reg optional-loss-behind-a-weight
        # pattern above. nll_weight default 0 (+ ret['nll'] absent when the head is off)
        # => zero contribution => regression-safe / bit-identical to base.
        _nll_w = self.config.get('uncertainty', {}).get('nll_weight', 0)
        if _nll_w > 0 and ret.get('nll') is not None:
            loss += _nll_w * ret['nll']

        # ARM-1 #4: what-kind seg-supervised attribution CE loss (region-coherent slot head).
        # Mirrors the NLL optional-loss-behind-a-weight pattern. whatkind_weight default 0
        # (+ ret['whatkind_loss'] absent unless the slot module is on AND target_seg supplied)
        # => zero contribution => regression-safe / bit-identical to base.
        _wk_w = self.config.get('uncertainty', {}).get('whatkind_weight', 0)
        if _wk_w > 0 and ret.get('whatkind_loss') is not None:
            loss += _wk_w * ret['whatkind_loss']

        if smooth and self.config['training']['smooth_weight']>0:
            loss += self.config['training']['smooth_weight'] * self.smoothness(self.config['training']['smooth_pts'],
                                                                                  self.config['training']['smooth_vox'],
                                                                                  margin=self.config['training']['smooth_margin'])
        return loss             

    def first_frame_mapping(self, batch, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        print('First frame mapping...')
        c2w = batch['c2w'][0].to(self.device)
        self.est_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w

        self.model.train()

        # Training
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])
            
            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_edge_semantic = batch['edge_semantic'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)
            # Inc-1 v2: per-ray DINO feature [N,C] (None unless mode:'dino' -> forward stays base/geo).
            target_dino = sample_dino_grid(batch['dino_grid'].squeeze(0), indice_h, indice_w, self.dataset.H, self.dataset.W).to(self.device) if 'dino_grid' in batch else None
            # ARM-1 #4: per-ray seg class id [N] at the SAME indices as target_dino (None unless
            # whatkind_weight>0 -> the what-kind CE prior; absent => base/per-pixel-dino unchanged).
            target_seg = batch['seg'].squeeze(0)[indice_h, indice_w].to(self.device) if 'seg' in batch else None

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)
            if self.config['dynamic']:
                cur_id =  0 * torch.ones(rays_o.shape[0])
                timestamps = (cur_id.to(self.device) / self.dataset.num_frames) if self.config['training'].get('time_normalize', False) else cur_id.to(self.device)  # T1.2: normalise frame_time to [0,1] (else freq-encoder parity-collapse); flag default off = upstream behaviour
                rays_o = torch.cat([rays_o,timestamps.unsqueeze(-1)],dim=1)
            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d, target_edge_semantic=target_edge_semantic, target_dino=target_dino, target_seg=target_seg, notFirstMap=False)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.map_optimizer.step()
        
        # First frame will always be a keyframe
        self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
        print('First frame mapping done')

        # Debug log — frame 0 (no tracking iters, use last mapping ret)
        try:
            _dvalid = ((batch['depth'] > 0) & (batch['depth'] < self.config['cam']['depth_trunc'])).float().mean().item()
            _dmean = batch['depth'][batch['depth'] > 0].mean().item() if (batch['depth'] > 0).any() else 0.0
            snap = self._snapshot_ret(ret)
            self.debug_logger.log_tracking(
                frame_id=0,
                est_c2w=self.est_c2w_data[0],
                gt_c2w=self.pose_gt[0],
                is_keyframe=True,
                init_c2w=self.est_c2w_data[0],
                tracking_iters_used=n_iters,
                tracking_iters_config=n_iters,
                best_loss=loss.item(),
                last_loss=loss.item(),
                loss_components={
                    'rgb': snap.get('rgb_loss'),
                    'depth': snap.get('depth_loss'),
                    'sdf': snap.get('sdf_loss'),
                    'fs': snap.get('fs_loss'),
                    'edge_semantic': snap.get('edge_semantic_loss'),
                },
                psnr=snap.get('psnr'),
                depth_valid_frac=_dvalid,
                depth_mean=_dmean,
                rgb_mean=batch['rgb'].mean().item(),
                sdf_stats=snap.get('sdf_stats'),
            )
        except Exception as _e:
            print(f'[DebugLogger] first-frame log failed: {_e}')

        return ret, loss

    def current_frame_mapping(self, batch, cur_frame_id):
        '''
        Current frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        if self.config['mapping']['cur_frame_iters'] <= 0:
            return
        print('Current frame mapping...')
        
        c2w = self.est_c2w_data[cur_frame_id].to(self.device)

        self.model.train()
        cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(c2w[None, ...], mapping=True)
        # DEFORM-SCALED map optimisation (the AMOUNT lever, made adaptive): the proven mapping lever is the NUMBER
        # of map iters (curmap 0->100 = +4 PSNR) -- spend MORE of it on DEFORMING frames (high moving-frac) and the
        # base on static ones that are already mapped. n_iters = cur_frame_iters + deform_iters_scale * frame
        # moving-frac (from the causal route, ready at run() line ~1132). Default scale 0 => n_iters = base =>
        # byte-identical. Unlike the dead map-loss RE-weights, this scales the optimisation AMOUNT, not its weight.
        _n_iters = self.config['mapping']['cur_frame_iters']
        _dis = self.config['mapping'].get('deform_iters_scale', 0)
        if _dis > 0 and getattr(self, '_route_map', None) is not None:
            _n_iters = _n_iters + int(_dis * float(self._route_map.mean()))
        # Training
        for i in range(_n_iters):
            pose_optimizer.zero_grad()
            self.cur_map_optimizer.zero_grad()
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)
            # chinaxiv gradient loss needs SPATIAL neighbours -> sample k×k PATCHES (in patch order) so the render
            # can reshape to [P,k,k,3] for the gradient term. Default off (gradloss_weight 0) => scattered = base.
            if self.config['training'].get('gradloss_weight', 0.0) > 0:
                _k = int(self.config['training'].get('grad_patch', 8))
                _P = max(1, self.config['mapping']['sample'] // (_k * _k))
                _th = torch.randint(0, self.dataset.H - _k, (_P,)); _tw = torch.randint(0, self.dataset.W - _k, (_P,))
                _dh = torch.arange(_k).view(1, _k, 1); _dw = torch.arange(_k).view(1, 1, _k)
                indice_h = (_th.view(_P, 1, 1) + _dh).expand(_P, _k, _k).reshape(-1)
                indice_w = (_tw.view(_P, 1, 1) + _dw).expand(_P, _k, _k).reshape(-1)
            else:
                indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])
                indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            # TOOL BINARY MASK: drop tool pixels (canonical seg==2) from this current-frame map update -- the
            # DOMINANT mapper under curmap100 (~100 iters/frame vs global_BA ~4) -- so the static map is not
            # corrupted by the moving instrument; the tool region then renders the tissue behind it, fused from
            # tool-free frames. Default off (tool_mask flag off OR no 'seg' key) => base byte-identical.
            if self.config['training'].get('tool_mask', False) and 'seg' in batch:
                _keep = (batch['seg'].squeeze(0)[indice_h, indice_w] != 2)
                if 0 < int(_keep.sum()) < _keep.numel():
                    indice_h, indice_w = indice_h[_keep], indice_w[_keep]
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_edge_semantic = batch['edge_semantic'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)
            # Inc-1 v2: per-ray DINO feature [N,C] (None unless mode:'dino').
            target_dino = sample_dino_grid(batch['dino_grid'].squeeze(0), indice_h, indice_w, self.dataset.H, self.dataset.W).to(self.device) if 'dino_grid' in batch else None
            # ARM-1 #4: per-ray seg class id [N] at the SAME indices as target_dino (None unless whatkind_weight>0).
            target_seg = batch['seg'].squeeze(0)[indice_h, indice_w].to(self.device) if 'seg' in batch else None
            # E0 mapping-routing: per-ray field-route weight at these pixels (1=moving->field ON, 0=static->OFF).
            # Gates the field's Δx in this map forward so the map co-adapts to a TISSUE-ONLY warp (bg stays
            # sharp). None unless map_route.enable AND the causal buffer filled -> unrouted (base behaviour).
            route_w = self._route_map[indice_h, indice_w].view(-1, 1) if getattr(self, '_route_map', None) is not None else None
            # inverse-flow MAP UP-WEIGHT: more DINO-flow residual (route_w high = deforming) -> up-weight THIS
            # current-frame map's RGB+depth loss (1 + alpha*route) so it captures the deforming tissue -- the
            # MIRROR of the tracking down-weight. None unless map_upweight>0 + the route exists -> base unchanged.
            _uw = self.config['training'].get('map_upweight', 0.0)
            map_ray_w = (1.0 + _uw * route_w).detach() if (_uw > 0 and route_w is not None) else None
            # ARM-2 Stage-1: per-ray baked deformation target Δx* [N,3] + trust [N,1] (None unless deformation_sup_weight>0).
            if 'deform_dx' in batch:
                deform_dx = sample_dino_grid(batch['deform_dx'].squeeze(0), indice_h, indice_w, self.dataset.H, self.dataset.W).to(self.device)
                deform_w  = sample_dino_grid(batch['deform_trust'].squeeze(0), indice_h, indice_w, self.dataset.H, self.dataset.W).to(self.device)
            else:
                deform_dx = deform_w = None

            rays_o = c2w_est[..., :3, -1].repeat(rays_d_cam.shape[0], 1)   # ACTUAL ray count -- tool_mask may have dropped tool px (1708<2048); == mapping.sample when unfiltered => base byte-identical
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[: ,:3, :3], -1)
            if self.config['dynamic']:
                cur_id = (cur_frame_id*torch.ones(rays_o.shape[0]))
                timestamps = (cur_id.to(self.device) / self.dataset.num_frames) if self.config['training'].get('time_normalize', False) else cur_id.to(self.device)  # T1.2: normalise frame_time to [0,1]; flag default off = upstream behaviour
                rays_o = torch.cat([rays_o,timestamps.unsqueeze(-1)],dim=1)
            # ARM-2 Stage-1 teacher. deform_teacher_only -> the field is trained on the teacher ALONE, so the
            # render forward is UNUSED: skip it entirely (big T4 memory + ~2x compute saving; fixes the
            # end-of-run OOM where the wasted forward tipped the GPU over on the last frame).
            _ds_w = self.config['training'].get('deformation_sup_weight', 0)
            # cur_frame_map_only: render-ONLY sharpening (no teacher loss here; the field is trained by the
            # replay and is excluded from cur_map_optimizer). Lets the map sharpen WITHOUT touching the field.
            _cfmo = self.config['training'].get('cur_frame_map_only', False)
            _teach = _ds_w > 0 and self.config['dynamic'] and deform_dx is not None and not _cfmo
            if _teach and self.config['training'].get('deform_teacher_only', False):
                Xk = rays_o[..., :3] + rays_d * target_d                       # [N,3] world surface pts
                def_sup = self.model.deform_teacher_loss(Xk, timestamps.unsqueeze(-1), deform_dx, deform_w)
                loss = _ds_w * def_sup
                ret = None                                                     # render forward skipped (teacher-only)
            else:
                ret = self.model.forward(rays_o, rays_d, target_s, target_d, target_edge_semantic=target_edge_semantic, target_dino=target_dino, target_seg=target_seg, route_w=route_w, map_ray_w=map_ray_w)
                loss = self.get_loss_from_ret(ret)
                if _teach:                                                     # joint: render loss + teacher
                    Xk = rays_o[..., :3] + rays_d * target_d
                    def_sup = self.model.deform_teacher_loss(Xk, timestamps.unsqueeze(-1), deform_dx, deform_w)
                    loss = loss + _ds_w * def_sup
            # diagnostics (frames<=3): does def_sup DROP it0->it99 (field learning)? / are targets attached?
            if _teach and cur_frame_id <= 3 and (i == 0 or i == self.config['mapping']['cur_frame_iters'] - 1):
                print(f'[teacher] frame {cur_frame_id} it{i:3d}: def_sup={def_sup.item():.7f}  weighted={_ds_w*def_sup.item():.5f}  |dx*|mean={deform_dx.norm(dim=-1).mean().item():.5f}  w.sum={deform_w.sum().item():.1f}')
            elif _ds_w > 0 and self.config['dynamic'] and deform_dx is None and i == 0 and cur_frame_id <= 3:
                print(f'[teacher] frame {cur_frame_id}: deform_dx is None (targets NOT attached) -> teacher INACTIVE')
            loss.backward()
            self.cur_map_optimizer.step()
        return ret, loss

    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points-1) * voxel_size
        offset_max = self.bounding_box[:, 1]-self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        if self.config['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        

        sdf = self.model.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:,...]-sdf[:-1,...], 2).sum()
        tv_y = torch.pow(sdf[:,1:,...]-sdf[:,:-1,...], 2).sum()
        tv_z = torch.pow(sdf[:,:,1:,...]-sdf[:,:,:-1,...], 2).sum()

        loss = (tv_x + tv_y + tv_z)/ (sample_points**3)

        return loss
    
    def get_pose_param_optim(self, poses, mapping=True):
        task = 'mapping' if mapping else 'tracking'
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config[task]['lr_rot']},
                                               {"params": cur_trans, "lr": self.config[task]['lr_trans']}])
        
        return cur_rot, cur_trans, pose_optimizer
    
    def global_BA(self, batch, cur_frame_id):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''
        pose_optimizer = None

        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])
        
        # frame ids for all KFs, used for update poses after optimization
        frame_ids_all = torch.tensor(list(range(0, cur_frame_id, self.config['mapping']['keyframe_every'])))

        if len(self.keyframeDatabase.frame_ids) < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)
        
        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]

            if self.config['mapping']['optim_cur']:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

            else:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(poses[1:])
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)
        
        # Set up optimizer
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()

        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None], batch['edge_semantic'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])

        

        for i in range(self.config['mapping']['iters']):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            _route_ba = self.config.get('map_route', {}).get('route_ba', False) and getattr(self, '_route_map', None) is not None
            if _route_ba:
                rays, ids, kf_route = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'], with_route=True)
            else:
                rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])

            #TODO: Checkpoint...
            idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W),max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.config['mapping']['min_pixels_cur']))
            current_rays_batch = current_rays[idx_cur, :]
            rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
            ids_all = torch.cat([ids//self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(torch.int64)
            # B2: route global_BA's KEYFRAME rays (the dominant map trainer) so the map co-adapts to the
            # routed warp = the global-render-confound fix. route_w_ba = [stored per-keyframe routes ;
            # current-frame route from self._route_map]. None unless route_ba -> forward route_w=None = base.
            route_w_ba = None
            if _route_ba:
                _idxc = torch.as_tensor(idx_cur, device=self.device)
                route_w_ba = torch.cat([kf_route.view(-1).to(self.device),
                                        self._route_map.reshape(-1)[_idxc]]).view(-1, 1)


            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)
            target_edge_semantic = rays[..., 7:8].to(self.device)
            # Inc-1 v2 (WildGS-faithful): global_BA is DINO-FREE -- the head trains via per-frame
            # current_frame mapping, not the keyframe DB. rays stay width 8, so no sigma2 here.
            target_dino = None

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            if self.config['dynamic']:
                cur_id = (cur_frame_id*torch.ones(current_rays_batch.shape[0]))
                timestamps = torch.cat([ids,cur_id],dim=0).to(self.device)
                # --- global_BA TIME-CONVENTION FIX (audit wmkk74lpr). global_BA is the DOMINANT field
                # trainer (mapping.iters per frame) yet it fed RAW integer frame_time, while tracking
                # (:580) / first-frame-map / per-frame render normalize by num_frames when
                # time_normalize:true. embed_time is a freq (sin/cos) encoder, so raw-int vs normalized
                # hit UNRELATED phases -> TimeNet was trained on TWO conflicting time axes = a candidate
                # cause of the dead field. Default OFF = legacy raw behaviour (regression-safe). With
                # global_ba_time_fix:true AND time_normalize:true, normalize here too so ALL field-training
                # paths share ONE time axis (mirrors the tracking normalization at :580).
                if self.config['training'].get('time_normalize', False) and self.config['training'].get('global_ba_time_fix', False):
                    timestamps = timestamps / self.dataset.num_frames
                rays_o = torch.cat([rays_o,timestamps.unsqueeze(-1)],dim=1)

            # MAP-ROUTE direction = the moving-region question (two OPPOSITE hypotheses, decided by metric):
            #   protect>0  -> EXCLUDE moving rays (route>=thr) from the static-map update. The dead static map
            #                 can't represent deforming tissue and only gets CORRUPTED fitting it -> route what
            #                 "should go to the field" OUT of the static map; leave it to the (wired) field.
            #   attend>0   -> OVER-sample moving rays (duplicate) -> "pay SPECIAL attention to the moving parts"
            #                 (user's intuition; correct IF a model can fit them).
            # Sample-level (losses come back meaned, no per-ray reweight). Default both 0 = base byte-identical.
            # Guard: protect never drops ALL rays.
            _pf = self.config.get('map_route', {}).get('protect', 0.0)
            _af = self.config.get('map_route', {}).get('attend', 0.0)
            if _route_ba and route_w_ba is not None and (_pf > 0 or _af > 0):
                _r = route_w_ba.view(-1)
                if _pf > 0:
                    _keep = (_r < _pf)
                    if 0 < int(_keep.sum().item()) < _keep.numel():
                        rays_o, rays_d = rays_o[_keep], rays_d[_keep]
                        target_s, target_d, target_edge_semantic = target_s[_keep], target_d[_keep], target_edge_semantic[_keep]
                        route_w_ba = route_w_ba[_keep]
                elif _af > 0:
                    _mv = (_r >= _af)
                    if bool(_mv.any()):
                        rays_o = torch.cat([rays_o, rays_o[_mv]]); rays_d = torch.cat([rays_d, rays_d[_mv]])
                        target_s = torch.cat([target_s, target_s[_mv]]); target_d = torch.cat([target_d, target_d[_mv]])
                        target_edge_semantic = torch.cat([target_edge_semantic, target_edge_semantic[_mv]])
                        route_w_ba = torch.cat([route_w_ba, route_w_ba[_mv]])

            ret = self.model.forward(rays_o, rays_d, target_s, target_d, target_edge_semantic=target_edge_semantic, target_dino=target_dino, route_w=route_w_ba)

            loss = self.get_loss_from_ret(ret, smooth=True)
            
            loss.backward(retain_graph=True)
            
            if (i + 1) % cfg["mapping"]["map_accum_step"] == 0:
               
                if (i + 1) > cfg["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

            if pose_optimizer is not None and (i + 1) % cfg["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                # So current pose is always unchanged
                if self.config['mapping']['optim_cur']:
                    poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
                
                else:
                    current_pose = self.est_c2w_data[cur_frame_id][None,...]
                    # SE3 poses

                    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)


                # zero_grad here
                pose_optimizer.zero_grad()
        
        if pose_optimizer is not None and len(frame_ids_all) > 1:
            for i in range(len(frame_ids_all[1:])):
                self.est_c2w_data[int(frame_ids_all[i+1].item())] = self.matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
        
            if self.config['mapping']['optim_cur']:
                print('Update current pose')
                self.est_c2w_data[cur_frame_id] = self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]

        # flow_track freeze_ba: re-apply the gate's FIX to gate-fixed KEYFRAMES that BA just re-optimised,
        # so the still-window freeze persists (keyframes anchor the non-keyframes via the relative poses).
        if self.config.get('flow_track', {}).get('freeze_ba', False) and getattr(self, '_gate_fixed_pose', None):
            _ke = self.config['mapping']['keyframe_every']
            for _f, _p in self._gate_fixed_pose.items():
                if _f % _ke == 0 and _f <= cur_frame_id and _f in self.est_c2w_data:
                    self.est_c2w_data[_f] = _p.to(self.device)

    def predict_current_pose(self, frame_id, constant_speed=True):
        '''
        Predict current pose from previous pose using camera motion model
        '''
        if frame_id == 1 or (not constant_speed):
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            self.est_c2w_data[frame_id] = c2w_est_prev
            
        else:
            c2w_est_prev_prev = self.est_c2w_data[frame_id-2].to(self.device)
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            delta = c2w_est_prev@c2w_est_prev_prev.float().inverse()
            self.est_c2w_data[frame_id] = delta@c2w_est_prev
        
        return self.est_c2w_data[frame_id]

    def _rgb_to_bgr_u8(self, rgb):
        '''batch['rgb'] [1,H,W,3] float RGB [0,1] -> [H,W,3] uint8 BGR (for RAFT/cv2).'''
        a = (rgb.squeeze(0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)

    def _compute_route_map(self, batch, frame_id):
        '''E0: compute (CAUSALLY) the per-pixel field-routing map for `frame_id` -> self._route_map
        ([H,W] float in {0,1}: 1 = scene-moving -> field ON there, 0 = static/camera -> field OFF -> sharp).
        Uses a PAST reference (ref index < frame_id). None until the causal buffer fills (-> field un-routed,
        neutral). Runs ONCE per frame; the mapper / render / teacher-buffer all read the SAME self._route_map.'''
        from Addons.motion.flow_track import region_route, dino_grid
        _mr = self.config['map_route']
        cur_bgr = self._rgb_to_bgr_u8(batch['rgb'])
        ref = self._map_route_buf[0] if (len(self._map_route_buf) == self._map_route_buf.maxlen) else None
        self._map_route_buf.append((frame_id, cur_bgr))   # ref captured above; append for FUTURE frames
        if ref is None:
            self._route_map = None
            return
        ref_id, ref_bgr = ref
        if ref_id >= frame_id:   # CAUSALITY: raise (NOT assert -> not stripped by python -O)
            raise RuntimeError(f"map_route NON-CAUSAL: ref {ref_id} >= cur {frame_id}")
        _mode = _mr.get('mode', 'region')
        dino_g = dino_grid(cur_bgr, self._dino, self.device) if _mode == 'region' else None   # pixel mode is DINO-free
        route = region_route(ref_bgr, cur_bgr, dino_g, self._raft, self._raft_tf, self.device,
                             mode=_mode, n_groups=int(_mr.get('n_groups', 12)),
                             deadband=float(_mr.get('deadband', 1.0)), smooth=int(_mr.get('smooth', 5)),
                             soft_scale=float(_mr.get('soft_scale', 0.0)),
                             ransac_thresh=float(_mr.get('ransac_thresh', 1.0)))
        # EDGE-MASK (user clue: the route lights up the image BORDER). RAFT flow + the homography
        # extrapolate badly at the frame edge -> spurious high residual there. Zero a border band so the
        # field is never routed to the edges. edge_mask=0 -> off (default).
        _em = int(_mr.get('edge_mask', 0))
        if _em > 0:
            route[:_em, :] = 0.0; route[-_em:, :] = 0.0; route[:, :_em] = 0.0; route[:, -_em:] = 0.0
        self._route_map = torch.from_numpy(route).float().to(self.device)   # [H,W] field-route weight
        if frame_id <= 3 or frame_id % 30 == 0:
            print(f'[map_route] frame {frame_id}: moving-frac={float(route.mean()):.3f} (causal ref {ref_id})')

    def _flowlog(self, header, row):
        """Append one per-frame diagnostic row to output/trust_log.csv (header written once). Used by the
        flow_track modes (solve_pnp / depth_pool). Best-effort; never breaks tracking."""
        try:
            import csv as _csv
            _p = os.path.join(self.config['data']['output'], 'trust_log.csv')
            _new = not os.path.exists(_p)
            with open(_p, 'a', newline='') as _f:
                _w = _csv.writer(_f)
                if _new:
                    _w.writerow(header)
                _w.writerow(row)
        except Exception as _e:
            print(f"[flowlog] skipped: {_e}")

    def tracking_render(self, batch, frame_id):
        '''
        Tracking camera pose using of the current frame
        Params:
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            frame_id: Current frame id (int)
        '''

        c2w_gt = batch['c2w'][0].to(self.device)
        self._trust_map = None   # reset per frame; the depth supervisor (gate=false, residual=rigid) sets it below if it runs

        # Initialize current pose
        if self.config['tracking']['iter_point'] > 0:
            cur_c2w = self.est_c2w_data[frame_id]
        else:
            cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

        indice = None
        best_sdf_loss = None
        best_ret = None
        thresh=0

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        # flow-as-sensor: from a PAST reference (CAUSAL, ref index < frame_id). Two modes:
        #   gate -> if the CAMERA is ~still (|median flow| <= cam_thresh): FIX the pose (= previous frame)
        #           and SKIP tracking (don't drift while the camera isn't moving); the mapper still runs.
        #   else -> per-pixel down-weight (scene-moving rays trusted less in the pose solve).
        # Nothing runs unless flow_track.enable -> base byte-identical.
        track_w_map = None
        if getattr(self, 'flow_track_on', False):
            _ft = self.config['flow_track']
            cur_bgr = self._rgb_to_bgr_u8(batch['rgb'])
            _cur_depth = batch['depth'].squeeze(0).detach().cpu().numpy().astype(np.float32)   # buffered for the depth supervisor (ref-frame Z-depth)
            ref = self._flow_buf[0] if (len(self._flow_buf) == self._flow_buf.maxlen) else None
            self._flow_buf.append((frame_id, cur_bgr, _cur_depth))   # ref captured above; append for FUTURE frames
            if ref is not None:
                ref_id, ref_bgr, ref_depth = ref
                if ref_id >= frame_id:   # CAUSALITY: raise (NOT assert -> not stripped by python -O)
                    raise RuntimeError(f"flow_track NON-CAUSAL: ref {ref_id} >= cur {frame_id}")
                _mode = _ft.get('mode', '')
                if _mode == 'solve_pnp':
                    # MODE A: robust 2D-3D PnP camera-motion solve (replaces the F-gate). Ref depth only; tool
                    # HARD-excluded (seg==2); flow-floor gated -> None = fall back to const-velocity. T_rel used
                    # INIT-ONLY (SDF tracker can overrule, never freeze); reproj residual -> per-ray trust weight.
                    # Up-to-scale (relative depth) -> NOT a metric anchor (per the internal review).
                    from Addons.motion.flow_track import rigid_solve_pnp, region_soft_weight, dino_grid
                    _tool = (batch['seg'].squeeze(0).detach().cpu().numpy() == 2) if 'seg' in batch else None
                    _out = rigid_solve_pnp(ref_bgr, cur_bgr, ref_depth,
                                           float(self.dataset.fx), float(self.dataset.fy),
                                           float(self.dataset.cx), float(self.dataset.cy),
                                           self._raft, self._raft_tf, self.device, tool_mask=_tool,
                                           flow_advance_px=float(_ft.get('flow_advance_px', 1.5)),
                                           reproj_px=float(_ft.get('reproj_px', 2.0)),
                                           min_inliers=int(_ft.get('min_inliers', 200)))
                    _cols = ['frame', 'applied', 't_norm', 'rot_deg', 'inlier_frac', 'reproj_med', 'w_mean', 'w_min', 'frac_dn']
                    if _out is not None:
                        _Trel, _resid, _info = _out
                        # INIT-ONLY: seed the tracker pose at c2w_ref @ inv(T_rel). T_rel maps ref-cam(CV)->cur-cam(CV),
                        # so c2w_cur_cv = c2w_ref_cv @ inv(T_rel); poses are OpenGL so wrap in GL<->CV (diag(1,-1,-1,1)).
                        _GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
                        _cr = self.est_c2w_data[ref_id].detach().cpu().numpy().astype(np.float32)
                        _cur_gl = _cr @ _GL2CV @ np.linalg.inv(_Trel).astype(np.float32) @ _GL2CV
                        cur_c2w = torch.from_numpy(np.ascontiguousarray(_cur_gl, dtype=np.float32)).to(self.device)  # init-only override
                        _dg = dino_grid(cur_bgr, self._dino, self.device)
                        _wfull, _wm, _wa, _ws = region_soft_weight(
                            _resid, _dg, n_groups=int(_ft.get('n_groups', 12)),
                            mad_c=float(_ft.get('mad_c', 2.0)), w_floor_px=float(_ft.get('w_floor_px', 1.0)),
                            w_min=float(_ft.get('w_min', 0.1)))
                        if _tool is not None and bool(_tool.any()):
                            _wfull[_tool] = float(_ft.get('w_min', 0.1))   # tool excluded from the SOLVE -> also force it LOW-trust in the weight (its resid is 0 -> would otherwise give w=1, re-opening the hijack in refinement)
                        self._trust_map = _wfull
                        track_w_map = torch.from_numpy(_wfull[iH:-iH, iW:-iW])
                        _wc = _wfull[iH:-iH, iW:-iW]
                        print(f"[solve_pnp] f{frame_id}: |t|={_info['t_norm']:.4f} rot={_info['rot_deg']:.2f}deg "
                              f"inl={_info['inlier_frac']:.2f} reproj_med={_info['reproj_med']:.2f}px "
                              f"w_mean={float(_wc.mean()):.3f} frac_dn={float((_wc < 0.5).mean()):.3f} (ref {ref_id})")
                        self._flowlog(_cols, [frame_id, 1, round(_info['t_norm'], 5), round(_info['rot_deg'], 3),
                                      round(_info['inlier_frac'], 3), round(_info['reproj_med'], 3),
                                      round(float(_wc.mean()), 4), round(float(_wc.min()), 4), round(float((_wc < 0.5).mean()), 4)])
                    else:
                        print(f"[solve_pnp] f{frame_id}: below-floor / no-solve -> const-velocity fallback (ref {ref_id})")
                        self._flowlog(_cols, [frame_id, 0, '', '', '', '', '', '', ''])
                elif _mode == 'depth_pool':
                    # MODE B (pose-free): pool flow+depth into DINO regions, *depth to a common plane, threshold the
                    # deviation from the robust consensus. Always-on soft down-weight (no gate, no freeze).
                    from Addons.motion.flow_track import depth_pooled_weight, dino_grid
                    _dg = dino_grid(cur_bgr, self._dino, self.device)
                    _wfull, _med, _mad, _scale = depth_pooled_weight(
                        ref_bgr, cur_bgr, ref_depth, _dg, self._raft, self._raft_tf, self.device,
                        n_groups=int(_ft.get('n_groups', 12)), mad_c=float(_ft.get('mad_c', 2.0)),
                        w_floor_px=float(_ft.get('w_floor_px', 1.0)), w_min=float(_ft.get('w_min', 0.1)))
                    self._trust_map = _wfull
                    track_w_map = torch.from_numpy(_wfull[iH:-iH, iW:-iW])
                    _wc = _wfull[iH:-iH, iW:-iW]
                    print(f"[depth_pool] f{frame_id}: dev_med={_med:.3f} mad={_mad:.3f} scale={_scale:.3f} "
                          f"w_mean={float(_wc.mean()):.3f} frac_dn={float((_wc < 0.5).mean()):.3f} (ref {ref_id})")
                    self._flowlog(['frame', 'dev_med', 'dev_mad', 'scale', 'w_mean', 'w_min', 'frac_dn'],
                                  [frame_id, round(_med, 4), round(_mad, 4), round(_scale, 4),
                                   round(float(_wc.mean()), 4), round(float(_wc.min()), 4), round(float((_wc < 0.5).mean()), 4)])
                elif _ft.get('gate', False):
                    if _ft.get('agreement', False):
                        # PER-REGION AGREEMENT (the probe in the loop): pool flow into DINO regions, do the
                        # per-region motion vectors AGREE with one rigid motion? TRACK iff the consensus is
                        # moving AND the features agree (camera/rigid); else FIX (still OR scene-deforming).
                        from Addons.motion.flow_track import agreement_gate, dino_grid
                        _dg = dino_grid(cur_bgr, self._dino, self.device)
                        cam_mag, disagree = agreement_gate(
                            ref_bgr, cur_bgr, _dg, self._raft, self._raft_tf, self.device,
                            n_groups=int(_ft.get('n_groups', 12)), ransac_thresh=float(_ft.get('ransac_thresh', 1.0)),
                            deadband=float(_ft.get('deadband', 3.0)))
                        do_track = (cam_mag > float(_ft.get('cam_thresh', 2.0))) and (disagree <= float(_ft.get('disagree_thresh', 0.2)))
                        _reason = f"cam_mag {cam_mag:.2f} disagree {disagree:.2f}"
                    else:
                        from Addons.motion.flow_track import camera_motion
                        cam_mag = camera_motion(ref_bgr, cur_bgr, self._raft, self._raft_tf, self.device)
                        do_track = cam_mag > float(_ft.get('cam_thresh', 2.0))
                        _reason = f"cam_mag {cam_mag:.2f}"
                    if not do_track:
                        # FIX pose = previous frame, skip tracking (mapper still runs)
                        self.est_c2w_data[frame_id] = self.est_c2w_data[frame_id - 1].detach().clone()
                        self._gate_fixed_pose[frame_id] = self.est_c2w_data[frame_id].detach().clone()  # freeze_ba: re-applied after BA
                        if frame_id % self.config['mapping']['keyframe_every'] != 0:
                            _kf = (frame_id // self.config['mapping']['keyframe_every']) * self.config['mapping']['keyframe_every']
                            self.est_c2w_data_rel[frame_id] = self.est_c2w_data[frame_id] @ self.est_c2w_data[_kf].float().inverse()
                        print(f"[flow_gate] f{frame_id}: {_reason} -> FIX pose, skip tracking")
                        return
                    # else: camera moving + features agree -> fall through to normal tracking
                elif _ft.get('residual', 'sampson') == 'rigid':
                    # DEPTH SUPERVISOR (L0): depth-predicted rigid-flow residual -> per-DINO-region
                    # adaptive soft weight. No F-matrix, no deadband, no semantic labels. gate=false ->
                    # ALWAYS track (soft down-weight), so the Inc-2 1/sigma^2 weight (scene_rep:597) also fires.
                    from Addons.motion.flow_track import rigid_flow_residual, region_soft_weight, dino_grid
                    # DDS stores poses + batch['direction'] in OPENGL (load_poses negates the y,z axes;
                    # get_camera_rays default type='OpenGL', z=-1). rigid_flow_residual builds OpenCV dirs
                    # (z=+1) + projects with z>0, so convert the c2w's GL->CV (c2w @ diag(1,-1,-1,1)) before
                    # forming T_rel -- else every point lands behind the camera (Z<0) and the bad-mask zeros
                    # the whole residual (the f9 resid=0 no-op seen in the first E3 ablation).
                    _GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
                    _c2w_ref = self.est_c2w_data[ref_id].detach().cpu().numpy().astype(np.float32) @ _GL2CV
                    _c2w_cur = cur_c2w.detach().cpu().numpy().astype(np.float32) @ _GL2CV   # const-velocity prior (line 783/785)
                    _T_rel = (np.linalg.inv(_c2w_cur) @ _c2w_ref).astype(np.float32)   # OpenCV ref-cam -> cur-cam
                    _resid = rigid_flow_residual(ref_bgr, cur_bgr, ref_depth, _T_rel,
                                                 float(self.dataset.fx), float(self.dataset.fy),
                                                 float(self.dataset.cx), float(self.dataset.cy),
                                                 self._raft, self._raft_tf, self.device)
                    _dg = dino_grid(cur_bgr, self._dino, self.device)
                    _wfull, _med, _mad, _scale = region_soft_weight(
                        _resid, _dg, n_groups=int(_ft.get('n_groups', 12)),
                        mad_c=float(_ft.get('mad_c', 2.0)), w_floor_px=float(_ft.get('w_floor_px', 1.0)),
                        w_min=float(_ft.get('w_min', 0.1)))
                    self._trust_map = _wfull   # full-res [H,W] for the trust/ diagnostic panel
                    track_w_map = torch.from_numpy(_wfull[iH:-iH, iW:-iW])   # CPU [H-2iH, W-2iW]
                    _wc = _wfull[iH:-iH, iW:-iW]
                    print(f"[depth_sup] f{frame_id}: resid med={_med:.2f} mad={_mad:.2f} scale={_scale:.2f} "
                          f"w_mean={float(_wc.mean()):.3f} w_min={float(_wc.min()):.3f} "
                          f"frac_dn={float((_wc < 0.5).mean()):.3f} (ref {ref_id})")
                    try:
                        import csv as _csv
                        _tl = os.path.join(self.config['data']['output'], 'trust_log.csv')
                        _new = not os.path.exists(_tl)
                        with open(_tl, 'a', newline='') as _f:
                            _wr = _csv.writer(_f)
                            if _new:
                                _wr.writerow(['frame', 'resid_med', 'resid_mad', 'scale', 'w_mean', 'w_min', 'frac_dn'])
                            _wr.writerow([frame_id, round(_med, 4), round(_mad, 4), round(_scale, 4),
                                          round(float(_wc.mean()), 4), round(float(_wc.min()), 4),
                                          round(float((_wc < 0.5).mean()), 4)])
                    except Exception as _e:
                        print(f"[depth_sup] trust_log.csv write skipped: {_e}")
                else:
                    from Addons.motion.flow_track import flow_residual, residual_to_weight
                    _resid = flow_residual(ref_bgr, cur_bgr, self._raft, self._raft_tf, self.device,
                                           float(_ft.get('ransac_thresh', 1.0)))
                    _w = residual_to_weight(_resid[iH:-iH, iW:-iW], float(_ft.get('alpha', 0.5)),
                                            float(_ft.get('w_min', 0.1)), float(_ft.get('w_max', 1.0)),
                                            deadband=float(_ft.get('deadband', 0.0)))
                    track_w_map = torch.from_numpy(_w)   # CPU [H-2iH, W-2iW]

        cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None,...], mapping=False)

        # capture init pose (const-velocity prediction) for debug
        _init_c2w_for_log = cur_c2w.detach().clone()
        loss_iter0 = None    # loss at iter 0, post-const-velocity init, BEFORE any optimiser step

        # LEAN CORE: constant zero-motion prior on the per-frame RELATIVE pose (cur vs previous committed
        # pose). Kills noise-driven over-travel/jitter; the observability anisotropy EMERGES from the SDF
        # loss's own per-DOF curvature (rotation-dominant/near/observable DOFs -> data overrules; sub-floor/
        # still DOFs -> pinned). No J_i scaling (that cancels vs H_data~J^2). lam_r,lam_t = ONE calibrated-
        # and-frozen balance. Default off (both lam 0, or no prev pose) => base byte-identical.
        # DDS_MP_LAM_R/T env = calibration-SWEEP convenience only; the FROZEN value belongs in the config.
        _mp_lr = float(os.environ.get('DDS_MP_LAM_R', self.config['tracking'].get('motion_prior_lam_r', 0.0)))
        _mp_lt = float(os.environ.get('DDS_MP_LAM_T', self.config['tracking'].get('motion_prior_lam_t', 0.0)))
        _mp_prev = None
        if (_mp_lr > 0 or _mp_lt > 0) and (int(frame_id) - 1) in self.est_c2w_data:
            from Addons.motion.flow_track import zero_motion_prior
            _mp_prev = self.est_c2w_data[int(frame_id) - 1].detach().to(self.device)

        # Start tracking
        for i in range(self.config['tracking']['iter']):
            pose_optimizer.zero_grad()
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

            # Note here we fix the sampled points for optimisation
            if indice is None:
                indice = self.select_samples(self.dataset.H-iH*2, self.dataset.W-iW*2, self.config['tracking']['sample'])

                # Slicing
                indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)
                # tool exclusion from the POSE solve ONLY -- the tool STAYS in the map and is still RENDERED;
                # we just drop the GT tool (seg==2) from the tracking rays so its independent motion can't drag
                # the camera pose. Default off (tool_mask_track flag off OR no seg) => base byte-identical.
                if self.config['training'].get('tool_mask_track', False) and 'seg' in batch:
                    _kt = (batch['seg'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w] != 2)
                    if 0 < int(_kt.sum()) < _kt.numel():
                        indice_h, indice_w = indice_h[_kt], indice_w[_kt]
                rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)
            target_edge_semantic = batch['edge_semantic'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)
            border = batch['border'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)
            # Inc-1 v2 (WildGS-faithful): sample the compact DINO grid; tracking samples the ignore-edge
            # cropped frame, so offset the pixel coords by (iH,iW) into the full-FOV grid.
            target_dino = sample_dino_grid(batch['dino_grid'].squeeze(0), indice_h + iH, indice_w + iW, self.dataset.H, self.dataset.W).to(self.device) if 'dino_grid' in batch else None

            rays_o = c2w_est[...,:3, -1].repeat(rays_d_cam.shape[0], 1)   # ACTUAL ray count (tool_mask_track may drop tool px); == tracking.sample unfiltered => base byte-identical
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

            if self.config['dynamic']:
                cur_id = (frame_id*torch.ones(rays_o.shape[0]))
                timestamps = (cur_id.to(self.device) / self.dataset.num_frames) if self.config['training'].get('time_normalize', False) else cur_id.to(self.device)  # T1.2: normalise frame_time to [0,1]; flag default off = upstream behaviour
                rays_o = torch.cat([rays_o,timestamps.unsqueeze(-1)],dim=1)

            # ARM-2 Inc-2: tracking=True enables the per-ray pose down-weight from
            # sigma^2 (TRACKING-ONLY). Mapping/BA forwards (first_frame/current_frame/
            # global_BA) leave tracking=False (default) so they are untouched. No-op
            # when uncertainty.enable=false (forward guards on unc_on).
            # gather the per-ray flow down-weight at the SAME sampled pixels (cropped-image coords); None on base
            track_ray_w = track_w_map[indice_h, indice_w].view(-1, 1).to(self.device) if track_w_map is not None else None
            ret = self.model.forward(rays_o, rays_d, target_s, target_d, target_edge_semantic=target_edge_semantic, target_dino=target_dino, border=border, UseBorder=True, tracking=True, track_ray_w=track_ray_w)
            loss = self.get_loss_from_ret(ret)
            if i == 0:
                loss_iter0 = float(loss.cpu().item())   # DATA loss (pre-prior), comparable across arms
            # Add the prior BEFORE best-iter selection + backward -- else the hold never binds (the optimiser
            # would pick a jittery low-DATA-loss iteration and skip the constraint entirely). fix-3.
            if _mp_prev is not None:
                loss = loss + zero_motion_prior(c2w_est[0], _mp_prev, _mp_lr, _mp_lt)

            if best_sdf_loss is None:
                best_sdf_loss = loss.cpu().item()
                best_c2w_est = c2w_est.detach()
                best_ret = self._snapshot_ret(ret)

            with torch.no_grad():
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                if loss.cpu().item() < best_sdf_loss:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()
                    best_ret = self._snapshot_ret(ret)
                    thresh = 0
                else:
                    thresh +=1

            if thresh >self.config['tracking']['wait_iters']:
                break

            loss.backward()

            pose_optimizer.step()

        # P2 regression guard: with the prior on, the solve MUST have fired (sigma^2 fires only inside the
        # solve). loss_iter0 is set at i==0, so None here would mean the loop was skipped (a freeze regression).
        if _mp_prev is not None:
            assert loss_iter0 is not None, "P2 guard: tracking solve did not fire with motion_prior on"

        if self.config['tracking']['best']:
            # Use the pose with smallest loss
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            # Use the pose after the last iteration
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

       # Save relative pose of non-keyframes
        if frame_id % self.config['mapping']['keyframe_every'] != 0:
            kf_id = frame_id // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta

        print('Best loss: {}, Last loss{}'.format(F.l1_loss(best_c2w_est.to(self.device)[0,:3], c2w_gt[:3]).cpu().item(), F.l1_loss(c2w_est[0,:3], c2w_gt[:3]).cpu().item()))

        # Debug log — per-frame tracking result, using loss components at the BEST-iter pose
        try:
            _tracking_iters_used = i + 1
            _is_kf = (frame_id % self.config['mapping']['keyframe_every'] == 0)
            _dvalid = ((batch['depth'] > 0) & (batch['depth'] < self.config['cam']['depth_trunc'])).float().mean().item()
            _dmean = batch['depth'][batch['depth'] > 0].mean().item() if (batch['depth'] > 0).any() else 0.0
            br = best_ret if best_ret is not None else {}
            self.debug_logger.log_tracking(
                frame_id=frame_id,
                est_c2w=self.est_c2w_data[frame_id],
                gt_c2w=self.pose_gt[frame_id],
                is_keyframe=_is_kf,
                init_c2w=_init_c2w_for_log,
                tracking_iters_used=_tracking_iters_used,
                tracking_iters_config=self.config['tracking']['iter'],
                best_loss=best_sdf_loss,
                last_loss=loss.cpu().item(),
                loss_iter0=loss_iter0,
                loss_components={
                    'rgb': br.get('rgb_loss'),
                    'depth': br.get('depth_loss'),
                    'sdf': br.get('sdf_loss'),
                    'fs': br.get('fs_loss'),
                    'edge_semantic': br.get('edge_semantic_loss'),
                },
                psnr=br.get('psnr'),
                depth_valid_frac=_dvalid,
                depth_mean=_dmean,
                rgb_mean=batch['rgb'].mean().item(),
                sdf_stats=br.get('sdf_stats'),
            )
        except Exception as _e:
            print(f'[DebugLogger] frame {frame_id} log failed: {_e}')
    
    def convert_relative_pose(self):
        poses = {}
        for i in range(len(self.est_c2w_data)):
            if i % self.config['mapping']['keyframe_every'] == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                c2w_key = self.est_c2w_data[kf_frame_id]
                delta = self.est_c2w_data_rel[i] 
                poses[i] = delta @ c2w_key
        
        return poses

    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
        # time_net (deformation field) is split into its OWN param group so its
        # weight_decay / lr can be controlled for revival experiments. Defaults
        # (timenet_weight_decay=1e-6, timenet_lr_mult=1.0) reproduce the original
        # single decoder group exactly (Adam state is per-param; identical hyperparams
        # in two groups == one group).
        def _dec_groups(include_timenet=True):
            lr_dec = self.config['mapping']['lr_decoder']
            tn_wd = self.config['training'].get('timenet_weight_decay', 1e-6)
            tn_mult = self.config['training'].get('timenet_lr_mult', 1.0)
            main = [p for n, p in self.model.decoder.named_parameters() if 'time_net' not in n]
            timep = [p for n, p in self.model.decoder.named_parameters() if 'time_net' in n]
            g = [{'params': main, 'weight_decay': 1e-6, 'lr': lr_dec}]
            if timep and include_timenet:
                g.append({'params': timep, 'weight_decay': tn_wd, 'lr': lr_dec * tn_mult})
            return g

        # deform_field_teacher_only: keep time_net OUT of the global_BA (map) optimizer, so the render
        # loss can NEVER step/collapse the deformation field -> the field is trained ONLY by the teacher
        # (current_frame_mapping). v0 showed global_BA collapses the field to 0 before the teacher acts;
        # this is the fix. Default off = field trained by both (upstream behaviour).
        _fto = self.config['training'].get('deform_field_teacher_only', False)

        # Optimizer for BA
        trainable_parameters = _dec_groups(include_timenet=not _fto) + [{'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
        if not self.config['grid']['oneGrid']:
            trainable_parameters.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))

        # Optimizer for current frame mapping (ALWAYS trains time_net -> the teacher's update path)
        if self.config['mapping']['cur_frame_iters'] > 0:
            # cur_frame_map_only: keep the current-frame SHARPENING pass but train the MAP ONLY -- exclude the
            # field from this optimizer so the render gradient can't collapse it (the replay keeps it alive).
            # The render forward still USES the alive field, so the map co-adapts to the warp. Default off =
            # field IN (base/teacher_on bit-identical). This is the render-recovery fix for the un-routed gap.
            params_cur_mapping = _dec_groups(include_timenet=not self.config['training'].get('cur_frame_map_only', False)) + [{'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
            if not self.config['grid']['oneGrid']:
                params_cur_mapping.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
            self.cur_map_optimizer = optim.Adam(params_cur_mapping, betas=(0.9, 0.99))

        # ARM-2 Stage-2: dedicated time_net optimizer for the teacher REPLAY -- independent of
        # cur_frame_iters, so the field is trained by replay even when current_frame_mapping is demoted
        # (cur_frame_iters:0). None when the teacher is off. With deform_field_teacher_only:true the field
        # is in NO other optimizer -> trained ONLY by the replay teacher (isolated from render-collapse).
        self.field_optimizer = None
        if self.config['training'].get('deformation_sup_weight', 0) > 0:
            _tnwd = self.config['training'].get('timenet_weight_decay', 1e-6)
            _tnm = self.config['training'].get('timenet_lr_mult', 1.0)
            _timep = [p for n, p in self.model.decoder.named_parameters() if 'time_net' in n]
            if _timep:
                self.field_optimizer = optim.Adam(_timep, lr=self.config['mapping']['lr_decoder'] * _tnm, weight_decay=_tnwd)

    def _buffer_deform(self, batch, cur_frame_id):
        '''Append the current frame's surface targets to the causal replay buffer (only frames seen so far).
        Pose-frozen => Xk is fixed at insertion (consistent with the identity-pose bake).'''
        if 'deform_dx' not in batch:
            return
        H, W = self.dataset.H, self.dataset.W
        n = self.config['mapping']['sample']
        indice = self.select_samples(H, W, n)
        ih, iw = indice % H, indice // H
        c2w = self.est_c2w_data[cur_frame_id].to(self.device)
        rays_d_cam = batch['direction'].squeeze(0)[ih, iw, :].to(self.device)
        target_d = batch['depth'].squeeze(0)[ih, iw].to(self.device).unsqueeze(-1)
        rays_o = c2w[:3, -1].repeat(n, 1)
        rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)
        Xk = rays_o + rays_d * target_d
        dx = sample_dino_grid(batch['deform_dx'].squeeze(0), ih, iw, H, W).to(self.device)
        w = sample_dino_grid(batch['deform_trust'].squeeze(0), ih, iw, H, W).to(self.device)
        # E0 tissue-only teacher buffering (OPT-IN, default OFF). E0 v0.1 turned this ON unconditionally and
        # the field COLLAPSED (pin-EPE -0.2%, |Δx|~0): restricting the field's TRAINING to route-flagged rays
        # zeroes its supervision wherever the (sparse) route missed -> starved -> dead. The field's LIFE must
        # be DECOUPLED from the route. Default: train the field on its FULL targets (stays ALIVE like
        # replay_sharp); the route gates ONLY the field's APPLICATION in the map/render, not its training.
        if getattr(self, '_route_map', None) is not None and self.config.get('map_route', {}).get('tissue_only_teacher', False):
            w = w * self._route_map[ih, iw].view(-1, 1)
        _t = (cur_frame_id / self.dataset.num_frames) if self.config['training'].get('time_normalize', False) else float(cur_frame_id)
        self.deform_replay.append({'Xk': Xk.detach().cpu(), 'dx': dx.detach().cpu(),
                                   'w': w.detach().cpu(), 't': torch.full((n, 1), float(_t))})

    def _sample_replay(self, nf):
        k = min(nf, len(self.deform_replay))
        n = len(self.deform_replay)
        # RECENCY-WEIGHTED replay (deform_replay_recency:true): equal-weight replay under-supervises the
        # LATE large-displacement frames (their share shrinks as the buffer grows) -> the field degrades
        # over time ("starts ok, doesn't match later"). Up-weight recent frames (weight ∝ i+1). Default
        # off = uniform random.sample (byte-identical base).
        if self.config['training'].get('deform_replay_recency', False) and n > k:
            w = np.arange(1, n + 1, dtype=np.float64); w /= w.sum()
            picks = list(np.random.choice(n, k, replace=False, p=w))
        else:
            picks = random.sample(range(n), k)
        cat = lambda key: torch.cat([self.deform_replay[i][key] for i in picks]).to(self.device)
        return cat('Xk'), cat('t'), cat('dx'), cat('w')

    def _deform_replay_step(self, batch, cur_frame_id):
        '''ARM-2 Stage-2 (online forgetting fix): buffer the current frame, then train the field on a REPLAY
        of PAST frames -- causal, so it accumulates the time-varying motion instead of forgetting it
        frame-by-frame. Gated on deform_replay_iters>0; trains time_net via field_optimizer only.'''
        _ds_w = self.config['training'].get('deformation_sup_weight', 0)
        n_it = self.config['training'].get('deform_replay_iters', 0)
        if _ds_w <= 0 or n_it <= 0 or not self.config['dynamic'] or 'deform_dx' not in batch or self.field_optimizer is None:
            return
        self._buffer_deform(batch, cur_frame_id)
        nf = self.config['training'].get('deform_replay_frames', 5)
        _first = _last = 0.0
        for it in range(n_it):
            self.field_optimizer.zero_grad()
            Xk, t, dx, w = self._sample_replay(nf)
            def_sup = self.model.deform_teacher_loss(Xk, t, dx, w)
            (_ds_w * def_sup).backward()
            self.field_optimizer.step()
            if it == 0: _first = def_sup.item()
            _last = def_sup.item()
        if cur_frame_id <= 3 or cur_frame_id % 30 == 0:
            print(f'[replay] frame {cur_frame_id}: buffer={len(self.deform_replay)}  def_sup {_first:.6f} -> {_last:.6f}  (replay {nf}f x {n_it}it)')

    def run(self):
        self.create_optimizer()
        data_loader = DataLoader(self.dataset, num_workers=self.config['data']['num_workers'])

        # Start Co-SLAM!
        for i, batch in tqdm(enumerate(data_loader)):
            # Visualisation
            if self.config['mesh']['visualisation']:
                try:
                    rgb = cv2.cvtColor(batch["rgb"].squeeze().cpu().numpy(), cv2.COLOR_BGR2RGB)
                    raw_depth = batch["depth"]
                    mask = (raw_depth >= self.config["cam"]["depth_trunc"]).squeeze(0)
                    depth_colormap = colormap_image(batch["depth"])
                    depth_colormap[:, mask] = 255.
                    depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()
                    image = np.hstack((rgb, depth_colormap))
                    cv2.namedWindow('RGB-D'.format(i), cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('RGB-D'.format(i), image)
                    key = cv2.waitKey(1)
                except cv2.error:
                    pass  # headless environment, skip display

            # First frame mapping
            if i == 0:
                self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
                #if save_rendering_result:
                self.rendering(batch, i)
            
            # Tracking + Mapping
            else:
                if self.config['tracking']['iter_point'] > 0:
                    self.tracking_pc(batch, i)
                self.tracking_render(batch, i)
                if getattr(self, 'map_route_on', False):
                    self._compute_route_map(batch, i)   # E0: causal field-routing map for this frame (mapper/render/teacher all read self._route_map)
                if i%self.config['mapping']['map_every']==0:
                    self.global_BA(batch, i)
                    self.current_frame_mapping(batch, i)
                    self._deform_replay_step(batch, i)   # ARM-2 Stage-2: causal teacher replay (no-op unless deform_replay_iters>0)

                if i % self.config['render_freq'] == 0:
                    self.rendering(batch, i)

                # Add keyframe
                if i % self.config['mapping']['keyframe_every'] == 0:
                    self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'],
                                                       route_map=(self._route_map if self.config.get('map_route', {}).get('route_ba', False) else None))
                    print('add keyframe:',i)
            

                if i % self.config['mesh']['vis']==0:
                    pose_relative = self.convert_relative_pose()
                    out_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'])
                    pose_evaluation(self.pose_gt, self.est_c2w_data, 1, out_dir, i)
                    pose_evaluation(self.pose_gt, pose_relative, 1, out_dir, i, img='pose_r', name='output_relative.txt')
                    pose_evaluation(self.pose_gt_identity, self.est_c2w_data, 1, out_dir, i, img='pose_id', name='output_identity.txt')
                    self.debug_logger.save_pose_snapshot(self.est_c2w_data, tag=f'frame_{i:05d}')

        model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'checkpoint{}.pt'.format(i))

        self.save_ckpt(model_savepath)

        pose_relative = self.convert_relative_pose()
        out_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'])
        pose_evaluation(self.pose_gt, self.est_c2w_data, 1, out_dir, i)
        pose_evaluation(self.pose_gt, pose_relative, 1, out_dir, i, img='pose_r', name='output_relative.txt')
        pose_evaluation(self.pose_gt_identity, self.est_c2w_data, 1, out_dir, i, img='pose_id', name='output_identity.txt')
        est_c2w_data_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'est_c2w_data.txt')
        with open(est_c2w_data_path, 'w') as f:
            for key, value in self.est_c2w_data.items():
                f.write(" ".join(map(str, value.cpu().numpy().reshape(16)[:12].tolist())) + "\n")
        print('Saved estimated camera poses to {}'.format(est_c2w_data_path))
        #TODO: Evaluation of reconstruction

    def rendering(self, batch, frame_id):
        H = self.dataset.H
        W = self.dataset.W
        cur_c2w = self.est_c2w_data[frame_id]

        # indice = None
        # best_sdf_loss = None
        # thresh = 0
        # iW = self.config['tracking']['ignore_edge_W']
        # iH = self.config['tracking']['ignore_edge_H']
        # indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)

        cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None, ...], mapping=True)
        c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

        rays_d_cam = batch['direction'].squeeze(0).to(self.device)
        target_s = batch['rgb'].squeeze(0)
        target_d = batch['depth'].squeeze(0).to(self.device).unsqueeze(-1).view(-1, 1)
        # target_edge = batch['edge'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)
        # target_edge_semantic = batch['edge_semantic'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)

        target_edge_semantic = batch['edge_semantic'].squeeze(0).to(self.device).unsqueeze(-1)
        # Inc-1 v2 (WildGS-faithful): sample the compact DINO grid at ALL pixels (row-major, matching
        # rays_d.view(-1,3)) -> [H*W,C]; chunked alongside rays below so render surfaces the sigma^2 viz.
        if 'dino_grid' in batch:
            _Hh, _Ww = self.dataset.H, self.dataset.W
            _hh = torch.arange(_Hh).repeat_interleave(_Ww); _ww = torch.arange(_Ww).repeat(_Hh)
            target_dino_full = sample_dino_grid(batch['dino_grid'].squeeze(0), _hh, _ww, _Hh, _Ww).to(self.device)
        else:
            target_dino_full = None

        rays_o = c2w_est[..., :3, -1].repeat(H * W, 1)
        rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1).view(-1, 3)
        # E0 mapping-routing: full-image field-route (row-major [H*W], matches rays_d.view(-1,3)); sliced per
        # chunk below. The render MUST share the mapper's routing or the background (co-adapted to a tissue-only
        # warp) would be re-warped here. None unless map_route.enable + buffer filled -> unrouted (base).
        route_flat = self._route_map.reshape(-1) if getattr(self, '_route_map', None) is not None else None

        rgb = []
        depth_chunks = []
        sigma_chunks = []   # ARM-2 Inc-1: per-pixel sigma^2 (only when the head is on)
        whatkind_chunks = []   # ARM-1 #4: per-pixel what-kind logits (only when the slot module is on)
        ray_batch_size = 240

        for i in range(0, rays_d.shape[0], ray_batch_size):
            torch.cuda.empty_cache()
            rays_o1 = rays_o[i:i + ray_batch_size]
            rays_d1 = rays_d[i:i + ray_batch_size]
            target_d1 = target_d[i:i + ray_batch_size]
            target_dino1 = target_dino_full[i:i + ray_batch_size] if target_dino_full is not None else None
            route_w1 = route_flat[i:i + ray_batch_size].view(-1, 1) if route_flat is not None else None
            if self.config['dynamic']:
                cur_id = (frame_id*torch.ones(rays_o1.shape[0]))
                timestamps = (cur_id.to(self.device) / self.dataset.num_frames) if self.config['training'].get('time_normalize', False) else cur_id.to(self.device)  # T1.2: normalise frame_time to [0,1]; flag default off = upstream behaviour
                rays_o1 = torch.cat([rays_o1,timestamps.unsqueeze(-1)],dim=1)
            # ret = self.model.render_rays(rays_o1, rays_d1, target_d1)
            ret = self.model.forward(rays_o1, rays_d1, target_s, target_d1,
                                     target_edge_semantic=target_edge_semantic, target_dino=target_dino1, notFirstMap=False,render_only=True, route_w=route_w1)
            rgb.append(ret['rgb'].detach().clone().cpu())
            if 'depth' in ret:
                depth_chunks.append(ret['depth'].detach().clone().cpu())
            if 'sigma2' in ret:
                sigma_chunks.append(ret['sigma2'].detach().clone().cpu())
            if 'whatkind' in ret:
                whatkind_chunks.append(ret['whatkind'].detach().clone().cpu())

        color = torch.cat(rgb, dim=0)
        color = color.reshape(H, W, 3)
        color_np = color.detach().cpu().numpy()


        # Removed per-frame min-max contrast stretch that was here previously.
        # The stretch was: color_np = (color_np - np.min(color_np)) / (np.max(color_np) - np.min(color_np))
        # It amplified contrast asymmetrically between rendered and GT (GT was never
        # stretched), disproportionately hurting LPIPS while leaving PSNR/SSIM mostly
        # intact. Replaced with plain clip to [0,1] which preserves the SDF renderer's
        # natural sigmoid output range.
        color_np = np.clip(color_np, 0, 1)
        color_path = os.path.join(self.config['data']['output'],'{:0>4d}.jpg'.format(frame_id))

        plt.imsave(color_path, color_np)

        # E0: save the per-pixel field-route map for the video overlay (255=moving=field-ON, 0=static->sharp).
        # Gated on map_route; warm-up frames (no causal ref yet) save an all-zero map so the panel stays
        # frame-aligned with the renders. Cheap single-channel PNG; OFF -> nothing written.
        if getattr(self, 'map_route_on', False):
            _rt_dir = os.path.join(self.config['data']['output'], 'route'); os.makedirs(_rt_dir, exist_ok=True)
            _rt = (self._route_map.detach().cpu().numpy() if getattr(self, '_route_map', None) is not None
                   else np.zeros((H, W), np.float32))
            cv2.imwrite(os.path.join(_rt_dir, '{:0>4d}.png'.format(frame_id)), (_rt * 255).astype(np.uint8))

        # Depth-supervisor: save the per-pixel TRUST weight map (1.0=moves-with-camera/trusted ->
        # 0=down-weighted/deforming) as uint16 for the video "Trust Weight" panel + offline diagnosis.
        # Gated on the depth supervisor having run this frame (_trust_map set in tracking_render); base -> nothing.
        if getattr(self, '_trust_map', None) is not None:
            _tr_dir = os.path.join(self.config['data']['output'], 'trust'); os.makedirs(_tr_dir, exist_ok=True)
            _tw = self._trust_map
            if _tw.shape != (H, W):
                _tw = cv2.resize(_tw, (W, H), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(_tr_dir, '{:0>4d}.png'.format(frame_id)), np.clip(_tw * 65535.0, 0, 65535).astype(np.uint16))

        # Save the model's rendered DEPTH as uint16 PNG for visualization.
        # IMPORTANT: this uses cam.output_depth_scale (NOT png_depth_scale).
        # png_depth_scale is the INPUT-data convention -- e.g. SemSup NPYs are
        # stored as depth_m*8, so SuperDataset divides loaded depth by 8 to
        # recover metres. Changing png_depth_scale to make output PNGs prettier
        # would break the INPUT loader. So we use a separate output_depth_scale
        # for save-only purposes; defaults to 10000 (gives 0.05-5m depth ->
        # uint16 500-50000, plenty of dynamic range for endoscopic scenes).
        # Falls back to png_depth_scale -> 10000 to keep old behaviour for
        # configs that haven't migrated.
        if depth_chunks:
            depth_dir = os.path.join(self.config['data']['output'], 'depth')
            os.makedirs(depth_dir, exist_ok=True)
            depth_render = torch.cat(depth_chunks, dim=0).reshape(H, W).numpy()
            cam_cfg = self.config.get('cam', {})
            output_depth_scale = float(cam_cfg.get('output_depth_scale',
                                                   cam_cfg.get('png_depth_scale', 10000.0)))
            depth_uint16 = np.clip(depth_render * output_depth_scale, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(depth_dir, '{:04d}.png'.format(frame_id)), depth_uint16)

        # ARM-2 Inc-1: save the model's volume-rendered per-pixel uncertainty (sigma^2)
        # EXACTLY as the model computes it on THIS render -- same estimated pose, same
        # depth-guided sampling (target_d), same un-normalised rays as the RGB above ->
        # a faithful "how the model sees its own uncertainty" map. Present ONLY when the
        # head is on (sigma_chunks empty otherwise), so base runs are byte-unchanged.
        # uint16 = sigma^2 * uncert_save_scale (recoverable); generate_video.py colormaps
        # it (inferno, robust p2..p98). Default scale 10000 like output depth.
        if sigma_chunks:
            unc_dir = os.path.join(self.config['data']['output'], 'uncert')
            os.makedirs(unc_dir, exist_ok=True)
            sig = torch.cat(sigma_chunks, dim=0).reshape(H, W).numpy()
            unc_scale = float(self.config.get('uncertainty', {}).get('save_scale', 10000.0))
            sig_uint16 = np.clip(sig * unc_scale, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(unc_dir, '{:04d}.png'.format(frame_id)), sig_uint16)

        # ARM-1 #4: save the LEARNED what-kind attribution map as a coloured PNG (argmax over the K
        # slots -> {0 bg, 1 tissue, 2 tool}). Present ONLY when the slot module is on (whatkind_chunks
        # empty otherwise), so base / per-pixel-dino runs are byte-unchanged. Colours are BGR (cv2):
        # bg=black, tissue=green [0,255,0], tool=red [0,0,255]. generate_video.py can panel this beside
        # the sigma^2 map. Self-contained, mirrors the uncert save.
        if whatkind_chunks:
            wk_dir = os.path.join(self.config['data']['output'], 'whatkind')
            os.makedirs(wk_dir, exist_ok=True)
            wk_logits = torch.cat(whatkind_chunks, dim=0)            # [H*W, K]
            wk = wk_logits.argmax(dim=-1).reshape(H, W).numpy().astype(np.int64)
            wk_rgb = np.zeros((H, W, 3), dtype=np.uint8)            # BGR; bg stays black
            wk_rgb[wk == 1] = (0, 255, 0)                           # tissue -> green
            wk_rgb[wk == 2] = (0, 0, 255)                           # tool   -> red
            cv2.imwrite(os.path.join(wk_dir, '{:04d}.png'.format(frame_id)), wk_rgb)


if __name__ == '__main__':
            
    print('Start running...')
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    if args.output is not None:
        cfg['data']['output'] = args.output

    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy("ddsslam.py", os.path.join(save_path, 'ddsslam.py'))

    with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))


    slam = DDSSLAM(cfg)

    slam.run()
