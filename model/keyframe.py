import torch
import numpy as np
import random

class KeyFrameDatabase(object):
    def __init__(self, config, H, W, num_kf, num_rays_to_save, device) -> None:
        self.config = config
        self.keyframes = {}
        self.device = device
        # Keyframe rays = [dir3, rgb3, depth1, edge1] = width 8 (depth@6, edge@7). Inc-1 v2 (WildGS-
        # faithful) does NOT store DINO here: the head trains via per-frame current_frame mapping with
        # on-demand grid sampling (ddsslam.sample_dino_grid), so global_BA stays DINO-free and the DB
        # is byte-identical to base. (An earlier per-ray-DINO tail bloated this to ~5GB -> removed.)
        self.ray_w = 8
        self.rays = torch.zeros((num_kf, num_rays_to_save, self.ray_w))
        # B2: parallel per-keyframe field-route weight (rays stay width 8 for parity). 1 = field ON (neutral);
        # add_keyframe overwrites it when a route_map is passed (map_route.route_ba).
        self.route = torch.ones((num_kf, num_rays_to_save, 1))
        # ORACLE-FIELD (deform_oracle): parallel per-keyframe baked dx* at the SAME subsampled pixels, so
        # global_BA rays can be warped without pixel identity (mirrors the route mechanism). Zeros = no warp.
        self.dx = torch.zeros((num_kf, num_rays_to_save, 3))
        self.num_rays_to_save = num_rays_to_save
        self.frame_ids = None
        self.H = H
        self.W = W

    
    def __len__(self):
        return len(self.frame_ids)
    
    def get_length(self):
        return self.__len__()
    
    def sample_single_keyframe_rays(self, rays, option='random'):
        '''
        Sampling strategy for current keyframe rays
        '''
        if option == 'random':
            idxs = random.sample(range(0, self.H*self.W), self.num_rays_to_save)
        elif option == 'filter_depth':
            # depth is at absolute index 6 ([dir3,rgb3,depth1,...]); was [...,-2] which only equals 6
            # when ray_w==8 -- with the DINO tail appended that would wrongly read a DINO channel.
            valid_depth_mask = (rays[..., 6] > 0.0) & (rays[..., 6] <= self.config["cam"]["depth_trunc"])
            rays_valid = rays[valid_depth_mask, :]  # [n_valid, ray_w]
            num_valid = len(rays_valid)
            idxs = random.sample(range(0, num_valid), self.num_rays_to_save)

        else:
            raise NotImplementedError()
        rays = rays[:, idxs]
        return rays, idxs   # B2: also return idxs so add_keyframe gathers the route at the SAME pixels
    
    def attach_ids(self, frame_ids):
        '''
        Attach the frame ids to list
        '''
        if self.frame_ids is None:
            self.frame_ids = frame_ids
        else:
            self.frame_ids = torch.cat([self.frame_ids, frame_ids], dim=0)
    
    def add_keyframe(self, batch, filter_depth=False, route_map=None, dx_map=None):
        '''
        Add keyframe rays to the keyframe database. route_map: [H,W] field-route for this frame (B2);
        None -> leave the neutral ONES (field ON / un-routed).
        '''
        # batch direction (Bs=1, H*W, 3). Pack [dir3, rgb3, depth1, edge1] = width 8 (DINO is NOT
        # stored here; v2 samples it on-demand per frame -- see ddsslam.sample_dino_grid).
        rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None], batch['edge_semantic'][..., None]], dim=-1)
        rays = rays.reshape(1, -1, rays.shape[-1])
        if filter_depth:
            rays, idxs = self.sample_single_keyframe_rays(rays, 'filter_depth')
        else:
            rays, idxs = self.sample_single_keyframe_rays(rays)

        if not isinstance(batch['frame_id'], torch.Tensor):
            batch['frame_id'] = torch.tensor([batch['frame_id']])

        self.attach_ids(batch['frame_id'])

        # Store the rays
        self.rays[len(self.frame_ids)-1] = rays
        # B2: store the per-keyframe route at the SAME subsampled pixels (idxs) -> aligned with rays.
        # None (warm-up / route_ba off) -> keep the ONES default (field ON = un-routed = neutral).
        if route_map is not None:
            self.route[len(self.frame_ids)-1] = route_map.reshape(-1)[idxs].view(-1, 1).float().cpu()
        # ORACLE-FIELD: this keyframe's per-pixel dx* at the SAME subsampled pixels (aligned with rays).
        if dx_map is not None:
            self.dx[len(self.frame_ids)-1] = dx_map.reshape(-1, 3)[idxs].float().cpu()
    
    def sample_global_rays(self, bs, with_route=False, with_dx=False):
        '''
        Sample rays from self.rays as well as frame_ids. with_route (B2): also return the stored
        per-keyframe field-route at the SAME sampled rays (to route global_BA's keyframe rays).
        with_dx (ORACLE-FIELD): also return the stored per-keyframe dx* at the SAME sampled rays.
        Flags append to the return tuple in (route, dx) order; both default False = base signature.
        '''
        num_kf = self.__len__()
        idxs = torch.tensor(random.sample(range(num_kf * self.num_rays_to_save), bs))
        sample_rays = self.rays[:num_kf].reshape(-1, self.ray_w)[idxs]

        frame_ids = self.frame_ids[idxs//self.num_rays_to_save]

        out = (sample_rays, frame_ids)
        if with_route:
            out = out + (self.route[:num_kf].reshape(-1, 1)[idxs],)
        if with_dx:
            out = out + (self.dx[:num_kf].reshape(-1, 3)[idxs],)
        return out
    
    def sample_global_keyframe(self, window_size, n_fixed=1):
        '''
        Sample keyframe globally
        Window size: limit the window size for keyframe
        n_fixed: sample the last n_fixed keyframes
        '''
        if window_size >= len(self.frame_ids):
            return self.rays[:len(self.frame_ids)], self.frame_ids
        
        current_num_kf = len(self.frame_ids)
        last_frame_ids = self.frame_ids[-n_fixed:]

        # Random sampling
        idx = random.sample(range(0, len(self.frame_ids) -n_fixed), window_size)

        # Include last n_fixed 
        idx_rays = idx + list(range(current_num_kf-n_fixed, current_num_kf))
        select_rays = self.rays[idx_rays]

        return select_rays, \
               torch.cat([self.frame_ids[idx], last_frame_ids], dim=0)
                    
    @torch.no_grad()
    def sample_overlap_keyframe(self, batch, frame_id, est_c2w_list, k_frame, n_samples=16, n_pixel=100, dataset=None):
        '''
        NICE-SLAM strategy for selecting overlapping keyframe from all previous frames

        batch: Information of current frame
        frame_id: id of current frame
        est_c2w_list: estimated c2w of all frames
        k_frame: num of keyframes for BA i.e. window size
        n_samples: num of sample points for each ray
        n_pixel: num of pixels for computing overlap
        '''
        c2w_est = est_c2w_list[frame_id]       

        indices = torch.randint(dataset.H* dataset.W, (n_pixel,))
        rays_d_cam = batch['direction'].reshape(-1, 3)[indices].to(self.device)
        target_d = batch['depth'].reshape(-1, 1)[indices].repeat(1, n_samples).to(self.device)
        rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:3, :3], -1)
        rays_o = c2w_est[None, :3, -1].repeat(rays_d.shape[0], 1).to(self.device)        

        t_vals = torch.linspace(0., 1., steps=n_samples).to(target_d)
        near = target_d*0.8
        far = target_d+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples, 3]
        pts_flat = pts.reshape(-1, 3).cpu().numpy()

        key_frame_list = []

        for i, frame_id in enumerate(self.frame_ids):
            frame_id = int(frame_id.item())
            c2w = est_c2w_list[frame_id].cpu().numpy()
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(pts_flat[:, 0]).reshape(-1, 1)
            pts_flat_homo = np.concatenate(
                [pts_flat, ones], axis=1).reshape(-1, 4, 1)  # (N, 4)
            cam_cord_homo = w2c@pts_flat_homo  # (N, 4, 1)=(4,4)*(N, 4, 1)
            cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
            K = np.array([[self.config['cam']['fx'], .0, self.config['cam']['cx']], 
                          [.0, self.config['cam']['fy'], self.config['cam']['cy']],
                         [.0, .0, 1.0]]).reshape(3, 3)
            cam_cord[:, 0] *= -1
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)
            edge = 20
            mask = (uv[:, 0] < self.config['cam']['W']-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < self.config['cam']['H']-edge)*(uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum()/uv.shape[0]
            key_frame_list.append(
                {'id': frame_id, 'percent_inside': percent_inside, 'sample_id':i})
        
            

        key_frame_list = sorted(
        key_frame_list, key=lambda i: i['percent_inside'], reverse=True)
        selected_keyframe_list = [dic['sample_id']
                                for dic in key_frame_list if dic['percent_inside'] > 0.00]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k_frame])

        last_id = len(self.frame_ids) - 1

        if last_id not in selected_keyframe_list:
            selected_keyframe_list.append(last_id)

        return self.rays[selected_keyframe_list], selected_keyframe_list