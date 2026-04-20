# package imports
import torch
import torch.nn as nn
import torchvision
#import wandb


# Local imports
from .encodings import get_encoder
from .decoder import ColorSDFNet, ColorSDFNet_v1,ColorSDFNet_v2
from .utils import sample_pdf, batchify, get_sdf_loss, mse2psnr, compute_loss

class JointEncoding(nn.Module):
    def __init__(self, config, bound_box):
        super(JointEncoding, self).__init__()
        self.config = config
        self.bounding_box = bound_box
        self.get_resolution()
        self.get_encoding(config)
        self.get_decoder(config)
        self.n_imgs = self.config['timesteps']



    def get_resolution(self):
        '''
        Get the resolution of the grid
        '''
        dim_max = (self.bounding_box[:,1] - self.bounding_box[:,0]).max()
        if self.config['grid']['voxel_sdf'] > 10:
            self.resolution_sdf = self.config['grid']['voxel_sdf']
        else:
            self.resolution_sdf = int(dim_max / self.config['grid']['voxel_sdf'])
        
        if self.config['grid']['voxel_color'] > 10:
            self.resolution_color = self.config['grid']['voxel_color']
        else:
            self.resolution_color = int(dim_max / self.config['grid']['voxel_color'])
        
        print('SDF resolution:', self.resolution_sdf)

    def get_encoding(self, config):
        '''
        Get the encoding of the scene representation
        '''
        # Coordinate encoding
        self.embedpos_fn, self.input_ch_pos = get_encoder(config['pos']['enc'], n_bins=self.config['pos']['n_bins'])

        # Sparse parametric encoding (SDF)
        self.embed_fn, self.input_ch = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_sdf)
        
        #time encoding
        self.embed_time,self.input_ch_time = get_encoder('freq',input_dim=1)
        self.embed_fre_pos, self.input_ch_fre = get_encoder('freq',input_dim=3)
        # Sparse parametric encoding (Color)
        if not self.config['grid']['oneGrid']:
            print('Color resolution:', self.resolution_color)
            self.embed_fn_color, self.input_ch_color = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_color)

    def get_decoder(self, config):
        '''
        Get the decoder of the scene representation
        '''
        if not self.config['grid']['oneGrid']:
            self.decoder = ColorSDFNet(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)
        else:
            self.decoder = ColorSDFNet_v2(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos,input_ch_time=self.input_ch_time,input_ch_fre=self.input_ch_fre)
        
        self.color_net = batchify(self.decoder.color_net, None)
        self.sdf_net = batchify(self.decoder.sdf_net, None)
        self.edgenet_semantic = batchify(self.decoder.edgenet_semantic, None)
        self.time_net = batchify(self.decoder.time_net, None)


    def sdf2weights(self, sdf, z_vals, args=None):
        '''
        Convert signed distance function to weights.

        Params:
            sdf: [N_rays, N_samples]
            z_vals: [N_rays, N_samples]
        Returns:
            weights: [N_rays, N_samples]
        '''
        weights = torch.sigmoid(sdf / args['training']['trunc']) * torch.sigmoid(-sdf / args['training']['trunc'])

        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds) # The first surface
        mask = torch.where(z_vals < z_min + args['data']['sc_factor'] * args['training']['trunc'], torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)
    
    def raw2outputs(self, raw, edge_semantic, z_vals, white_bkgd=False):
        '''
        Perform volume rendering using weights computed from sdf.

        Params:
            raw: [N_rays, N_samples, 4]
            z_vals: [N_rays, N_samples]
        Returns:
            rgb_map: [N_rays, 3]
            disp_map: [N_rays]
            acc_map: [N_rays]
            weights: [N_rays, N_samples]
        '''
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        edge_semantic = torch.sigmoid(edge_semantic) 

        weights = self.sdf2weights(raw[..., 3], z_vals, args=self.config)
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
        edge_semantic_map = torch.sum(weights[...,None] * edge_semantic, -2)


        depth_map = torch.sum(weights * z_vals, -1)
        depth_var = torch.sum(weights * torch.square(z_vals - depth_map.unsqueeze(-1)), dim=-1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return rgb_map, disp_map, acc_map, weights, depth_map, depth_var, edge_semantic_map
      
    def query_color_sdf(self, query_points):
        '''
        Query the color and sdf at query_points.

        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            raw: [N_rays, N_samples, 4]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        embed = self.embed_fn(inputs_flat)
        embe_pos = self.embedpos_fn(inputs_flat)
        if not self.config['grid']['oneGrid']:
            embed_color = self.embed_fn_color(inputs_flat)
            return self.decoder(embed, embe_pos, embed_color)
        return self.decoder(embed, embe_pos)
    
    def run_network(self, inputs):
        """
        Run the network on a batch of inputs.

        Params:
            inputs: [N_rays, N_samples, 3]
        Returns:
            outputs: [N_rays, N_samples, 4]
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        
        if self.config['dynamic']:
            pts = inputs_flat[:,:3]
            frame_time = inputs_flat[:,3].unsqueeze(-1)
            embed_time = self.embed_time(frame_time)
            embed_pos = self.embed_fre_pos(pts)
            h = torch.cat([embed_time,embed_pos],dim=-1)
            vox_motion = self.time_net(h)
            vox_motion = torch.where(frame_time.reshape(-1, frame_time.shape[-1]) == 0, torch.zeros_like(vox_motion), vox_motion)
            inputs_flat = pts + vox_motion
        
        # Normalize the input to [0, 1] (TCNN convention)
        if self.config['grid']['tcnn_encoding']:
            inputs_flat = (inputs_flat - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        outputs_flat, edge_semantic = batchify(self.query_color_sdf, None)(inputs_flat)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        edge_semantic = torch.reshape(edge_semantic, list(inputs.shape[:-1]) + [edge_semantic.shape[-1]])
        return outputs, edge_semantic
    
    def query_sdf(self, query_points, return_geo=False, embed=False):
        '''
        Get the SDF value of the query points
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            sdf: [N_rays, N_samples]
            geo_feat: [N_rays, N_samples, channel]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
  
        embedded = self.embed_fn(inputs_flat)
        if embed:
            return torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])

        embedded_pos = self.embedpos_fn(inputs_flat)
        out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))
        sdf, geo_feat = out[..., :1], out[..., 1:]

        sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
        if not return_geo:
            return sdf
        geo_feat = torch.reshape(geo_feat, list(query_points.shape[:-1]) + [geo_feat.shape[-1]])

        return sdf, geo_feat

    def query_sdf_at_time(self, query_points, timestamp):
        '''
        Query SDF with deformation applied for a specific timestamp.
        Used for time-aware mesh extraction.

        Params:
            query_points: [N, 3] points in world space
            timestamp: scalar frame index (0, 1, ..., N-1)
        Returns:
            sdf: [N] signed distance values at the deformed positions
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        if self.config['dynamic']:
            pts = inputs_flat[:, :3]
            frame_time = torch.full((pts.shape[0], 1), timestamp,
                                    device=pts.device, dtype=pts.dtype)
            embed_time = self.embed_time(frame_time)
            embed_pos = self.embed_fre_pos(pts)
            h = torch.cat([embed_time, embed_pos], dim=-1)
            vox_motion = self.time_net(h)
            # Zero out motion at frame 0 (canonical frame)
            if timestamp == 0:
                vox_motion = torch.zeros_like(vox_motion)
            inputs_flat = pts + vox_motion

        # Normalize for TCNN
        if self.config['grid']['tcnn_encoding']:
            inputs_flat = (inputs_flat - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
            inputs_flat = torch.clamp(inputs_flat, 1e-6, 1.0 - 1e-6)

        embedded = self.embed_fn(inputs_flat)
        embedded_pos = self.embedpos_fn(inputs_flat)
        out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))
        sdf = out[..., :1]

        return torch.reshape(sdf, list(query_points.shape[:-1]))

    def query_color_at_time(self, query_points, timestamp):
        '''
        Query color with deformation applied for a specific timestamp.

        Params:
            query_points: [N, 3] points in world space (already in TCNN [0,1] space)
            timestamp: scalar frame index
        Returns:
            color: [N, 3] RGB values (after sigmoid)
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        if self.config['dynamic']:
            pts = inputs_flat[:, :3]
            frame_time = torch.full((pts.shape[0], 1), timestamp,
                                    device=pts.device, dtype=pts.dtype)
            embed_time = self.embed_time(frame_time)
            embed_pos = self.embed_fre_pos(pts)
            h = torch.cat([embed_time, embed_pos], dim=-1)
            vox_motion = self.time_net(h)
            if timestamp == 0:
                vox_motion = torch.zeros_like(vox_motion)
            inputs_flat = pts + vox_motion

        if self.config['grid']['tcnn_encoding']:
            inputs_flat = (inputs_flat - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
            inputs_flat = torch.clamp(inputs_flat, 1e-6, 1.0 - 1e-6)

        out = self.query_color_sdf(inputs_flat)
        if isinstance(out, tuple):
            raw_color = out[0][..., :3]
        else:
            raw_color = out[..., :3]

        return torch.sigmoid(raw_color)

    def render_rays(self, rays_o, rays_d, target_d=None):
        '''
        Params:
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            target_d: [N_rays, 1]

        '''
        n_rays = rays_o.shape[0]

        # Sample depth
        if target_d is not None:
            z_samples = torch.linspace(-self.config['training']['range_d'], self.config['training']['range_d'], steps=self.config['training']['n_range_d']).to(target_d) 
            z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d
            z_samples[target_d.squeeze()<=0] = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], steps=self.config['training']['n_range_d']).to(target_d) 

            if self.config['training']['n_samples_d'] > 0:
                z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples_d'])[None, :].repeat(n_rays, 1).to(rays_o)
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            else:
                z_vals = z_samples
        else:
            z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples']).to(rays_o)
            z_vals = z_vals[None, :].repeat(n_rays, 1) # [n_rays, n_samples]

        # Perturb sampling depths
        if self.config['training']['perturb'] > 0.:
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            z_vals = lower + (upper - lower) * torch.rand(z_vals.shape).to(rays_o)

        # Run rendering pipeline
        pts = rays_o[...,None,:3] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        timestamps = (rays_o[...,3]).reshape(-1,1)
        timestamps = timestamps.repeat(1,pts.shape[1]).unsqueeze(-1)
        pts = torch.cat([pts,timestamps],dim=-1)
        raw, edge_semantic = self.run_network(pts)
        rgb_map, disp_map, acc_map, weights, depth_map, depth_var, edge_semantic_map = self.raw2outputs(raw,edge_semantic, z_vals, self.config['training']['white_bkgd'])

        # Importance sampling
        if self.config['training']['n_importance'] > 0:

            rgb_map_0, disp_map_0, acc_map_0, depth_map_0, depth_var_0, edge_map_0,edge_semantic_map = rgb_map, disp_map, acc_map, depth_map, depth_var, edge_map,edge_semantic_map

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.config['training']['n_importance'], det=(self.config['training']['perturb']==0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            raw, edge_semantic = self.run_network(pts)
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var, edge_map,edge_semantic_map = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])

        # Return rendering outputs
        ret = {
            'rgb' : rgb_map,
            'depth' :depth_map,
            'disp_map' : disp_map,
            'acc_map' : acc_map,
            'depth_var':depth_var,
            'edge_semantic':edge_semantic_map
        }
        ret = {**ret, 'z_vals': z_vals}
        ret['raw'] = raw


        # n_importance = 0
        if self.config['training']['n_importance'] > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['depth0'] = depth_map_0
            ret['depth_var0'] = depth_var_0
            ret['edge0'] = edge_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)


        return ret
    
    def forward(self, rays_o, rays_d, target_rgb, target_d, global_step=0,target_edge_semantic=None, border=None, notFirstMap=True, UseBorder=False,render_only=False):
        '''
        Params:
            rays_o: ray origins (Bs, 3)
            rays_d: ray directions (Bs, 3)
            frame_ids: use for pose correction (Bs, 1)
            target_rgb: rgb value (Bs, 3)
            target_d: depth value (Bs, 1)
            c2w_array: poses (N, 4, 4) 
             r r r tx
             r r r ty
             r r r tz
        '''

        # Get render results
        rend_dict = self.render_rays(rays_o, rays_d, target_d=target_d)

        if not self.training:
            return rend_dict
        
        # Get depth and rgb weights for loss
        valid_depth_mask = (target_d.squeeze() > 0.) * (target_d.squeeze() < self.config['cam']['depth_trunc'])
        rgb_weight = valid_depth_mask.clone().unsqueeze(-1)
        rgb_weight[rgb_weight==0] = self.config['training']['rgb_missing']

        # Get render loss
        if not render_only:
 
            rgb_loss = compute_loss(rend_dict["rgb"]*rgb_weight, target_rgb*rgb_weight)
            psnr = mse2psnr(rgb_loss)
            depth_loss = compute_loss(rend_dict["depth"].squeeze()[valid_depth_mask], target_d.squeeze()[valid_depth_mask])

            if UseBorder is False:
                edge_semantic_loss = compute_loss(
                    rend_dict["edge_semantic"].squeeze()[valid_depth_mask],
                    target_edge_semantic.squeeze()[valid_depth_mask],
                    UsePercentage=notFirstMap
                )
            else:
                edge_semantic_loss = compute_loss(
                    rend_dict["edge_semantic"].squeeze()[valid_depth_mask],
                    target_edge_semantic.squeeze()[valid_depth_mask],
                    border=border.squeeze()[valid_depth_mask],
                    UsePercentage=notFirstMap
                )

            if 'rgb0' in rend_dict:
                rgb_loss += compute_loss(rend_dict["rgb0"]*rgb_weight, target_rgb*rgb_weight)
                depth_loss += compute_loss(rend_dict["depth0"].squeeze()[valid_depth_mask], target_d.squeeze()[valid_depth_mask])

            # Get sdf loss
            z_vals = rend_dict['z_vals']  # [N_rand, N_samples + N_importance]
            sdf = rend_dict['raw'][..., -1]  # [N_rand, N_samples + N_importance]
            truncation = self.config['training']['trunc'] * self.config['data']['sc_factor']
            fs_loss, sdf_loss, sdf_stats = get_sdf_loss(z_vals, target_d, sdf, truncation, 'l2', grad=None)

        else:
            rgb_loss = depth_loss = edge_loss = edge_semantic_loss = sdf_loss = fs_loss = psnr = None
            sdf_stats = None

        ret = {
            "rgb": rend_dict["rgb"],
            "depth": rend_dict["depth"],
            "edge_semantic": rend_dict["edge_semantic"],
            "rgb_loss": rgb_loss,
            "depth_loss": depth_loss,
            "edge_semantic_loss": edge_semantic_loss,
            "sdf_loss": sdf_loss,
            "fs_loss": fs_loss,
            "psnr": psnr,
            "sdf_stats": sdf_stats,
        }

        return ret
