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
        
        # Frequency encodings for the 4D deformation network.
        # Paper Section III-A Eq 2 specifies: L=10 for spatial coord x, L=4 for time t.
        # encodings.py:get_encoder default is n_frequencies=12 for both -- i.e. 20% over
        # paper for x and 3x over paper for t. We expose both as config knobs (under
        # decoder.freq_n_xyz / decoder.freq_n_t) so existing runs default to the 12/12
        # behaviour while a paper-faithful config can override to 10/4.
        _n_freq_t   = self.config.get('decoder', {}).get('freq_n_t',   12)
        _n_freq_xyz = self.config.get('decoder', {}).get('freq_n_xyz', 12)
        self.embed_time,    self.input_ch_time = get_encoder('freq', input_dim=1, n_frequencies=_n_freq_t)
        self.embed_fre_pos, self.input_ch_fre  = get_encoder('freq', input_dim=3, n_frequencies=_n_freq_xyz)
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

        # --- ARM-2 stack flags (Inc-0 plumbing). Read once here; ALL default-OFF.
        # These are inert attribute reads (no module, no RNG, no forward branch) so the
        # build stays bit-identical to base. Inc-1+ will construct self.uncertainty_net /
        # self.sni_film ONLY inside these guards (so flags-off consumes zero RNG).
        self.unc_on  = bool(self.config.get('uncertainty', {}).get('enable', False))
        self.nrgs_on = bool(self.config.get('nrgs', {}).get('enable', False))
        self.sni_on  = bool(self.config.get('sni', {}).get('enable', False))


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
    
    def raw2outputs(self, raw, edge_semantic, z_vals, white_bkgd=False, sigma2=None):
        '''
        Perform volume rendering using weights computed from sdf.

        Params:
            raw: [N_rays, N_samples, 4]
            z_vals: [N_rays, N_samples]
            sigma2: [N_rays, N_samples, 1] raw per-sample uncertainty (Inc-1) or None
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

        # --- ARM-2 Inc-1: volume-render sigma^2 to a per-ray map, mirroring
        # edge_semantic_map. softplus(+eps) in FP32 (geo_feat/tcnn can be fp16;
        # softplus on half can over/underflow and 1/sigma^2 / log(sigma^2) must be
        # finite). Guarded by `sigma2 is not None` so the off-path adds no ops and
        # the 7-tuple return is unchanged when the head is absent.
        sigma2_map = None
        if sigma2 is not None:
            sigma2 = torch.nn.functional.softplus(sigma2.float()) + 1e-6
            sigma2_map = torch.sum(weights[...,None] * sigma2, -2)
            # BUG-FIX (harden): the per-sample +1e-6 does NOT survive volume rendering — on
            # empty/low-density rays sum(weights)=acc~0 so sigma2_map -> ~0, making log(sigma2)
            # and 1/sigma2 in the NLL non-finite. Floor the PER-RAY map so both stay finite.
            sigma2_map = torch.clamp_min(sigma2_map, 1e-6)


        depth_map = torch.sum(weights * z_vals, -1)
        depth_var = torch.sum(weights * torch.square(z_vals - depth_map.unsqueeze(-1)), dim=-1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        if sigma2_map is not None:
            return rgb_map, disp_map, acc_map, weights, depth_map, depth_var, edge_semantic_map, sigma2_map
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
    
    def run_network(self, inputs, oracle_w=None, surf_w=None):
        """
        Run the network on a batch of inputs.

        Params:
            inputs: [N_rays, N_samples, 3]
            oracle_w: [N_rays] or [N_rays, 1] attribution weights in (0,1] for T1.3
                      oracle routing. When given, Δx is multiplied by w broadcast
                      over samples (routes deformation to w>0 regions + w-weights
                      the field's gradient). None = ungated (default, upstream).
        Returns:
            outputs: [N_rays, N_samples, 4]
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        def_reg = inputs_flat.new_zeros(())   # deformation-magnitude reg accumulator (0 if static/off)

        if self.config['dynamic']:
            pts = inputs_flat[:,:3]
            frame_time = inputs_flat[:,3].unsqueeze(-1)
            # PHASE 0 DIAGNOSIS HOOK (2026-06-05, workflow wx3zjzfyh):
            # deformation_off config flag forces Δx=0 in this forward pass,
            # turning the deformation field OFF without retraining.  Required
            # for Tests 0/2/5 (depth-floor, tool-cancellation, strain).
            # Default: False (normal SLAM behavior unchanged).
            if self.config.get('deformation_off', False):
                vox_motion = torch.zeros(pts.shape[0], 3, device=pts.device, dtype=pts.dtype)
            else:
                embed_time = self.embed_time(frame_time)
                embed_pos = self.embed_fre_pos(pts)
                h = torch.cat([embed_time,embed_pos],dim=-1)
                vox_motion = self.time_net(h)
                # HARD BOUND (physical prior): cap |Δx| to ~a couple mm via tanh so the field CANNOT
                # explode (removes the bistable explode mode by construction). deform_hardbound=b in model
                # units (e.g. 0.04). Bounds the RAW field output, before anchor + gate. Default 0 = upstream.
                _hb = self.config.get('deform_hardbound', 0)
                if _hb and _hb > 0:
                    vox_motion = _hb * torch.tanh(vox_motion / _hb)
                # frame-0 canonical anchor (Δx≡0 at t=0). deformation_anchor_off:true
                # disables it (revival experiment — lets t=0 deform too). Default: on.
                if not self.config.get('deformation_anchor_off', False):
                    vox_motion = torch.where(frame_time.reshape(-1, frame_time.shape[-1]) == 0, torch.zeros_like(vox_motion), vox_motion)
            # T1.3 ORACLE ROUTING: gate Δx by a per-ray attribution w in (0,1]
            # broadcast over samples. Restricts deformation to w>0 regions and
            # w-weights the field's gradient (chain rule). def_reg below then
            # regularises the GATED motion. None = ungated (default).
            if oracle_w is not None:
                ow = oracle_w.reshape(inputs.shape[0], 1, 1).expand(inputs.shape[0], inputs.shape[1], 1).reshape(-1, 1)
                vox_motion = vox_motion * ow
            # SURFACE/TRUNC BINDING: zero deformation away from the surface (surf_w per-sample in (0,1]),
            # killing the unconstrained off-surface field explosion (battery-6 raw 108). None = off.
            if surf_w is not None:
                vox_motion = vox_motion * surf_w.reshape(-1, 1)
            inputs_flat = pts + vox_motion
            def_reg = (vox_motion ** 2).mean()   # ||Δx||^2 magnitude (differentiable -> time_net)
        
        # Normalize the input to [0, 1] (TCNN convention)
        if self.config['grid']['tcnn_encoding']:
            inputs_flat = (inputs_flat - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        # ColorSDFNet_v2 ALWAYS returns a 4-tuple (sigma2_flat / geo_feat_flat are None when the
        # respective head/fusion is absent) — arity is constant => flags-off control flow is
        # byte-identical to base.
        outputs_flat, edge_semantic, sigma2_flat, geo_feat_flat = batchify(self.query_color_sdf, None)(inputs_flat)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        edge_semantic = torch.reshape(edge_semantic, list(inputs.shape[:-1]) + [edge_semantic.shape[-1]])
        # ARM-2 Inc-1: reshape sigma2 like edge_semantic; passthrough None when off.
        sigma2 = None
        if sigma2_flat is not None:
            sigma2 = torch.reshape(sigma2_flat, list(inputs.shape[:-1]) + [sigma2_flat.shape[-1]])
        # geo_feat SNI-fusion: reshape the per-sample geometry feature like sigma2; None unless _surface_geo.
        geo_feat = None
        if geo_feat_flat is not None:
            geo_feat = torch.reshape(geo_feat_flat, list(inputs.shape[:-1]) + [geo_feat_flat.shape[-1]])
        return outputs, edge_semantic, sigma2, geo_feat, def_reg

    def deform_teacher_loss(self, Xk, t, dx_target, w):
        '''ARM-2 Stage-1: supervise the deformation field DIRECTLY (the contribution the paper lacks).
        Regress the field output D(Xk,t) toward the baked self-supervised target Δx* (depth+DINO
        correspondence, observed->canonical pull-back), trust-weighted. Replicates the render-path field
        forward EXACTLY (run_network :204-218) so the supervised D is the SAME D the renderer uses:
        embed_time ⊕ embed_fre_pos -> time_net -> deform_hardbound -> t=0 anchor. No caller unless
        deformation_sup_weight>0 (default 0) => bit-identical to base.
        Params: Xk [N,3] world surface pts; t [N,1] frame_time; dx_target [N,3] Δx*; w [N,1] trust.'''
        embed_time = self.embed_time(t)
        embed_pos = self.embed_fre_pos(Xk)
        h = torch.cat([embed_time, embed_pos], dim=-1)
        D = self.time_net(h)
        _hb = self.config.get('deform_hardbound', 0)
        if _hb and _hb > 0:
            D = _hb * torch.tanh(D / _hb)
        if not self.config.get('deformation_anchor_off', False):
            D = torch.where(t.reshape(-1, t.shape[-1]) == 0, torch.zeros_like(D), D)
        w = w.detach()                                          # trust is a fixed weight, never a gradient path
        return (w * (D - dx_target.detach()) ** 2).sum() / w.sum().clamp_min(1.0)

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

    def render_rays(self, rays_o, rays_d, target_d=None, oracle_w=None):
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
        # PHASE 0 DIAGNOSIS GUARD (2026-06-05, workflow wx3zjzfyh):
        # rays_o[...,3] was previously read unconditionally — crashes when
        # the caller (e.g. infra/deform_off_render.py with dynamic=False) only
        # provides 3-column rays_o.  Guard by checking shape; default timestamp=0
        # if missing (the deformation network already masks frame_time==0 via
        # the `where` clause at run_network).  Documented in Addons/RESULTS_LOG.md:26.
        if rays_o.shape[-1] >= 4:
            timestamps = (rays_o[...,3]).reshape(-1,1)
        else:
            timestamps = torch.zeros(rays_o.shape[0], 1, device=rays_o.device, dtype=rays_o.dtype)
        timestamps = timestamps.repeat(1,pts.shape[1]).unsqueeze(-1)
        pts = torch.cat([pts,timestamps],dim=-1)
        # SURFACE/TRUNC BINDING (2026-06-14): restrict deformation to near the measured surface
        # (|z - target_d| within k*trunc), killing the unconstrained off-surface field explosion.
        # Flag deform_surface_bind=k (default 0=off ⇒ regression-safe). soft Gaussian falloff.
        surf_w = None
        _sb = self.config.get('deform_surface_bind', 0)
        if _sb and _sb > 0 and target_d is not None:
            _trunc_w = self.config['training']['trunc'] * self.config['data'].get('sc_factor', 1.0)
            surf_w = torch.exp(-((z_vals - target_d) / (_sb * _trunc_w + 1e-9)) ** 2)  # [N_rays, N_samples]
        raw, edge_semantic, sigma2, geo_feat, def_reg = self.run_network(pts, oracle_w=oracle_w, surf_w=surf_w)
        # ARM-2 Inc-1: raw2outputs returns an 8th element (sigma2_map) ONLY when
        # sigma2 is not None (uncertainty head on); otherwise the 7-tuple unpack
        # below is byte-identical to base.
        sigma2_map = None
        if sigma2 is not None:
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var, edge_semantic_map, sigma2_map = self.raw2outputs(raw, edge_semantic, z_vals, self.config['training']['white_bkgd'], sigma2=sigma2)
        else:
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var, edge_semantic_map = self.raw2outputs(raw, edge_semantic, z_vals, self.config['training']['white_bkgd'])
        # geo_feat SNI-fusion: volume-render the per-sample geometry feature to per-ray with the SAME
        # blend weights as rgb/sigma2; None unless the dino head fuses geo_feat (decoder._surface_geo).
        geo_feat_map = None
        if geo_feat is not None:
            geo_feat_map = torch.sum(weights[..., None] * geo_feat, -2)   # [N_rays, Cgeo]

        # Importance sampling
        if self.config['training']['n_importance'] > 0:

            rgb_map_0, disp_map_0, acc_map_0, depth_map_0, depth_var_0, edge_map_0,edge_semantic_map = rgb_map, disp_map, acc_map, depth_map, depth_var, edge_map,edge_semantic_map

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.config['training']['n_importance'], det=(self.config['training']['perturb']==0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            # NOTE: n_importance>0 branch is DEAD/BROKEN upstream (raw2outputs called
            # with wrong arity, edge_map undefined). Kept disabled via n_importance:0.
            # Unpack updated only to match run_network's new 4-tuple arity.
            raw, edge_semantic, sigma2, geo_feat, def_reg = self.run_network(pts, oracle_w=oracle_w)
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var, edge_map,edge_semantic_map = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])

        # Return rendering outputs
        ret = {
            'rgb' : rgb_map,
            'depth' :depth_map,
            'disp_map' : disp_map,
            'acc_map' : acc_map,
            'depth_var':depth_var,
            'edge_semantic':edge_semantic_map,
            'def_reg': def_reg
        }
        # ARM-2 Inc-1: per-ray sigma^2 map, only present when the head is on.
        if sigma2_map is not None:
            ret['sigma2'] = sigma2_map
        # geo_feat SNI-fusion: per-ray geometry feature, only present when the dino head fuses it.
        if geo_feat_map is not None:
            ret['geo_feat'] = geo_feat_map
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
    
    def forward(self, rays_o, rays_d, target_rgb, target_d, global_step=0,target_edge_semantic=None, border=None, notFirstMap=True, UseBorder=False,render_only=False, tracking=False, target_dino=None):
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
        # T1.3 oracle routing: use the per-ray seg edge-prior as the attribution
        # weight w. Training-time gradient mechanism only (disabled on render_only
        # to avoid the full-image vs ray-batch shape mismatch at the eval call).
        oracle_w = target_edge_semantic if (self.config.get('oracle_routing', False) and target_edge_semantic is not None and not render_only) else None
        rend_dict = self.render_rays(rays_o, rays_d, target_d=target_d, oracle_w=oracle_w)

        # Inc-1 v2 (mode:'dino'): per-PIXEL DINO uncertainty. The per-point geo head is ABSENT in dino
        # mode (run_network returns sigma2=None -> render_rays writes no 'sigma2'); instead derive
        # sigma2 from the gathered per-ray DINO feature via the decoder-owned head, writing the SAME
        # rend_dict['sigma2'] key the NLL (~line 543), the Inc-2 down-weight (~501) and the render viz
        # (~569) consume. softplus(.float())+1e-6 then floor mirrors raw2outputs. Placed BEFORE the
        # eval early-return so render_only also surfaces sigma2 for the viz. Gated on (target_dino present
        # AND head built) -> off/geo add no key/op -> byte-identical base (Inc-0).
        if target_dino is not None and hasattr(self.decoder, 'dino_unc_net'):
            # SNI-spirit fusion (uncertainty.fuse): '' = DINO only; 'rgbd' = [DINO ; rgb ; depth];
            # 'geo' = [DINO ; geo_feat] (the geometry jitter-killer); 'geo_rgbd' = [DINO ; geo_feat ; rgb ;
            # depth]. All extra modalities DETACHED so sigma^2 READS them but doesn't drive geometry/render
            # (stop-gradient, lit #2). Concat order [dino, geo, rgb, depth] matches the decoder in_dim
            # arithmetic (decoder.py:495). '' / 'rgbd' byte-identical to before.
            _fuse = self.config.get('uncertainty', {}).get('fuse', '')
            _parts = [target_dino]
            if _fuse in ('geo', 'geo_rgbd') and 'geo_feat' in rend_dict:
                _parts.append(rend_dict['geo_feat'].detach().reshape(target_dino.shape[0], -1))
            if _fuse in ('rgbd', 'geo_rgbd'):
                _parts.append(rend_dict['rgb'].detach().reshape(target_dino.shape[0], 3))
                _parts.append(rend_dict['depth'].detach().reshape(target_dino.shape[0], 1))
            _feat = torch.cat(_parts, dim=-1) if len(_parts) > 1 else target_dino
            _sig = torch.nn.functional.softplus(self.decoder.dino_unc_net(_feat).float()) + 1e-6
            rend_dict['sigma2'] = torch.clamp_min(_sig, 1e-6)

        if not self.training:
            return rend_dict
        
        # Get depth and rgb weights for loss
        valid_depth_mask = (target_d.squeeze() > 0.) * (target_d.squeeze() < self.config['cam']['depth_trunc'])
        rgb_weight = valid_depth_mask.clone().unsqueeze(-1)
        rgb_weight[rgb_weight==0] = self.config['training']['rgb_missing']

        # Get render loss
        if not render_only:

            # --- ARM-2 Inc-2: tracking-only pose down-weight from per-ray sigma^2.
            # w = clip(1/sigma^2, w_min, w_max).detach() is passed via the compute_loss
            # weights= kwarg (utils.py: linear loss*weights ONCE — NOT the rgb*w trick
            # which squares to 1/sigma^4). .detach() stops the pose optimiser from gaming
            # sigma^2. Fires ONLY when tracking AND the head is on, so mapping/BA and the
            # flags-off base path are byte-identical (rgb_unc_w / depth_unc_w stay None).
            rgb_unc_w = None
            depth_unc_w = None
            if tracking and getattr(self, 'unc_on', False) and ('sigma2' in rend_dict):
                _wmin = self.config.get('uncertainty', {}).get('w_min', 0.1)
                _wmax = self.config.get('uncertainty', {}).get('w_max', 10.0)
                _w_ray = torch.clamp(1.0 / rend_dict['sigma2'], _wmin, _wmax).detach()  # [N_rays,1]
                rgb_unc_w = _w_ray  # broadcasts over the 3 rgb channels
                depth_unc_w = _w_ray.squeeze()[valid_depth_mask]  # match masked depth shape

            rgb_loss = compute_loss(rend_dict["rgb"]*rgb_weight, target_rgb*rgb_weight, weights=rgb_unc_w)
            psnr = mse2psnr(rgb_loss)
            depth_loss = compute_loss(rend_dict["depth"].squeeze()[valid_depth_mask], target_d.squeeze()[valid_depth_mask], weights=depth_unc_w)

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

            # --- ARM-2 Inc-1: self-supervised aleatoric NLL on the photometric residual.
            # L_nll = mean( 0.5 * (rgb_err^2 / sigma^2 + log sigma^2) ). Trains the
            # uncertainty head on the SAME residual as rgb_loss (gradient flows into
            # sigma^2 AND the photometric path; the dominant rgb_loss anchors it).
            # Computed in fp32 over valid pixels. Stashed in ret['nll']; only ADDED to
            # the total loss in get_loss_from_ret behind nll_weight (default 0), so the
            # base path stays inert. Guarded by 'sigma2' so off-path adds no key.
            if ('sigma2' in rend_dict) and (not tracking):   # NLL trains the head on MAPPING residual ONLY
                _s2 = rend_dict['sigma2'].float()                            # [N_rays,1]
                # Arm-1 #1 (lit scan): configurable σ² TEACHER. 'l2' (DEFAULT, byte-identical to the
                # original) trains σ² on the raw photometric residual -> the σ∝‖C−Ĉ‖ degeneracy
                # (NeRF-On-the-go) = a contrast/edge detector (= the "uncertainty in the wrong place").
                # 'depth'/'rgb_depth' add a SCALE-INVARIANT depth-consistency residual (|d̂−d|/d)² so σ²
                # tracks GEOMETRIC/deformation uncertainty, not contrast. (WildGS/On-the-go use
                # SSIM+depth; SSIM needs spatial patches we don't sample -> depth-consistency is the
                # doable structural signal.) flag-gated default-off -> base/geo-l2 unchanged.
                _teacher = self.config.get('uncertainty', {}).get('teacher', 'l2')
                # Arm-1 #2 (lit scan): STOP-GRADIENT. The err^2/sigma^2 term back-props a 1/sigma^2-weighted
                # gradient into the PREDICTION (rgb/depth), so the model can lower the NLL by INFLATING
                # sigma^2 on hard pixels instead of fixing the prediction (beta-NLL "explain-away",
                # Seitzer ICLR22). Detaching the prediction in the residual decouples them: the NLL then
                # trains sigma^2 ONLY, and the prediction is shaped by its own undistorted rgb/depth loss
                # (the base already carries a separate full-gradient rgb_loss, so this is the clean fix
                # for our setup). flag-gated default-off (detach_residual False) -> byte-identical base.
                _detach = self.config.get('uncertainty', {}).get('detach_residual', False)
                _p_rgb = rend_dict['rgb'].detach() if _detach else rend_dict['rgb']
                if _teacher == 'l2':
                    _rgb_err2 = ((_p_rgb - target_rgb) ** 2).float()   # [N_rays,3] EXACT original when not detached
                    _nll = 0.5 * (_rgb_err2 / _s2 + torch.log(_s2))             # broadcast [N_rays,3]
                else:
                    _rgb_e2 = ((_p_rgb - target_rgb) ** 2).float().mean(-1, keepdim=True)  # [N,1]
                    _p_d = rend_dict['depth'].detach() if _detach else rend_dict['depth']
                    _d_hat = _p_d.float().reshape(-1, 1)
                    _d_gt = target_d.float().reshape(-1, 1)
                    _dep_e2 = ((_d_hat - _d_gt) / (_d_gt.abs() + 1e-6)) ** 2     # [N,1] scale-invariant
                    if _teacher == 'depth':       _err = _dep_e2
                    elif _teacher == 'rgb_depth': _err = _rgb_e2 + _dep_e2
                    else: raise ValueError(f"uncertainty.teacher {_teacher!r} not in l2/depth/rgb_depth")
                    _nll = 0.5 * (_err / _s2 + torch.log(_s2))                  # [N,1]
                nll_loss = _nll[valid_depth_mask].mean()

        else:
            rgb_loss = depth_loss = edge_loss = edge_semantic_loss = sdf_loss = fs_loss = psnr = None
            sdf_stats = None

        ret = {
            "rgb": rend_dict["rgb"],
            "depth": rend_dict["depth"],
            "edge_semantic": rend_dict["edge_semantic"],
            "def_reg": rend_dict["def_reg"],
            "rgb_loss": rgb_loss,
            "depth_loss": depth_loss,
            "edge_semantic_loss": edge_semantic_loss,
            "sdf_loss": sdf_loss,
            "fs_loss": fs_loss,
            "psnr": psnr,
            "sdf_stats": sdf_stats,
        }

        # --- ARM-2 Inc-1: surface the uncertainty map + NLL only when the head is on
        # (keys absent on the base path => get_loss_from_ret's .get() guard skips them).
        if 'sigma2' in rend_dict:
            ret['sigma2'] = rend_dict['sigma2']
            if (not render_only) and (not tracking):   # NLL is a MAPPING-only training signal, never on pose
                ret['nll'] = nll_loss

        return ret
