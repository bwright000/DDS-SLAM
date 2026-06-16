# Package imports
import torch
import torch.nn as nn
import tinycudann as tcnn

class TimeNet(nn.Module):
    def __init__(self, config, input_ch=3,
                hidden_dim_time=64, num_layers_time=3):
        super(TimeNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.hidden_dim_time = hidden_dim_time
        self.num_layers_time = num_layers_time

        self.model = self.get_model(config['decoder']['tcnn_network'])
    
    def forward(self, input_feat):
        # h = torch.cat([embedded_dirs, geo_feat], dim=-1)
        return self.model(input_feat)
    
    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('TIME net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim_time,
                    "n_hidden_layers": self.num_layers_time - 1,
                },
                #dtype=torch.float
            )

        time_net =  []
        for l in range(self.num_layers_time):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = self.hidden_dim_time
            
            if l == self.num_layers_time - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = self.hidden_dim_time
            
            time_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers_time - 1:
                time_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(time_net))

class ColorNet(nn.Module):
    def __init__(self, config, input_ch=4, geo_feat_dim=15, 
                hidden_dim_color=64, num_layers_color=3):
        super(ColorNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim_color = hidden_dim_color
        self.num_layers_color = num_layers_color

        self.model = self.get_model(config['decoder']['tcnn_network'])
    
    def forward(self, input_feat):
        # h = torch.cat([embedded_dirs, geo_feat], dim=-1)
        return self.model(input_feat)
    
    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('Color net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch + self.geo_feat_dim,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim_color,
                    "n_hidden_layers": self.num_layers_color - 1,
                },
                #dtype=torch.float
            )

        color_net =  []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = self.input_ch + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_color
            
            if l == self.num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = self.hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers_color - 1:
                color_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(color_net))

class EdgeNet(nn.Module):
    def __init__(self, config, input_ch=4, geo_feat_dim=15, 
                hidden_dim_color=64, num_layers_color=3):
        super(EdgeNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim_color = hidden_dim_color
        self.num_layers_color = num_layers_color

        self.model = self.get_model(config['decoder']['tcnn_network'])
    
    def forward(self, input_feat):
        return self.model(input_feat)
    
    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('Edge net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch + self.geo_feat_dim,
                n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim_color,
                    "n_hidden_layers": self.num_layers_color - 1,
                },
                #dtype=torch.float
            )

        edge_net =  []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = self.input_ch + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_color
            
            if l == self.num_layers_color - 1:
                out_dim = 1
            else:
                out_dim = self.hidden_dim_color
            
            edge_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers_color - 1:
                edge_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(edge_net))

class ColorEdgeNet(nn.Module):
    def __init__(self, config, input_ch=4, geo_feat_dim=15,
                 hidden_dim_color=64, num_layers_color=3):
        super(ColorEdgeNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim_color = hidden_dim_color
        self.num_layers_color = num_layers_color

        self.model = self.get_model(config['decoder']['tcnn_network'])

    def forward(self, input_feat):
        return self.model(input_feat)

    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('Color net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch + self.geo_feat_dim,
                n_output_dims=4,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim_color,
                    "n_hidden_layers": self.num_layers_color - 1,
                },
                # dtype=torch.float
            )

        color_edge_net = []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = self.input_ch + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_color

            if l == self.num_layers_color - 1:
                out_dim = 4  # 3 rgb & 1 edge
            else:
                out_dim = self.hidden_dim_color

            color_edge_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers_color - 1:
                color_edge_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(color_edge_net))

class SDFNet(nn.Module):
    def __init__(self, config, input_ch=3, geo_feat_dim=15, hidden_dim=64, num_layers=2):
        super(SDFNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.model = self.get_model(tcnn_network=config['decoder']['tcnn_network'])
    
    def forward(self, x, return_geo=True):
        out = self.model(x)

        if return_geo:  # return feature
            return out
        else:
            return out[..., :1]

    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('SDF net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch,
                n_output_dims=1 + self.geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.num_layers - 1,
                },
                #dtype=torch.float
            )
        else:
            sdf_net = []
            for l in range(self.num_layers):
                if l == 0:
                    in_dim = self.input_ch
                else:
                    in_dim = self.hidden_dim 
                
                if l == self.num_layers - 1:
                    out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
                else:
                    out_dim = self.hidden_dim 
                
                sdf_net.append(nn.Linear(in_dim, out_dim, bias=False))
                if l != self.num_layers - 1:
                    sdf_net.append(nn.ReLU(inplace=True))

            return nn.Sequential(*nn.ModuleList(sdf_net))

class EdgeNet_Semantic(nn.Module): 
    def __init__(self, config, input_ch=4, geo_feat_dim=15,
                 hidden_dim_color=64, num_layers_color=3):
        super(EdgeNet_Semantic, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim_color = hidden_dim_color
        self.num_layers_color = num_layers_color

        self.model = self.get_model(config['decoder']['tcnn_network'])

    def forward(self, input_feat):
        # h = torch.cat([embedded_dirs, geo_feat], dim=-1)
        return self.model(input_feat)

    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('Edge net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch + self.geo_feat_dim,
                n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim_color,
                    "n_hidden_layers": self.num_layers_color - 1,
                },
                # dtype=torch.float
            )

        edge_net = []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = self.input_ch + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_color

            if l == self.num_layers_color - 1:
                out_dim = 1
            else:
                out_dim = self.hidden_dim_color

            edge_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers_color - 1:
                edge_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(edge_net))


class UncertaintyDINONet(nn.Module):
    """Inc-1 v2 — per-PIXEL aleatoric sigma^2 from a C-dim DINO image feature.

    Faithful reimplementation of WildGS-SLAM's uncertainty MLP
    (WildGS-SLAM/src/utils/dyn_uncertainty/uncertainty_model.py:5-67):
      * 2 dense layers in_dim->hidden->hidden, ReLU between (their net_depth=2, :14-25),
      * he_uniform / kaiming_uniform init (:18-19,29),
      * a final hidden->1 output layer (:28),
      * softplus to make the output positive (:33,59).

    DELIBERATE DIVERGENCES (each justified, each an A/B knob):
      1. softplus + 1e-6 FLOOR is applied at the scene_rep call site, NOT here — this net returns
         the RAW pre-softplus logit. That mirrors v1's raw->softplus split (raw2outputs:140) so the
         geo-vs-dino A/B isolates the FEATURE SOURCE, not the floor constant. (WildGS floors with
         clip(.,min=0.1)+1e-3 on sigma; that is a separate one-line A/B knob, not folded in here.)
      2. dropout default 0 (deterministic est_c2w + Inc-0 bit-identity) vs WildGS's ALWAYS-ON p=0.2
         (uncertainty_model.py:55). Exposed as uncertainty.dino_dropout for opt-in MC-dropout.
    SUBSTRATE DIVERGENCE: WildGS is per-pixel Gaussian-splat; we feed the gathered per-RAY DINO
    feature on a neural-SDF and reuse the EXISTING Inc-1 NLL + Inc-2 down-weight verbatim.
    """
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.dropout = float(dropout)
        layers = []
        for l in range(num_layers):
            lin = nn.Linear(in_dim if l == 0 else hidden_dim, hidden_dim)  # bias ON (WildGS)
            nn.init.kaiming_uniform_(lin.weight, nonlinearity='relu')      # WildGS he_uniform :19
            nn.init.zeros_(lin.bias)
            layers.append(lin)
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, 1)                       # WildGS :28
        nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, feat):            # feat: [N, C]  ->  [N, 1] RAW (pre-softplus; floored at call site)
        x = feat
        for lin in self.layers:
            x = torch.nn.functional.relu(lin(x))                          # WildGS :53-54
            if self.dropout > 0:
                x = torch.nn.functional.dropout(x, p=self.dropout, training=True)  # WildGS always-on :55
        return self.output_layer(x)


class ColorSDFNet(nn.Module):
    '''
    Color grid + SDF grid
    '''
    def __init__(self, config, input_ch=3, input_ch_pos=12):
        super(ColorSDFNet, self).__init__()
        self.config = config
        self.color_net = ColorNet(config,
                input_ch=input_ch+input_ch_pos, 
                geo_feat_dim=config['decoder']['geo_feat_dim'], 
                hidden_dim_color=config['decoder']['hidden_dim_color'], 
                num_layers_color=config['decoder']['num_layers_color'])
        self.edge_net = EdgeNet(config,
                input_ch=input_ch+input_ch_pos,
                geo_feat_dim=config['decoder']['geo_feat_dim'],
                hidden_dim=config['decoder']['hidden_dim'], 
                num_layers=config['decoder']['num_layers'])
        self.sdf_net = SDFNet(config,
                input_ch=input_ch+input_ch_pos,
                geo_feat_dim=config['decoder']['geo_feat_dim'],
                hidden_dim=config['decoder']['hidden_dim'], 
                num_layers=config['decoder']['num_layers'])
            
    def forward(self, embed, embed_pos, embed_color):

        if embed_pos is not None:
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1), return_geo=True) 
        else:
            h = self.sdf_net(embed, return_geo=True) 
        
        sdf, geo_feat = h[...,:1], h[...,1:]
        if embed_pos is not None:
            rgb = self.color_net(torch.cat([embed_pos, embed_color, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([embed_color, geo_feat], dim=-1))

        if embed_pos is not None:
            edge = self.edge_net(torch.cat([embed_pos, embed_color, geo_feat], dim=-1))
        else:
            edge = self.edge_net(torch.cat([embed_color, geo_feat], dim=-1))
        
        return torch.cat([rgb, edge, sdf], -1)
    
class ColorSDFNet_v1(nn.Module):
    '''
    No color grid
    '''
    def __init__(self, config, input_ch=3, input_ch_pos=12):
        super(ColorSDFNet_v1, self).__init__()
        self.config = config
        self.color_net = ColorNet(config,
                input_ch=input_ch_pos, 
                geo_feat_dim=config['decoder']['geo_feat_dim'], 
                hidden_dim_color=config['decoder']['hidden_dim_color'], 
                num_layers_color=config['decoder']['num_layers_color'])
        self.edge_net = EdgeNet(config,
                input_ch=input_ch_pos, 
                geo_feat_dim=config['decoder']['geo_feat_dim'], 
                hidden_dim_color=config['decoder']['hidden_dim_color'], 
                num_layers_color=config['decoder']['num_layers_color'])
        self.sdf_net = SDFNet(config,
                input_ch=input_ch+input_ch_pos,
                geo_feat_dim=config['decoder']['geo_feat_dim'],
                hidden_dim=config['decoder']['hidden_dim'], 
                num_layers=config['decoder']['num_layers'])
            
    def forward(self, embed, embed_pos):

        if embed_pos is not None:
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1), return_geo=True) 
        else:
            h = self.sdf_net(embed, return_geo=True) 
        
        sdf, geo_feat = h[...,:1], h[...,1:]
        if embed_pos is not None:
            rgb = self.color_net(torch.cat([embed_pos, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([geo_feat], dim=-1))
        
        if embed_pos is not None:
            edge = self.edge_net(torch.cat([embed_pos,geo_feat], dim=-1))
        else:
            edge = self.edge_net(torch.cat([geo_feat], dim=-1))
        
        return torch.cat([rgb, edge, sdf], -1)

class ColorSDFNet_v2(nn.Module):
    '''
    No color grid
    '''

    def __init__(self, config, input_ch=3, input_ch_pos=12,input_ch_time=10,input_ch_fre=10):
        super(ColorSDFNet_v2, self).__init__()
        self.config = config
        self.color_net = ColorNet(config,
                                  input_ch=input_ch_pos,
                                  geo_feat_dim=config['decoder']['geo_feat_dim'],
                                  hidden_dim_color=config['decoder']['hidden_dim_color'],
                                  num_layers_color=config['decoder']['num_layers_color'])
        
        self.time_net = TimeNet(config,
            input_ch=input_ch_time+input_ch_fre,
            hidden_dim_time=config['decoder']['hidden_dim_time'], 
            num_layers_time=config['decoder']['num_layers_time'])

        self.edgenet_semantic = EdgeNet_Semantic(config,
                                input_ch=input_ch_pos,
                                geo_feat_dim=config['decoder']['geo_feat_dim'],
                                hidden_dim_color=config['decoder']['hidden_dim_color'],
                                num_layers_color=config['decoder']['num_layers_color'])

        self.sdf_net = SDFNet(config,
                              input_ch=input_ch + input_ch_pos,
                              geo_feat_dim=config['decoder']['geo_feat_dim'],
                              hidden_dim=config['decoder']['hidden_dim'],
                              num_layers=config['decoder']['num_layers'])

        # --- ARM-2 Inc-1: per-ray aleatoric uncertainty head (sigma^2). Mirrors edgenet_semantic 1:1.
        # CONSTRUCTED ONLY when uncertainty.enable, and CONSTRUCTED LAST (after ALL base modules:
        # color/time/edge/sdf) so the ON run's base backbone draws the SAME RNG as base -> base-vs-uncert
        # is a CLEAN single-variable A/B (only the head differs; fixes the audit's init-parity confound).
        # Off-path: no module built -> bit-identical base -> Inc-0 harness PASSES.
        # v2 (mode:'dino') does NOT build this per-point head — it uses the per-PIXEL DINO head on
        # scene_rep instead (UncertaintyDINONet). So geo+off keep the v1 head exactly; dino skips it.
        if config.get('uncertainty', {}).get('enable', False) \
                and config.get('uncertainty', {}).get('mode', 'geo') == 'geo':
            self.uncertainty_net = EdgeNet_Semantic(config,
                                input_ch=input_ch_pos,
                                geo_feat_dim=config['decoder']['geo_feat_dim'],
                                hidden_dim_color=config.get('uncertainty', {}).get('hidden_dim', 32),
                                num_layers_color=config.get('uncertainty', {}).get('num_layers', 2))

    def forward(self, embed, embed_pos):

        if embed_pos is not None:
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1), return_geo=True)
        else:
            h = self.sdf_net(embed, return_geo=True)

        sdf, geo_feat = h[..., :1], h[..., 1:]
        if embed_pos is not None:
            rgb = self.color_net(torch.cat([embed_pos, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([geo_feat], dim=-1))

        if embed_pos is not None:
            edge_semantic = self.edgenet_semantic(torch.cat([embed_pos, geo_feat], dim=-1))
        else:
            edge_semantic = self.edgenet_semantic(torch.cat([geo_feat], dim=-1))

        # --- ARM-2 Inc-1: per-sample raw uncertainty (mirrors edge_semantic input).
        # ALWAYS-3-tuple with sigma2_raw=None when the head is absent: keeps the
        # return ARITY constant (control flow byte-stable) so flags-off stays
        # bit-identical to base while producing zero new tensors/ops when off.
        # softplus(+eps) in fp32 is applied later in scene_rep.raw2outputs (the
        # mirror site of the edge_semantic sigmoid), NOT here.
        sigma2_raw = None
        if hasattr(self, 'uncertainty_net'):
            if embed_pos is not None:
                sigma2_raw = self.uncertainty_net(torch.cat([embed_pos, geo_feat], dim=-1))
            else:
                sigma2_raw = self.uncertainty_net(torch.cat([geo_feat], dim=-1))

        return torch.cat([rgb,sdf], -1), edge_semantic, sigma2_raw