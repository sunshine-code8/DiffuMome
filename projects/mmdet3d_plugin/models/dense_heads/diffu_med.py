# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import collections
import copy

import math
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean, build_bbox_coder
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.models import builder
from torch.nn import functional as F
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
import matplotlib.pyplot as plt
import os

REF_CLAMP_EPS = 1e-4


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    if len(t.shape) > 1:
        t = t.squeeze()
    out = a.gather(-1, t.squeeze())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def pos2embed(pos, num_pos_feats=128, temperature=10000):  ### 濞达絽绉堕悿鍡欑磽閺嶎偆鍨?  婵繐绲肩紞鎴濐嚕閿旇法绉寸紓鍐惧枤缁鳖亪鎯?
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


@HEADS.register_module()
class DiffuMultiExpertDecoding(BaseModule):   ### 閻庤鐭粻鐔肺涢埀顒€霉鐎ｎ亗浠?
    def __init__(self,
                 in_channels,
                 num_query=900,   ### 闁稿﹥鐟╅埀顒€顦卞ú浼村冀閸ャ剱顐ｆ媴?
                 hidden_dim=128,
                 depth_num=64,
                 norm_bbox=True,
                 downsample_scale=8,
                 pc_range=[],   ### 闁绘劙鈧稓闅樼紒灞炬そ濡潡鎳犻崘銊︾函
                 modalities=dict(),
                 scalar=10,   ### 閻炴稏鍔庨妵姘跺储鐠囧弶鐝?query 闁汇劌瀚晶鎸庢櫠閻愭祴鍋撳鍥ц姵闁挎稑鏈崹銊╂嚀閸涱収妲婚柛?GT 闁汇垻鍠愰崹?DN queries 闁汇劌瀚划宥夊极閻楀牆浠橀柛鎺曟硾瀵剟寮?
                 noise_scale=1.0,
                 noise_trans=0.0,   ### 闁革綆浜滈敍鎰扮嵁瀹曞泦?
                 dn_weight=1.0,   ## 闁告顕у▍鏃堝箲閻旀灚浜奸柡澶婂暣閸?
                 split=0.75,
                 train_cfg=None,
                 test_cfg=None,
                 common_heads=dict(
                     center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
                 ),   ### head_name = (閺夊牊鎸搁崵顓犵磼閺夋垵顔? head閻忕偛鍊归弳?
                 tasks=[
                     dict(num_class=1, class_names=['car']),
                     dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                     dict(num_class=2, class_names=['bus', 'trailer']),
                     dict(num_class=1, class_names=['barrier']),
                     dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                     dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
                 ],
                 transformer=None,
                 bbox_coder=None,
                 loss_cls=dict(   ## 闁告帒妫涚悮顐﹀箲閻旀灚浜?
                     type="FocalLoss",
                     use_sigmoid=True,
                     reduction="mean",
                     gamma=2, alpha=0.25, loss_weight=1.0
                 ),
                 loss_bbox=dict(  # 婵℃妫楀ú鏍亹閹烘挸鐤?
                     type="L1Loss",
                     reduction="mean",
                     loss_weight=0.25,
                 ),
                 separate_head=dict(
                     type='SeparateMlpHead', init_bias=-2.19, final_kernel=3),
                 init_cfg=None,
                 **kwargs):
        timesteps = kwargs.pop('timesteps', 100)
        sampling_steps = kwargs.pop('sampling_steps', 3)
        diffusion_scale = kwargs.pop('diffusion_scale', 2.0)
        test_noise_seed = kwargs.pop('test_noise_seed', 0)
        box_renewal = kwargs.pop('box_renewal', True)
        use_ensemble = kwargs.pop('use_ensemble', True)
        diffuse_query_content = kwargs.pop('diffuse_query_content', True)
        use_feature_proposal_init = kwargs.pop('use_feature_proposal_init', True)
        proposal_init_mode = kwargs.pop('proposal_init_mode', 'fused')
        proposal_loss_weight = kwargs.pop('proposal_loss_weight', 1.0)
        assert init_cfg is None
        super(DiffuMultiExpertDecoding, self).__init__(init_cfg=init_cfg)
        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.hidden_dim = hidden_dim
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_query = num_query
        self.in_channels = in_channels
        self.depth_num = depth_num  ### 婵烇絽宕€瑰磭绮嬬紒妯绘疇闁告牗鐗滃▓?bin 闁?  閻㈩垰鎽滈弫銈夊捶?2D --> 3D闁靛棔绔籈V闁汇劌瀚换浣规償閿曗偓缂傛挸螣? 闁?婵烇絽宕€规娊寮悷鐗堝€婚柣銊ュ椤洭寮敐鍛€婚弶鍫涘妿瀹?
        self.norm_bbox = norm_bbox
        self.downsample_scale = downsample_scale
        ### 闁告顕у▍鏃傛媼椤撶姷鐭婇柛娆忓€归弳?
        self.pc_range = pc_range  # 闁绘劙鈧稓闅橀柟鎵枔閻擄繝鎳犻崘銊︾函
        self.modalities = modalities
        self.scalar = scalar   # DN query 闁汇劌瀚伴崳鍛婂緞瀹ュ洨鐭嬮柡渚€顣︾粭鍌炴⒔?
        self.bbox_noise_scale = noise_scale  # 閻庨潧婀卞﹢锟犲磹閸忕⒈鏀卞☉鎿冨幖缁洪箖宕濋悩鍙夌彣濠㈣澹嗗▓鎴犵磽閳哄倹鏉圭紒顖滅帛閺?
        self.bbox_noise_trans = noise_trans  # 闁革綆浜滈敍鎰板磻韫囨泤鈺呮煂?
        self.dn_weight = dn_weight   # DN 闁瑰湱鍠庨妵鎴︽儍閸曨剚缍€闂?
        self.split = split    # 闁革綆浜滈敍鎰扮嵁閸涱厼顔婇梻鍐ㄧ墕閳ь剛銆嬬槐婵堟惥閸涙壆绠栭柛鎺撶懄閻栵絿鎷嬫０浣界閻犳劗鍠愰悧閬嶅嫉?闁挎稓鍣︾槐鐢告晬閻曞倻鍚归柨?
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_steps
        self.ddim_sampling_eta = 0.0
        self.diffusion_scale = diffusion_scale
        self.test_noise_seed = test_noise_seed
        self.box_renewal = box_renewal
        self.use_ensemble = use_ensemble
        self.diffuse_query_content = diffuse_query_content
        self.use_feature_proposal_init = use_feature_proposal_init
        self.proposal_init_mode = proposal_init_mode
        self.proposal_loss_weight = proposal_loss_weight

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)  ### 閻?
        self.fp16_enabled = False

        self.shared_conv = ConvModule(
            in_channels,
            hidden_dim,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=dict(type="BN2d")
        )  # 闁硅泛锕埀顒佸哺娴滅偓绂?in_channels 闁哄嫮濮撮惃鐘诲礆?hidden_dim

        # transformer
        self.transformer = build_transformer(transformer)
        # 闁告瑥鍊介埀顒€鍟伴崑?
        self.reference_points = nn.Embedding(num_query, 3)
        self.total_num_classes = sum(self.num_classes)
        self.proposal_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.total_num_classes),
        )
        self.proposal_output = nn.Linear(hidden_dim, hidden_dim)
        self.proposal_output_norm = nn.LayerNorm(hidden_dim)
        self.proposal_ref = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.label_enc = nn.Embedding(self.total_num_classes + 1, hidden_dim)
        self.query_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 3 // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder_ref_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 10),
            )
            for _ in range(transformer.decoder.num_layers)
        ])
        self.bev_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rv_embedding = nn.Sequential(
            nn.Linear(self.depth_num * 3, self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        # task head
        self.task_heads = nn.ModuleList()
        for num_cls in self.num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(cls_logits=(num_cls, 2)))
            separate_head.update(
                in_channels=hidden_dim,
                heads=heads, num_cls=num_cls,
                groups=transformer.decoder.num_layers
            )
            self.task_heads.append(builder.build_head(separate_head))

        # assigner
        if train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self._criterion = nn.CrossEntropyLoss()
        self.register_diffusion_schedule(timesteps)

    def init_weights(self):
        super(DiffuMultiExpertDecoding, self).init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

    @property
    def coords_bev(self):
        #### 闁汇垻鍠愰崹?BEV 闁绘鎳撶欢娑㈠炊閸欍儳鐟愭慨锝呯箣闁叉粎绱旈幋鐐靛濞戞搩鍘肩缓楣冩倷閸︻厽鐣辩憸鐗堝笂缁旀挳宕?2D 闁秆勫姈閻?
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        ### 閻犱緤绱曢悾?BEV 闁绘鎳撶欢娑㈠炊閸撗勭暠閻庡湱鍋ゅ顖滀焊閸濆嫷鍤?
        x_size, y_size = (
            cfg['grid_size'][1] // self.downsample_scale,    ## grid[1]闁哄嫮妞?
            cfg['grid_size'][0] // self.downsample_scale     ## grid[0]闁哄嫮妞?
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]   ### 閻庤鐭粻鐔哥▔閵堝嫰鍤嬮弶鐐差嚟濞堟垿宕ｉ崒娑欐闁挎稒淇洪幑锝夋倷楠炲簱鍋撴担铏圭煉闁绘劕绠嶉埀顑跨窔閸ｄ即寮芥搴′化闁?
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        ### 鐟滅増甯婄粩鎾礌閺嵮冪厒 [0, 1]闁挎稑鑻懟鐔煎磻韫囨泤鈺呭礆閻楀牏妲ㄥ☉鎿冧簽缂嶅寮介懖鈺傜暠濞戞搩鍘肩缓?
        # 濞戞捁妗ㄧ划鍫熺▕?+0.5闁挎稓鍠庡ú婊勭▔閸濆嫭缍忛柡宥呮搐缁ㄨ尙鎷犻妷锕€鐦归柛姘灩缂嶅寮介懖鈺傜暠濞戞搩鍘肩缓楣冩嚀鐏炶偐鐟濋柡鍕靛灠娑斿繑绋夋繝鍜佹健
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        ## 闁哄牃鍋撶紓浣哥墣缁额參宕欓崫鍕Ш濞戞挴鍋撻柛鏍ㄧ墱濞?2D 闁秆勫姈閻栵綁鏁嶇仦鍓фЖ濞戞搩浜滈顔芥償?BEV 闁绘鎳撶欢娑㈠炊閸撗勭暠濞戞挴鍋撳☉鎿冧簻閸庢氨妲愰悩杈╃Т缂?
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0)  # (H*W, 2)
        return coord_base

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        ### 濞戞挸鎼獮鎾诲闯椤忓浂鍞茬紓?(Denoising Training) 闁告垵妫楅ˇ顒勫箥閳ь剟寮垫径瀣闁硅鍣槐鐧塏 queries闁靛棔宸tention mask闁靛棔妞掓禍鎺楀矗婵犲嫭鏆忓ù婊冩唉椤撳摜绮诲Δ浣哥柈濠㈡儼浜▓?mask_dict
        if self.training:
            targets = [
                torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),
                          dim=1) for img_meta in img_metas]       ### 濞寸姴瀛╅惁鈩冪▔椤忓懐澹夐柡鍫墰濞?img_metas 濞戞搩鍘借ぐ渚€宕ｉ弽顐ｅ焸闁稿﹤鍚嬮、瀣Υ娣囩幎avity_center 闁哄嫷鍨堕崳鎼佸礉濞戞鍘煫?(x,y,z)闁挎稑顔揺nsor[:, 3:] 闁?(w,l,h,yaw,vx,vy)
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas]  ## 闁圭粯鍔曡ぐ鍥冀閸モ晩鍔?
            targets, labels = self._sanitize_gt_lists(targets, labels, reference_points.device)
            if sum(t.size(0) for t in targets) == 0:
                if reference_points.dim() == 2:
                    reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
                return reference_points, None, None
            ### 闁汇垻鍠愰崹姘跺礂?1 闁汇劌瀚敮娲儘娓氬﹦绀夐悶娑栧妿閵?闁圭鍋撻柡鍫濐槺濠€锟犲磹閸忕⒈鏀遍梺顔挎瀵剚绋?DN"闁挎稑鐗婇惀鍛村嫉婢舵劒绱曢柦鍕灊閹广垺鎷呴弴鐕佹敱闁?
            known = [torch.ones_like(t, device=reference_points.device) for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]  # 婵絽绻戞竟鎺戔枎閳╁啰澹夐柡鍫墯濠€浣瑰緞濮橆剛姣岄柡浣哄瀹?
            labels = torch.cat([t.to(reference_points.device) for t in labels])
            boxes = torch.cat([t.to(reference_points.device) for t in targets])
            batch_idx = torch.cat([
                torch.full((t.size(0),), i, dtype=torch.long, device=reference_points.device)
                for i, t in enumerate(targets)
            ])  # 闁哄秴娲╅鍥掕箛搴ㄥ殝婵℃妫楅惈妯荤鎼粹剝鎲垮☉鎿冧簼閻楅亶寮?
            known_indice = torch.nonzero(unmask_label + unmask_bbox)   # 閻忓繐妫欏﹢顓犱沪韫囨碍鏉稿ù锝呯Ф閻ゅ棝鎯冮崟顓炲亶鐎殿喗娲栬ぐ鍥礄?
            known_indice = known_indice.view(-1)
            # add noise  闂佹彃绉撮ˇ?groups 婵?闁?known_labels_raw 闁哄嫷鍨粭澶愬礉閻樿鲸鍙忛柡鈧崷顓熺暠闁告鍠庨～鎰板冀閸モ晩鍔柛鎿冨灡濠€浼存晬鐏炶姤鍊甸梻鍫涘灱椤撳摜绮?DN 闁瑰湱鍠庨妵鎴﹀籍閸洘浠橀悷鏇氳兌閺?
            # 闂佹彃娲﹂悧杈╂惥婵犲偆妯嬮柨娑樼焷椤╊偊鎯勯弽顒傂ㄩ柛蹇嬪姂濞间即鏁嶇仦鐏镐線宕圭€ｎ剚鐣遍柛妯款嚙濞呮棃鎳楅挊澶婎潝閻℃帒锕ゅ?
            groups = min(self.scalar, self.num_query // max(known_num))
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_labels_raw = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)  ## 閻庡湱鍋ゅ顖滀沪閻愭壆鑹鹃柛婵愪簷闁叉競atch
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            # 闁告梻濮村▍鏃€绔? 闂佽棄鐗嗛顕€骞嶉埀顒勫嫉婢跺本鐣辨俊顐熷亾婵炴潙顑嗛、瀣焾閸婄喓绠婚悶娑樿嫰闁解晝绮?
            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans  # 閻犱緤绱曢悾濠氬闯椤忓嫸绱ｉ柣銊ュ濞撹埖寰勮娴滃摜绮旈弰蹇撶槺闁?= 婵℃妫楅弰鍌溾偓闈涙憸濞堟垶绋夐埀顒勫础?+ 闁稿绻掍簺闂?
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0  #闁汇垻鍠愰崹?[-1, 1] 濞戞柨顑夊Λ鍧楁儍閸曨偅缍嗛柛鏍ゅ亾闂傚懎绻戝┃鈧柡浣稿簻缁?0 濞?DN query 闁告艾瀚粭澶愭儎缁嬫寧鍊?
                known_bbox_center += torch.mul(rand_prob,
                                               diff) * self.bbox_noise_scale    ## 婵烇綀顕ф慨鐐哄闯椤忓嫸绱?
                ### 鐟滅増甯婄粩鎾礌閺嵮冪厒 [0, 1]闁挎稑濂旂粭?reference_points 闁汇劌瀚€垫牠宕剁紙鐘殿伇闁?
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (
                        self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (
                        self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (
                        self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                ### 闁革綆浜滈敍鎰緞椤忓嫨浜ｉ柣銊ュ閻栵絿鎷嬫０浣界閻犳劗鍠愰悧閬嶅嫉?
                # 闁革綆浜滈敍鎰板礉閻樿尙绻佸棰濅簻閵囧洭鏁嶇仦鎴掔剨缂佸倽宕靛﹢锟犲磹閻撳簺浜伴弶鈺傜玻缁辨繃娼诲▎鎴綒 query 閹煎瓨妫侀姘辨偖椤愩垻绉煎ù?閺夆晜鐟╅崳閿嬬閳ь剚绋婇崼銉ュ幋婵炲备鍓濆﹢?闁哄鍎撮鍕磼?
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = sum(self.num_classes)

            # Padding 闁告粌鑻顒勬嚀閸愵亜浠柟宄板悑鐢?
            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            # 閻?30 濞?DN 濞达絽绉堕悿鍡涙晬閸繂鐏ュ┑顔碱儎鐠愮喖姊跨拋鍦闁瑰嘲鍚嬬敮鎾捶?900 濞戞搩浜濋婊呮暜缁嬪灝妫橀柤鏉垮暟閸嬶綁宕滃澶嬫〃
            padding_bbox = torch.zeros(batch_size, pad_size, 3, device=reference_points.device)
            if reference_points.dim() == 2:
                base_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                base_reference_points = reference_points
            padded_reference_points = torch.cat([padding_bbox, base_reference_points], dim=1)
            #濠靛鍋勯崢?DN 闁告瑥鍊介埀顒€鍟伴崑锝夊锤閹邦厾鍨?
            # 闁哄瀚紓鎾诲及閻樿尙娈哥紒渚垮灩缁扁晠鍨鹃弬琛″亾閺傛寧鍟為悹鍥ь槹閻︹剝绋?DN query 閹煎瓨妫侀姘跺绩閹勮含 padded_reference_points 闁汇劌瀚幗銏＄▔椤忓啰绉寸紓?
            if len(known_num):
                map_known_indice = torch.cat([
                    torch.arange(num, dtype=torch.long, device=reference_points.device)
                    for num in known_num
                ])
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(groups)]).long()  # 闁革负鍔岄崣蹇曚沪閳ь剟鎯冮崟顏嗙Т缂傚啰鎽礜
            # 闁?(batch_idx, position_idx) 缂佷究鍨圭槐鈺呮晬鐏炴儳惟闁告梻濮崇花锟犲闯椤忓嫸绱ｉ柣銊ュ濠€锟犲磹闂傜鍘煫鍥у暙濞兼寮介崶褝缍栭柛蹇嬪劚椤曨喗鎯旈弬鍓хТ缂?
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(
                    reference_points.device)
                # padding 濞达絽绉堕悿鍡涘绩閸撗勭暠闁哄嫷鍨版慨鐐哄闯椤忓嫸绱ｉ柣銊ュ濞兼寮介崶顭戞敱

            # 闁哄瀚伴埀?Attention Mask
            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size, device=reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            # # 濞戞挸绉撮幃?DN 缂備礁瀚粻锝夋⒒缂堢姷闉嶅☉鎾崇Т瑜拌尙鎲?
            for i in range(groups):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == groups - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': known_indice.long(),
                'batch_idx': batch_idx.long(),
                'known_bid': known_bid.long(),
                'map_known_indice': map_known_indice.long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'known_labels_raw': known_labels_raw,
                'know_idx': know_idx,
                'pad_size': pad_size
            }

        else:
            if reference_points.dim() == 2:
                padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                padded_reference_points = reference_points
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _rv_pe(self, img_feats, img_metas):
        # 闁搞儲鍎抽崕姘舵偋閻熸壆绐欓柣?RV 濞达絽绉堕悿鍡欑磽閺嶎偆鍨?缂備焦鐟ュù姗€宕撹箛鏇烆棗鐎甸绀佸ù姗€鎯冮崟顒傛Ж濞戞搩浜滈崕姘辨閻樺灚鏅搁柟瀛樺姃缁斿瓨绋?3D 缂佸本妞藉Λ鎸庢媴瀹ュ洨鏋傜紓鍌涚墱閻?
        BN, C, H, W = img_feats.shape
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]   # 闁兼儳鍢茶ぐ?padding 闁告艾娴峰▓鎴﹀储閻斿娼楅柛銉﹀劤閸庢氨浜搁崫鍕靛殶
        ### 閻忓繐妫涙竟鎺戭嚗娴ｅ憡绂堥柛褎鍔栭悥锝夊及閻樿尙娈搁柛銉у仜鐢偅鎱ㄧ€ｎ亝绂堥柛宥呯箰閸庢氨妲愰悩鍙夌稄闁?
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W
        ### 闁汇垻鍠愰崹?64 濞戞搩浜濈换浣规償閿曞倸娅氶柡宥呭槻閳?
        coords_d = 1 + torch.arange(self.depth_num, device=img_feats[0].device).float() * (
                self.pc_range[3] - 1) / self.depth_num
        coords_h, coords_w, coords_d = torch.meshgrid([coords_h, coords_w, coords_d])  # 闁汇垻鍠愰崹?3D 缂傚啯鍨堕悧鎼佹晬濮橆厾妲ㄥ☉鎿冧簻閸庢氨妲?閼?婵絽绻嬮柌婊兦庨崡鐐差唺 = 濞戞挴鍋撳☉鎿冧邯閸ｄ即寮芥搴′化

        coords = torch.stack([coords_w, coords_h, coords_d, coords_h.new_ones(coords_h.shape)], dim=-1)  # 婵絽绻嬮柌婊堟倷? (u, v, d, 1) 闁汇劌瀚扮紞鍫濃枎閳ヨ櫕缍忛柡?
        coords[..., :2] = coords[..., :2] * coords[..., 2:3]   # 閻熸洑绀侀崢娑㈠箮?(u, v) 濞戞梹眉娴滄帒菐閸楃偛顔?d闁挎稑鏈晶鐘绘嚄閻ｅ本鏆忛梺顐㈡婵洩銇愭潏鈺冨彁闂傚啴娼у浠嬪箮閺囩偛顨涢柛?3D 缂佸本妞藉Λ?

        ### 閻犱緤绱曢悾濠氭儎閸涘﹥绨氶柍顐ｅ⒍iDAR 闁汇劌瀚伴埀顒€妫欐慨鍥亹鏉堚晝鍙愰梻?
        imgs2lidars = np.concatenate([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(coords.device)
        # 闁哄秶顭堢缓楣冨箼瀹ュ嫮绋婇柨娑欒壘閻ㄣ垽宕撹箛鏇狀槺闁秆勫姈閻栵綁宕ｅ鍡楊潓鐟滄澘宕崺?LiDAR 3D 闁秆勫姈閻栵絿鍖?
        coords_3d = torch.einsum('hwdo, bco -> bhwdc', coords, imgs2lidars)
        # 鐟滅増甯婄粩鎾礌閺嵮冪厒 [0, 1]
        coords_3d = (coords_3d[..., :3] - coords_3d.new_tensor(self.pc_range[:3])[None, None, None, :]) \
                    / (coords_3d.new_tensor(self.pc_range[3:]) - coords_3d.new_tensor(self.pc_range[:3]))[None, None,
                      None, :]
        #閻?64 濞戞搩浜濈换浣规償閿旀儳浠柣?3D 闁秆勫姈閻栵綁骞忛崗鐓庡闁挎稑鐭傞埀顒佷亢缁?MLP 缂傚倹鐗滈悥?
        return self.rv_embedding(coords_3d.reshape(*coords_3d.shape[:-2], -1))

    def _bev_query_embed(self, ref_points, img_metas):
        #  Query 闁?BEV 濞达絽绉堕悿鍡欑磽閺嶎偆鍨?
        # 闁?BEV 濞ｅ浂鍨甸～瀣喆閹烘梹鐣?(x, y) 闁秆勫姈閻栵綁寮堕妷銊ｂ偓鍐矆閻戞妲ㄥ☉?Query 闁汇劌瀚埞鏍⒒缂堢姷绉寸紓?
        bev_embeds = self.bev_embedding(pos2embed(ref_points, num_pos_feats=self.hidden_dim))
        return bev_embeds

    def _rv_query_embed(self, ref_points, img_metas):
        # Query 闁?RV 濞达絽绉堕悿鍡欑磽閺嶎偆鍨?
        """
        缂備焦鐟﹂惁鈩冪▔?Query 闁?3D 闁告瑥鍊介埀顒€鍟伴崑锝夋偨閻旂鐏囧☉鎾亾濞戞搩浜炲ù澶愬嫉妤︽娼掗悷娆愬笧濞堟垶鎷呭鍥╂瀭缂傚倹鐗滈悥婊堝Υ閸屾繄绠栫紒瀣儐濡叉悂鏁?D 闁?闁硅埖娲栨總鏍礆閹殿喗绁查柡?闁?婵炲苯鐏濋惃鐘电棯閸ф娅氶柡宥夋敱缁讳焦鎯?闁?闁告瑥绉垫慨鍥亹閸楃偞绀€ 3D 闁?MLP 缂傚倹鐗滈悥?
        濞戞捁妗ㄧ划鍫熺▕閸粎鐟濋柣鈺佺摠鐢挳鎮?3D 闁秆勫姈閻栵綁鏁嶉悢閿嬬濞戞捇缂氱换鏍ㄧ▔椤忓棛妞介柣顔绘祰椤╋箓宕仦鑺ョ闁稿秴绻掓竟鎺戭嚗娴ｉ晲绮?cross-attention闁挎稑鐭傚〒鍓佹啺娴ｅ憡瀚查柛銉﹀劤閸庢岸鎮ч悷鎵獧闁?RV 濞达絽绉堕悿鍡欑磽閺嶎偆鍨抽柨娑樻綀rv_pe闁挎稑顦﹢顏堝触鐏炶偐顏卞☉鎿冧海閵嗗啰绮堥搹鍏夋晞闂傚倻绻濋懙?
        """
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        ### 闁告垵妫楅ˇ顒€顫㈤敐鍛€婚柛婊冪焸閳ь剙妫楅幃婊堝箮閺囩偛顨涢柣顓涙櫊濡偓
        lidars2imgs = np.stack([meta['lidar2img'] for meta in img_metas])
        lidars2imgs = torch.from_numpy(lidars2imgs).float().to(ref_points.device)
        imgs2lidars = np.stack([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(ref_points.device)

        # 閻忓繐妫楀顒勬嚀閸愵亜浠ù鐘查缂嶅﹥绋夐埀顒勫礌閺嵮勭稄闁哄秴娲╃换鏇㈠储閻斿嘲鐓傞柣顏嗗枎閻?3D 闁秆勫姈閻?
        ref_points = ref_points * (ref_points.new_tensor(self.pc_range[3:]) - ref_points.new_tensor(
            self.pc_range[:3])) + ref_points.new_tensor(self.pc_range[:3])
        # 閻?3D 闁告瑥鍊介埀顒€鍟伴崑锝夊箮閺囩偛顨涢柛?6 濞戞搩浜炲ù澶愬嫉閾忚鐣?2D 闁稿秴绻掔粈宀勫锤閹邦厾鍨?
        proj_points = torch.einsum('bnd, bvcd -> bvnc',
                                   torch.cat([ref_points, ref_points.new_ones(*ref_points.shape[:-1], 1)], dim=-1),
                                   lidars2imgs)
        # 闂侇偄绻楅～瀣⒔閵堝棛銆婇柨娑樼墦濞呭孩绂掗妷锔剧畳閹?z闁?
        proj_points_clone = proj_points.clone()
        z_mask = proj_points_clone[..., 2:3].detach() > 0
        proj_points_clone[..., :3] = proj_points[..., :3] / (
                proj_points[..., 2:3].detach() + z_mask * 1e-6 - (~z_mask) * 1e-6)
        # proj_points_clone[..., 2] = proj_points.new_ones(proj_points[..., 2].shape)

        # 婵☆偀鍋撻柡灞诲劚閹姐垺绂嶅☉妯烘闁兼澘鍟伴崑锝夋媰閽樺韬柣鈺佹啞濠р偓閻熸瑥妫濋崳褰掑礃?
        mask = (proj_points_clone[..., 0] < pad_w) & (proj_points_clone[..., 0] >= 0) & (
                proj_points_clone[..., 1] < pad_h) & (proj_points_clone[..., 1] >= 0)
        mask &= z_mask.squeeze(-1)

        # 婵炲苯鐏濋惃鐘电棯閸ф娅氶柡?64 濞戞搩浜濈换浣规償?
        coords_d = 1 + torch.arange(self.depth_num, device=ref_points.device).float() * (
                self.pc_range[3] - 1) / self.depth_num
        # 閻忓繐妫欓惁鈩冪▔?2D 闁稿秴绻掔粈宀勫锤閹邦厾鍨?閼?64 濞戞搩浜濈换浣规償閿旇偐绀夌€电増顨呴崺?64 闁哄鈧尙鐟濋柛姘湰缁讳焦鎯旈敂鐐暠濮掔粯鍔栭濂稿锤閹邦厾鍨?
        proj_points_clone = torch.einsum('bvnc, d -> bvndc', proj_points_clone, coords_d)
        # 闁告瑥绉垫慨鍥亹閸楃偞绀€ LiDAR 3D 闁秆勫姈閻?
        proj_points_clone = torch.cat(
            [proj_points_clone[..., :3], proj_points_clone.new_ones(*proj_points_clone.shape[:-1], 1)], dim=-1)
        projback_points = torch.einsum('bvndo, bvco -> bvndc', proj_points_clone, imgs2lidars)

        # 鐟滅増甯婄粩鎾礌?+ MLP 缂傚倹鐗滈悥?
        projback_points = (projback_points[..., :3] - projback_points.new_tensor(self.pc_range[:3])[None, None, None,
                                                      :]) \
                          / (projback_points.new_tensor(self.pc_range[3:]) - projback_points.new_tensor(
            self.pc_range[:3]))[None, None, None, :]

        rv_embeds = self.rv_embedding(projback_points.reshape(*projback_points.shape[:-2], -1))

        ## 闁告梻濮靛鍫澬ч崒姘闁挎稑鐗嗚ぐ褎绌卞┑鍫熸畬闁革负鍔忛～瀣煂鎼粹€虫暥闁汇劌瀚ù澶愬嫉閻氬绀?
        rv_embeds = (rv_embeds * mask.unsqueeze(-1)).sum(dim=1)
        return rv_embeds

    def query_embed(self, ref_points, img_metas):
        ### # 闁告鍠庨～?ref_points 闁告瑯鍨甸崗姗€寮垫径瀣偓顒傜博椤栨埃鍋撶涵椋庣闂傚牏鍋涢悥鍫曞箳閵夈劎绠?0 闁?1闁?
        # inverse_sigmoid 闁告劕鎳橀崕鎾儍?clamp 濞村吋纰嶉崺鍛村棘椤撶偛鐓?[eps, 1-eps]
        # sigmoid 闁告劕绉靛Σ褏浜搁崟顐ｇ [0, 1]
        # 闁轰礁鐗婇悘? 閻?ref_points 婵炴挴鏅涢幏浼村捶閹殿喖顔婇柡澶屽枎閸?(0, 1) 闁告牗妞藉Λ鍧楁晬瀹€鍕級闁稿繐绉甸悗顒傜博椤栨埃鍋?
        ref_points = inverse_sigmoid(ref_points.clone()).sigmoid()  # inverse_sigmoid 濞村吋鑹鹃顔芥綇閹惧啿寮抽柛?clamp闁挎稑鐗婇崺鍛村棘椤撶偛鐓?[eps, 1-eps]闁挎稑顧€缁辨繈鎮炵捄鐑樺€甸悹渚婄磿閻?log(x / (1-x))
        bev_embeds = self._bev_query_embed(ref_points, img_metas)  ## BEV 閻熸瑦甯掔€规娊鎯冮崟顏嗙Т缂?
        rv_embeds = self._rv_query_embed(ref_points, img_metas)  ## 闁烩晛鎲″┃鈧悷娆愬笒鐎规娊鎯冮崟顏嗙Т缂?
        return bev_embeds, rv_embeds

    def register_diffusion_schedule(self, timesteps):
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).float())
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).float())

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def _unwrap_data(self, data):
        return data._data if hasattr(data, '_data') else data

    def _sanitize_gt_lists(self, targets, labels, device):
        clean_targets, clean_labels = [], []
        for boxes, lbs in zip(targets, labels):
            boxes = boxes.to(device)
            lbs = lbs.long().to(device)
            if boxes.numel() == 0 or lbs.numel() == 0:
                clean_targets.append(boxes.new_zeros((0, boxes.shape[-1])))
                clean_labels.append(lbs.new_zeros((0,), dtype=torch.long))
                continue
            valid = torch.isfinite(boxes).all(dim=-1)
            valid = valid & (lbs >= 0) & (lbs < self.total_num_classes)
            if boxes.shape[-1] >= 6:
                valid = valid & torch.isfinite(boxes[:, 3:6]).all(dim=-1)
                valid = valid & (boxes[:, 3:6] > 0).all(dim=-1)
            clean_targets.append(boxes[valid])
            clean_labels.append(lbs[valid])
        return clean_targets, clean_labels

    def _safe_labels(self, labels, device):
        labels = labels.long().to(device)
        bg_labels = torch.full_like(labels, self.total_num_classes)
        valid = (labels >= 0) & (labels < self.total_num_classes)
        return torch.where(valid, labels, bg_labels)

    def _check_query_labels(self, query_labels, context):
        if query_labels.numel() == 0:
            return
        min_label = int(query_labels.min().item())
        max_label = int(query_labels.max().item())
        if min_label < 0 or max_label > self.total_num_classes:
            raise RuntimeError(
                '{} query_labels out of range: min={}, max={}, total_num_classes={}'.format(
                    context, min_label, max_label, self.total_num_classes))

    def _center_to_ref(self, center):
        ref = center.new_zeros(center.shape)
        ref[..., 0:1] = (center[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        ref[..., 1:2] = (center[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        ref[..., 2:3] = (center[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        return ref.clamp(REF_CLAMP_EPS, 1. - REF_CLAMP_EPS)

    def _world_to_ref(self, center, height):
        ref = center.new_zeros(center.shape[:-1] + (3,))
        ref[..., 0:1] = (center[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        ref[..., 1:2] = (center[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        ref[..., 2:3] = (height[..., 0:1] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        return ref.clamp(REF_CLAMP_EPS, 1. - REF_CLAMP_EPS)

    def apply_box_noise(self, reference_points, t, noise=None):
        x_start = (reference_points.clamp(REF_CLAMP_EPS, 1. - REF_CLAMP_EPS) * 2. - 1.) * self.diffusion_scale
        x_t = self.q_sample(x_start, t, noise=noise)
        noised_reference = ((x_t / self.diffusion_scale) + 1.) * 0.5
        return noised_reference.clamp(REF_CLAMP_EPS, 1. - REF_CLAMP_EPS), noise

    def _normal_ref_from_gt(self, img_metas, base_reference_points, device):
        batch_refs = []
        batched_reference_points = base_reference_points.dim() == 3
        for batch_id, img_meta in enumerate(img_metas):
            fallback_refs = base_reference_points[batch_id] if batched_reference_points else base_reference_points
            gt_bboxes = self._unwrap_data(img_meta.get('gt_bboxes_3d', None))
            if gt_bboxes is not None and hasattr(gt_bboxes, 'gravity_center') and gt_bboxes.gravity_center.numel() > 0:
                gt_centers = gt_bboxes.gravity_center.to(device)
                gt_centers = gt_centers[torch.isfinite(gt_centers).all(dim=-1)]
                if gt_centers.numel() == 0:
                    rand_idx = torch.randint(0, fallback_refs.shape[0], (self.num_query,), device=device)
                    batch_refs.append(fallback_refs[rand_idx])
                    continue
                gt_refs = self._center_to_ref(gt_centers)
                if gt_refs.shape[0] >= self.num_query:
                    perm = torch.randperm(gt_refs.shape[0], device=device)[:self.num_query]
                    refs = gt_refs[perm]
                else:
                    rand_idx = torch.randint(0, fallback_refs.shape[0],
                                             (self.num_query - gt_refs.shape[0],), device=device)
                    refs = torch.cat([gt_refs, fallback_refs[rand_idx]], dim=0)
                    refs = refs[torch.randperm(refs.shape[0], device=device)]
            else:
                rand_idx = torch.randint(0, fallback_refs.shape[0], (self.num_query,), device=device)
                refs = fallback_refs[rand_idx]
            batch_refs.append(refs)
        return torch.stack(batch_refs, dim=0)

    def _normal_queries_from_gt(self, img_metas, base_reference_points, device):
        batch_refs = []
        batch_labels = []
        batched_reference_points = base_reference_points.dim() == 3
        for batch_id, img_meta in enumerate(img_metas):
            fallback_refs = base_reference_points[batch_id] if batched_reference_points else base_reference_points
            gt_bboxes = self._unwrap_data(img_meta.get('gt_bboxes_3d', None))
            gt_labels = self._unwrap_data(img_meta.get('gt_labels_3d', None))
            labels = torch.full((self.num_query,), self.total_num_classes,
                                dtype=torch.long, device=device)

            has_gt = (gt_bboxes is not None and gt_labels is not None and
                      hasattr(gt_bboxes, 'gravity_center') and
                      gt_bboxes.gravity_center.numel() > 0 and gt_labels.numel() > 0)
            if has_gt:
                gt_centers = gt_bboxes.gravity_center.to(device)
                gt_labels = gt_labels.long().to(device)
                valid = torch.isfinite(gt_centers).all(dim=-1)
                valid = valid & (gt_labels >= 0) & (gt_labels < self.total_num_classes)
                gt_centers = gt_centers[valid]
                gt_labels = gt_labels[valid]
                if gt_centers.numel() > 0:
                    gt_refs = self._center_to_ref(gt_centers)
                    if gt_refs.shape[0] >= self.num_query:
                        perm = torch.randperm(gt_refs.shape[0], device=device)[:self.num_query]
                        refs = gt_refs[perm]
                        labels = gt_labels[perm]
                    else:
                        rand_idx = torch.randint(0, fallback_refs.shape[0],
                                                 (self.num_query - gt_refs.shape[0],), device=device)
                        refs = torch.cat([gt_refs, fallback_refs[rand_idx]], dim=0)
                        labels = torch.cat([
                            gt_labels,
                            labels.new_full((self.num_query - gt_labels.shape[0],), self.total_num_classes)
                        ], dim=0)
                        perm = torch.randperm(refs.shape[0], device=device)
                        refs = refs[perm]
                        labels = labels[perm]
                    batch_refs.append(refs)
                    batch_labels.append(labels)
                    continue

            rand_idx = torch.randint(0, fallback_refs.shape[0], (self.num_query,), device=device)
            batch_refs.append(fallback_refs[rand_idx])
            batch_labels.append(labels)
        return torch.stack(batch_refs, dim=0), torch.stack(batch_labels, dim=0)

    def _label_content_from_gt(self, img_metas, device):
        labels = torch.full((len(img_metas), self.num_query), self.total_num_classes,
                            dtype=torch.long, device=device)
        for batch_id, img_meta in enumerate(img_metas):
            gt_labels = self._unwrap_data(img_meta.get('gt_labels_3d', None))
            if gt_labels is None or gt_labels.numel() == 0:
                continue
            gt_labels = self._safe_labels(gt_labels, device)
            n = min(gt_labels.shape[0], self.num_query)
            labels[batch_id, :n] = gt_labels[:n]
        return labels

    def _build_noisy_queries(self, base_reference_points, img_metas):
        device = base_reference_points.device
        batch_size = len(img_metas)
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(batch_size, base_reference_points, img_metas)
        time_steps = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        pad_size = 0 if mask_dict is None else mask_dict.get('pad_size', 0)

        normal_refs, normal_labels = self._normal_queries_from_gt(img_metas, base_reference_points, device)
        normal_refs, _ = self.apply_box_noise(normal_refs, time_steps)
        reference_points[:, pad_size:] = normal_refs

        query_labels = torch.full((batch_size, reference_points.shape[1]), self.total_num_classes,
                                  dtype=torch.long, device=device)
        query_labels[:, pad_size:] = normal_labels
        if mask_dict is not None and pad_size > 0:
            dn_labels = self._safe_labels(mask_dict['known_lbs_bboxes'][0], device)
            dn_batch_idx = mask_dict.get('known_bid', None)
            if dn_batch_idx is None:
                dn_batch_idx = mask_dict['batch_idx'].long()[mask_dict['known_indice'].long()]
            query_labels[(dn_batch_idx.long().to(device),
                          mask_dict['map_known_indice'].long().to(device))] = dn_labels

        self._check_query_labels(query_labels, 'train')
        # Keep query content and noisy 3D reference points separated, matching
        # DiffuDETR decoder semantics while preserving MOAD's zero target +
        # BEV/RV query_pos design. Labels remain in mask_dict for DN losses.
        query_content = None
        return reference_points, attn_mask, mask_dict, time_steps, query_content

    def _test_query_content(self, batch_size, num_queries, device, proposal_query_content=None):
        if proposal_query_content is not None:
            return proposal_query_content
        query_labels = torch.full((batch_size, num_queries), self.total_num_classes,
                                  dtype=torch.long, device=device)
        self._check_query_labels(query_labels, 'test')
        return self.label_enc(query_labels)

    def _forward_single_with_reference(self, x, x_img, img_metas, points,
                                       reference_points, attn_mask=None,
                                       mask_dict=None, time_steps=None,
                                       query_content=None,
                                       proposal_outputs=None):
        ret_dicts = []
        x = self.shared_conv(x)

        rv_pos_embeds = self._rv_pe(x_img, img_metas)
        bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))
        bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)

        modalities = copy.deepcopy(self.modalities["train" if self.training else "test"])
        outs_dec, ca_dict = self.transformer(
            x, x_img, bev_query_embeds, rv_query_embeds, bev_pos_embeds, rv_pos_embeds, img_metas,
            attn_masks=attn_mask, modalities=modalities, ref_points=reference_points, pc_range=self.pc_range,
            reg_branch=self.decoder_ref_branches, time_steps=time_steps, query_content=query_content,
            query_embedding=self.query_embedding)
        num_queries_per_modality = [m.shape[2] for m in outs_dec]
        outs_dec = torch.cat(outs_dec, dim=2)
        outs_dec = torch.nan_to_num(outs_dec)

        reference_layers = []
        inter_references = ca_dict.get('inter_references', [])
        for refs in inter_references:
            if refs is None:
                modality_refs = reference_points[None].repeat(outs_dec.shape[0], 1, 1, 1)
            else:
                modality_refs = torch.cat([reference_points[None], refs[:-1]], dim=0)
            reference_layers.append(modality_refs)
        if len(reference_layers) == 0:
            reference = reference_points[None].repeat(outs_dec.shape[0], 1, len(modalities), 1)
        else:
            reference = torch.cat(reference_layers, dim=2)
        reference = inverse_sigmoid(reference.clamp(REF_CLAMP_EPS, 1. - REF_CLAMP_EPS))

        flag = 0
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)
            center = (outs['center'] + reference[..., :2]).sigmoid()
            height = (outs['height'] + reference[..., 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            outs['modalities'] = modalities
            outs['loss_weight_f'] = None

            if self.training:
                for key in list(outs.keys()):
                    if key in ['r_loss', 'weight_list', 'loss_weight_f', 'modalities']:
                        continue
                    outs[key] = list(outs[key].split(num_queries_per_modality, dim=2))

            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]
                known_lbs_bboxes_label = task_mask_dict['known_lbs_bboxes'][0]
                known_labels_raw = task_mask_dict['known_labels_raw']
                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)
                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [torch.where(known_lbs_bboxes_label == class_name.index(i) + flag) for i in class_name]
                task_masks_raw = [torch.where(known_labels_raw == class_name.index(i) + flag) for i in class_name]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)

                for key in list(outs.keys()):
                    if key not in ['modalities', 'r_loss', 'weight_list', 'loss_weight_f']:
                        outs['dn_' + key] = []
                        pad_s = mask_dict['pad_size']
                        for i in range(len(outs[key])):
                            outs['dn_' + key].append(outs[key][i][:, :, :pad_s, :])
                            outs[key][i] = outs[key][i][:, :, pad_s:, :]
                outs['dn_mask_dict'] = task_mask_dict
            ret_dicts.append(outs)
        if proposal_outputs is not None:
            ret_dicts[-1]['proposal_refs'] = proposal_outputs[0]
            ret_dicts[-1]['proposal_logits'] = proposal_outputs[1]
        if self._is_loss_tensor(ca_dict.get('qmod_sel_loss', None)):
            ret_dicts[-1]['qmod_sel_loss'] = ca_dict['qmod_sel_loss']
        return ret_dicts

    def _ret_dicts_to_ref(self, ret_dicts):
        best_score, best_ref = None, None
        for outs in ret_dicts:
            center = outs['center'][-1]
            height = outs['height'][-1]
            logits = outs['cls_logits'][-1].sigmoid().max(dim=-1).values
            ref = self._world_to_ref(center, height)
            if best_score is None:
                best_score, best_ref = logits, ref
            else:
                update_mask = logits > best_score
                best_score = torch.where(update_mask, logits, best_score)
                best_ref = torch.where(update_mask.unsqueeze(-1), ref, best_ref)
        return best_ref, best_score

    def _feature_proposal_init(self, x, x_img, img_metas,
                               return_logits=False, return_query_content=False):
        batch_size = x.shape[0]
        bev_feat = self.shared_conv(x)
        bev_pos = self.bev_embedding(
            pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))
        bev_tokens = bev_feat.flatten(2).transpose(1, 2)
        bev_tokens = bev_tokens + bev_pos.unsqueeze(0)

        img_channels = x_img.shape[1]
        rv_pos = self._rv_pe(x_img, img_metas)
        rv_tokens = x_img.reshape(batch_size, -1, img_channels, *x_img.shape[-2:])
        rv_tokens = rv_tokens.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, img_channels)
        rv_tokens = rv_tokens + rv_pos.reshape(batch_size, -1, img_channels)

        if self.proposal_init_mode == 'bev':
            proposal_tokens = bev_tokens
        elif self.proposal_init_mode in ['img', 'rv']:
            proposal_tokens = rv_tokens
        elif self.proposal_init_mode == 'fused':
            proposal_tokens = torch.cat([bev_tokens, rv_tokens], dim=1)
        else:
            raise ValueError(f'Unsupported proposal_init_mode: {self.proposal_init_mode}')

        proposal_tokens = self.proposal_output_norm(
            self.proposal_output(proposal_tokens))
        proposal_logits = self.proposal_score(proposal_tokens)
        proposal_scores = proposal_logits.sigmoid().max(dim=-1).values
        topk = min(self.num_query, proposal_scores.shape[1])
        topk_indices = torch.topk(proposal_scores, topk, dim=1).indices
        topk_tokens = torch.gather(
            proposal_tokens, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, proposal_tokens.shape[-1]))
        proposal_refs = self.proposal_ref(topk_tokens).sigmoid()
        topk_logits = torch.gather(
            proposal_logits, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, proposal_logits.shape[-1]))

        if topk < self.num_query:
            pad_refs = self.reference_points.weight.to(x.device).unsqueeze(0).repeat(batch_size, 1, 1)
            pad_logits = proposal_logits.new_zeros(batch_size, self.num_query - topk, self.total_num_classes)
            pad_tokens = proposal_tokens.new_zeros(batch_size, self.num_query - topk, proposal_tokens.shape[-1])
            proposal_refs = torch.cat([proposal_refs, pad_refs[:, topk:]], dim=1)
            topk_logits = torch.cat([topk_logits, pad_logits], dim=1)
            topk_tokens = torch.cat([topk_tokens, pad_tokens], dim=1)
        proposal_refs = proposal_refs.clamp(REF_CLAMP_EPS, 1. - REF_CLAMP_EPS)
        if return_logits and return_query_content:
            return proposal_refs, topk_logits, topk_tokens
        if return_logits:
            return proposal_refs, topk_logits
        if return_query_content:
            return proposal_refs, topk_tokens
        return proposal_refs

    def ddim_sample_single(self, x, x_img, img_metas, points):
        device = x.device
        batch_size = x.shape[0]
        if self.use_feature_proposal_init:
            base_reference_points = self._feature_proposal_init(x, x_img, img_metas)
        else:
            base_reference_points = self.reference_points.weight.to(device).unsqueeze(0).repeat(batch_size, 1, 1)
        init_t = torch.full((batch_size,), self.num_timesteps - 1, device=device, dtype=torch.long)
        generator = None
        init_noise = None
        if self.test_noise_seed is not None and self.test_noise_seed >= 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self.test_noise_seed))
            init_noise = torch.randn(base_reference_points.shape, device=device, generator=generator)
        reference_points, _ = self.apply_box_noise(base_reference_points, init_t, noise=init_noise)
        reference_scaled = (reference_points * 2. - 1.) * self.diffusion_scale
        # Keep MOAD decoder query semantics at test time: proposal FFN only
        # initializes reference points, while query content stays the original
        # zero target driven by MOAD query_pos/key_pos.
        query_content = None

        times = torch.linspace(
            0, self.num_timesteps - 1,
            steps=self.sampling_timesteps + 1,
            device=device).long()
        times = list(reversed(times.tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        ensemble_outputs = []
        final_outputs = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            outputs = self._forward_single_with_reference(
                x, x_img, img_metas, points, reference_points,
                attn_mask=None, mask_dict=None, time_steps=time_cond, query_content=query_content)
            final_outputs = outputs
            if self.use_ensemble:
                ensemble_outputs.append(outputs)

            pred_ref, scores = self._ret_dicts_to_ref(outputs)
            pred_scaled = (pred_ref * 2. - 1.) * self.diffusion_scale
            pred_noise = self.predict_noise_from_start(reference_scaled, time_cond, pred_scaled)
            alpha = extract(self.alphas_cumprod, time_cond, reference_scaled.shape)
            next_time_cond = torch.full((batch_size,), time_next, device=device, dtype=torch.long)
            alpha_next = extract(self.alphas_cumprod, next_time_cond, reference_scaled.shape)
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            step_noise = torch.randn_like(reference_scaled)
            reference_scaled = pred_scaled * alpha_next.sqrt() + c * pred_noise + sigma * step_noise

            if self.box_renewal and time_next > 0:
                threshold = float(time_next) / float(max(self.num_timesteps - 1, 1))
                keep = scores > threshold
                fresh_noise = torch.randn_like(reference_scaled) * self.diffusion_scale
                reference_scaled = torch.where(keep.unsqueeze(-1), reference_scaled, fresh_noise)
            reference_points = ((reference_scaled / self.diffusion_scale) + 1.) * 0.5
            reference_points = reference_points.clamp(REF_CLAMP_EPS, 1. - REF_CLAMP_EPS)

        if self.use_ensemble and len(ensemble_outputs) > 1:
            merged_outputs = []
            for task_id in range(len(final_outputs)):
                merged_task = copy.deepcopy(final_outputs[task_id])
                for key in ['center', 'height', 'dim', 'rot', 'vel', 'cls_logits']:
                    if key in merged_task:
                        merged_task[key] = torch.cat(
                            [step_outputs[task_id][key][-1] for step_outputs in ensemble_outputs], dim=1).unsqueeze(0)
                merged_outputs.append(merged_task)
            return merged_outputs
        return final_outputs

    def forward_single(self, x, x_img, img_metas, points, gt_bboxes_3d=None, gt_labels_3d=None):
        if self.training:
            proposal_outputs = None
            base_reference_points = self.reference_points.weight
            if self.use_feature_proposal_init:
                proposal_outputs = self._feature_proposal_init(
                    x.detach(), x_img.detach(), img_metas, return_logits=True)
                base_reference_points = proposal_outputs[0].detach()
            reference_points, attn_mask, mask_dict, time_steps, query_content = \
                self._build_noisy_queries(base_reference_points, img_metas)
            return self._forward_single_with_reference(
                x, x_img, img_metas, points, reference_points, attn_mask,
                mask_dict, time_steps, query_content, proposal_outputs)
        return self.ddim_sample_single(x, x_img, img_metas, points)

    def forward(self, pts_feats, img_feats=None, img_metas=None, points=[None],
                gt_bboxes_3d=None, gt_labels_3d=None):
        """
            list([bs, c, h, w])
            DiffuMoME follows config modalities: train can use all experts, test can use fused only.
        """
        img_metas = [img_metas for _ in range(len(pts_feats))]
        gt_bboxes_3d = [gt_bboxes_3d for _ in range(len(pts_feats))]
        gt_labels_3d = [gt_labels_3d for _ in range(len(pts_feats))]
        return multi_apply(self.forward_single, pts_feats, img_feats, img_metas, points,
                           gt_bboxes_3d, gt_labels_3d)

    def _get_targets_single(self, gt_bboxes_3d, gt_labels_3d, pred_bboxes, pred_logits):
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).to(device)
        gt_labels_3d = gt_labels_3d.long().to(device)
        if gt_bboxes_3d.numel() > 0:
            valid = torch.isfinite(gt_bboxes_3d).all(dim=-1)
            valid = valid & (gt_labels_3d >= 0) & (gt_labels_3d < self.total_num_classes)
            if gt_bboxes_3d.shape[-1] >= 6:
                valid = valid & (gt_bboxes_3d[:, 3:6] > 0).all(dim=-1)
            gt_bboxes_3d = gt_bboxes_3d[valid]
            gt_labels_3d = gt_labels_3d[valid]

        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(name) + flag)
                for name in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag = 0
        for masks in task_masks:
            task_box = []
            task_class = []
            for mask in masks:
                task_box.append(gt_bboxes_3d[mask])
                task_class.append(gt_labels_3d[mask] - flag)
            task_boxes.append(torch.cat(task_box, dim=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag += len(masks)

        def task_assign(bbox_pred, logits_pred, gt_bboxes, gt_labels, num_classes):
            num_bboxes = bbox_pred.shape[0]
            assign_results = self.assigner.assign(bbox_pred, logits_pred, gt_bboxes, gt_labels)
            sampling_result = self.sampler.sample(assign_results, bbox_pred, gt_bboxes)
            pos_inds, neg_inds = sampling_result.pos_inds, sampling_result.neg_inds
            labels = gt_bboxes.new_full((num_bboxes,), num_classes, dtype=torch.long)
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights = gt_bboxes.new_ones(num_bboxes)
            code_size = gt_bboxes.shape[1]
            bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
            bbox_weights = torch.zeros_like(bbox_pred)
            bbox_weights[pos_inds] = 1.0
            if len(sampling_result.pos_gt_bboxes) > 0:
                bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

        return multi_apply(
            task_assign, pred_bboxes, pred_logits, task_boxes, task_classes, self.num_classes)

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits):
        (labels_list, labels_weight_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_targets_single, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits)

        task_num = len(labels_list[0])
        num_total_pos_tasks, num_total_neg_tasks = [], []
        task_labels_list, task_labels_weight_list = [], []
        task_bbox_targets_list, task_bbox_weights_list = [], []
        for task_id in range(task_num):
            num_total_pos_tasks.append(sum(inds[task_id].numel() for inds in pos_inds_list))
            num_total_neg_tasks.append(sum(inds[task_id].numel() for inds in neg_inds_list))
            task_labels_list.append(
                [labels_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_labels_weight_list.append(
                [labels_weight_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_targets_list.append(
                [bbox_targets_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_weights_list.append(
                [bbox_weights_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])

        return (task_labels_list, task_labels_weight_list, task_bbox_targets_list,
                task_bbox_weights_list, num_total_pos_tasks, num_total_neg_tasks)

    def _loss_single_task(self, pred_bboxes, pred_logits, labels_list,
                          labels_weights_list, bbox_targets_list, bbox_weights_list,
                          num_total_pos, num_total_neg):
        labels = torch.cat(labels_list, dim=0)
        labels_weights = torch.cat(labels_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)

        pred_bboxes_flatten = pred_bboxes.flatten(0, 1)
        pred_logits_flatten = pred_logits.flatten(0, 1)

        cls_avg_factor = max(num_total_pos * 1.0 + num_total_neg * 0.1, 1)
        loss_cls = self._loss_cls_torch(
            pred_logits_flatten, labels, labels_weights, avg_factor=cls_avg_factor)

        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]
        loss_bbox = self.loss_bbox(
            pred_bboxes_flatten[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=max(num_total_pos, 1))

        return torch.nan_to_num(loss_cls), torch.nan_to_num(loss_bbox)

    def loss_single(self, pred_bboxes, pred_logits, gt_bboxes_3d, gt_labels_3d):
        batch_size = pred_bboxes[0].shape[0]
        pred_bboxes_list, pred_logits_list = [], []
        for idx in range(batch_size):
            pred_bboxes_list.append([task_pred_bbox[idx] for task_pred_bbox in pred_bboxes])
            pred_logits_list.append([task_pred_logits[idx] for task_pred_logits in pred_logits])

        cls_reg_targets = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, pred_bboxes_list, pred_logits_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._loss_single_task,
            pred_bboxes,
            pred_logits,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg)

        return sum(loss_cls_tasks), sum(loss_bbox_tasks)

    def _dn_loss_single_task(self, pred_bboxes, pred_logits, mask_dict):
        device = pred_logits.device
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        known_labels = known_labels.to(device)
        known_bboxs = known_bboxs.to(device)
        map_known_indice = mask_dict['map_known_indice'].long().to(device)
        known_indice = mask_dict['known_indice'].long().to(device)
        batch_idx = mask_dict['batch_idx'].long().to(device)
        bid = batch_idx[known_indice]
        known_labels_raw = mask_dict['known_labels_raw'].to(device)

        pred_logits = pred_logits[(bid, map_known_indice)]
        pred_bboxes = pred_bboxes[(bid, map_known_indice)]
        num_tgt = known_indice.numel()

        task_mask = known_labels_raw != pred_logits.shape[-1]
        task_mask_sum = task_mask.sum()
        if task_mask_sum > 0:
            pred_bboxes = pred_bboxes[task_mask]
            known_bboxs = known_bboxs[task_mask]

        cls_avg_factor = max(num_tgt * 3.14159 / 6 * self.split * self.split * self.split, 1)
        label_weights = torch.ones_like(known_labels)
        loss_cls = self._loss_cls_torch(
            pred_logits, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        num_tgt = loss_cls.new_tensor([num_tgt])
        num_tgt = torch.clamp(reduce_mean(num_tgt), min=1).item()
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = torch.ones_like(pred_bboxes)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]
        loss_bbox = self.loss_bbox(
            pred_bboxes[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_tgt)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        if task_mask_sum == 0:
            loss_bbox = loss_bbox * 0.0
        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox

    def dn_loss_single(self, pred_bboxes, pred_logits, dn_mask_dict):
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._dn_loss_single_task, pred_bboxes, pred_logits, dn_mask_dict)
        return sum(loss_cls_tasks), sum(loss_bbox_tasks)

    def _loss_cls_torch(self, pred, target, label_weights=None, avg_factor=1.0):
        if getattr(self.loss_cls, 'use_sigmoid', False):
            num_classes = pred.shape[-1]
            valid_mask = (target >= 0) & (target < num_classes)
            target_onehot = pred.new_zeros(pred.shape)
            if valid_mask.any():
                target_onehot[valid_mask, target[valid_mask].long()] = 1

            pred_sigmoid = pred.sigmoid()
            pt = (1 - pred_sigmoid) * target_onehot + pred_sigmoid * (1 - target_onehot)
            focal_weight = (getattr(self.loss_cls, 'alpha', 0.25) * target_onehot +
                            (1 - getattr(self.loss_cls, 'alpha', 0.25)) * (1 - target_onehot))
            focal_weight = focal_weight * pt.pow(getattr(self.loss_cls, 'gamma', 2.0))
            loss = F.binary_cross_entropy_with_logits(
                pred, target_onehot, reduction='none') * focal_weight
            if label_weights is not None:
                if label_weights.dim() == loss.dim() - 1:
                    label_weights = label_weights.unsqueeze(-1)
                loss = loss * label_weights
            loss = loss.sum() / max(float(avg_factor), 1.0)
            return loss * getattr(self.loss_cls, 'loss_weight', 1.0)

        return self.loss_cls(
            pred, target, label_weights, avg_factor=avg_factor)

    @staticmethod
    def _is_loss_tensor(value):
        if torch.is_tensor(value):
            return True
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return all(torch.is_tensor(item) for item in value)
        return False

    def _add_loss_if_tensor(self, loss_dict, name, value):
        if self._is_loss_tensor(value):
            loss_dict[name] = value

    def _sanitize_loss_dict(self, loss_dict):
        return {
            name: value
            for name, value in loss_dict.items()
            if self._is_loss_tensor(value)
        }

    def proposal_loss(self, proposal_refs, proposal_logits, gt_bboxes_3d, gt_labels_3d):
        batch_size, num_proposals, _ = proposal_refs.shape
        labels = proposal_logits.new_full(
            (batch_size, num_proposals), self.total_num_classes, dtype=torch.long)
        label_weights = proposal_logits.new_ones((batch_size, num_proposals))
        ref_targets = proposal_refs.new_zeros((batch_size, num_proposals, 3))
        ref_weights = proposal_refs.new_zeros((batch_size, num_proposals, 3))
        num_pos = 0

        for batch_id, (gt_bboxes, gt_labels) in enumerate(zip(gt_bboxes_3d, gt_labels_3d)):
            if gt_bboxes is None or gt_labels is None or gt_labels.numel() == 0:
                continue
            gt_boxes = torch.cat(
                (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(proposal_refs.device)
            gt_labels = gt_labels.long().to(proposal_refs.device)
            valid = torch.isfinite(gt_boxes).all(dim=-1)
            valid = valid & (gt_labels >= 0) & (gt_labels < self.total_num_classes)
            if gt_boxes.shape[-1] >= 6:
                valid = valid & (gt_boxes[:, 3:6] > 0).all(dim=-1)
            if valid.sum() == 0:
                continue
            gt_refs = self._center_to_ref(gt_boxes[valid, :3])
            gt_labels = gt_labels[valid]
            cost = torch.cdist(gt_refs, proposal_refs[batch_id], p=1)
            proposal_inds = cost.argmin(dim=1)
            labels[batch_id, proposal_inds] = gt_labels
            ref_targets[batch_id, proposal_inds] = gt_refs
            ref_weights[batch_id, proposal_inds] = 1.0
            num_pos += int(torch.unique(proposal_inds).numel())

        # Proposal init is only an auxiliary warm start for DiffuDETR-style
        # sampling. Normalize over all generated proposals so the many
        # background slots do not dominate the three decoder experts.
        cls_avg_factor = max(batch_size * num_proposals, 1)
        loss_cls = self._loss_cls_torch(
            proposal_logits.reshape(-1, self.total_num_classes),
            labels.reshape(-1),
            label_weights.reshape(-1),
            avg_factor=cls_avg_factor)
        loss_ref = F.l1_loss(
            proposal_refs * ref_weights,
            ref_targets * ref_weights,
            reduction='sum') / max(num_pos, 1)
        return (torch.nan_to_num(loss_cls * self.proposal_loss_weight),
                torch.nan_to_num(loss_ref * self.proposal_loss_weight))

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """"Loss function.
        闁诡剛绮畷顖涘緞鏉堫偒鍚€缂佺姵顨呴崣鍡涘矗?
        婵懓娲﹂埀顒傜帛婢у秹寮垫径瀣嗕線骞€娴ｇ鍋撴担鐟邦暡闁?Decoder 閻忕偛鍊堕埀顑跨劍椤掓粎鏁?query 闁?DN query 闁汇劌瀚畷顖涘緞閹插绀夐柛鏃傚С缁?AQR 閻犱警鍨抽弫閬嶅箲閻旀灚浜肩紒娑橆槼缁剁喖宕濋埡鍌氱柈濠㈡湹绌堕埀?
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            preds_dicts(tuple[list[dict]]): nb_tasks x num_lvl
                center: (num_dec, batch_size, num_query, 2)
                height: (num_dec, batch_size, num_query, 1)
                dim: (num_dec, batch_size, num_query, 3)
                rot: (num_dec, batch_size, num_query, 2)
                vel: (num_dec, batch_size, num_query, 2)
                cls_logits: (num_dec, batch_size, num_query, task_classes)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = dict()
        if 'loss_weight_f' in preds_dicts[0][0] and preds_dicts[0][0]['loss_weight_f'] is None:
            for i, modality in enumerate(preds_dicts[0][0]["modalities"]):   # 閻庝絻顫夐惁鈩冪▔椤忓惀渚€骞€娴ｇ鐎婚柛鎺濆亯椤撳摜绮诲Δ浣哥柈濠?
                ### 婵繐绲介悥?query 闁汇劌瀚畷顖涘緞?
                # 缂備礁瀚ˉ濠冿紣閸曨剛銈寸紓浣规尰閻忓鏁嶅顒傛闁告帒妫欓弳搴ㄦ儍?center/height/dim/rot/vel 闁瑰嘲鍚嬬敮瀛樼▔閸濆嫮鏆氶柡浣割嚟濞?bbox 濡澘瀚粊?
                num_decoder = preds_dicts[0][0]['center'][i].shape[0]
                all_pred_bboxes, all_pred_logits = collections.defaultdict(list), collections.defaultdict(list)

                for task_id, preds_dict in enumerate(preds_dicts, 0):
                    for dec_id in range(num_decoder):
                        pred_bbox = [preds_dict[0]['center'][i][dec_id], preds_dict[0]['height'][i][dec_id],
                                    preds_dict[0]['dim'][i][dec_id], preds_dict[0]['rot'][i][dec_id]]
                        if 'vel' in preds_dict[0]:
                            pred_bbox.append(preds_dict[0]['vel'][i][dec_id])
                        pred_bbox = torch.cat(pred_bbox, dim=-1)
                        all_pred_bboxes[dec_id].append(pred_bbox)
                        all_pred_logits[dec_id].append(preds_dict[0]['cls_logits'][i][dec_id])
                all_pred_bboxes = [all_pred_bboxes[idx] for idx in range(num_decoder)]
                all_pred_logits = [all_pred_logits[idx] for idx in range(num_decoder)]

                # 閻庝絻顫夐惁锛勪沪?Decoder 閻犲鍟伴弫?loss_single
                loss_cls, loss_bbox = multi_apply(
                    self.loss_single, all_pred_bboxes, all_pred_logits,
                    [gt_bboxes_3d for _ in range(num_decoder)],
                    [gt_labels_3d for _ in range(num_decoder)],
                )

                # 濞戞搩鍙冨Λ璺ㄤ沪閸屾粍纾ч柣?(Auxiliary Loss)闁挎稒鐭粭澶嬬閸涱喗浠橀柛姘凹缁斿浠?Decoder 闁哄牆顦板畷顖涘緞閹插绀夐柛鎾崇Ч濞间即鎯冮崟顐ゆ勾濞戞梻鍠愬﹢渚€鏁嶇仦钘夘潱闂侇偆鍠曢鍕磼閸愨晜鏆柡?
                loss_dict[f'loss_cls_{modality}'] = loss_cls[-1]
                loss_dict[f'loss_bbox_{modality}'] = loss_bbox[-1]

                num_dec_layer = 0
                for loss_cls_i, loss_bbox_i in zip(loss_cls[:-1],
                                                loss_bbox[:-1]):
                    loss_dict[f'd{num_dec_layer}.loss_cls_{modality}'] = loss_cls_i
                    loss_dict[f'd{num_dec_layer}.loss_bbox_{modality}'] = loss_bbox_i
                    num_dec_layer += 1

                # DN query 闁汇劌瀚畷顖涘緞?
                if 'dn_center' in preds_dicts[0][0] and 'dn_mask_dict' in preds_dicts[0][0]:
                    dn_pred_bboxes, dn_pred_logits = collections.defaultdict(list), collections.defaultdict(list)
                    dn_mask_dicts = collections.defaultdict(list)
                    for task_id, preds_dict in enumerate(preds_dicts, 0):
                        for dec_id in range(num_decoder):
                            pred_bbox = [preds_dict[0]['dn_center'][i][dec_id],
                                         preds_dict[0]['dn_height'][i][dec_id],
                                         preds_dict[0]['dn_dim'][i][dec_id],
                                         preds_dict[0]['dn_rot'][i][dec_id]]
                            if 'vel' in preds_dict[0]:
                                pred_bbox.append(preds_dict[0]['dn_vel'][i][dec_id])
                            pred_bbox = torch.cat(pred_bbox, dim=-1)
                            dn_pred_bboxes[dec_id].append(pred_bbox)
                            dn_pred_logits[dec_id].append(preds_dict[0]['dn_cls_logits'][i][dec_id])
                            dn_mask_dicts[dec_id].append(preds_dict[0]['dn_mask_dict'])
                    dn_pred_bboxes = [dn_pred_bboxes[idx] for idx in range(num_decoder)]
                    dn_pred_logits = [dn_pred_logits[idx] for idx in range(num_decoder)]
                    dn_mask_dicts = [dn_mask_dicts[idx] for idx in range(num_decoder)]
                # 濞戞挸瀛╅婊呮暜?query 閻庣懓鑻崣蹇曗偓闈涙贡琚ㄩ柨娑樺缁查箖鎮介妸褎鐣遍柡?dn_center/dn_height/... 缂?DN 濡澘瀚粊鎾磹?
                    dn_loss_cls, dn_loss_bbox = multi_apply(
                        self.dn_loss_single, dn_pred_bboxes, dn_pred_logits, dn_mask_dicts
                    )

                    loss_dict[f'dn_loss_cls_{modality}'] = dn_loss_cls[-1]
                    loss_dict[f'dn_loss_bbox_{modality}'] = dn_loss_bbox[-1]
                    num_dec_layer = 0
                    for loss_cls_i, loss_bbox_i in zip(dn_loss_cls[:-1],
                                                       dn_loss_bbox[:-1]):
                        loss_dict[f'd{num_dec_layer}.dn_loss_cls_{modality}'] = loss_cls_i
                        loss_dict[f'd{num_dec_layer}.dn_loss_bbox_{modality}'] = loss_bbox_i
                        num_dec_layer += 1
        # 閺夊牆鎳庢慨顏堝箲閻旀灚浜?
        if 'r_loss' in preds_dicts[0][0]:
            self._add_loss_if_tensor(loss_dict, 'r_loss', preds_dicts[0][0]['r_loss'])

        #  婵☆垪鍓濋埀顑跨窔閳ь剙顦扮€氥劑骞戦悢鏋杭
        if 'qmod_sel_loss' in preds_dicts[0][0]:
            self._add_loss_if_tensor(loss_dict, 'qmod_sel_loss', preds_dicts[0][0]['qmod_sel_loss'])
            
        # for the purpose not to get loss (failure_pred) 婵☆垪鍓濋埀顑跨劍濞煎牓鏌屽鍡楃柈濠? 闂佽棄鐗嗛顕€寮甸埀顒勫触鎼达絾鐣卞Λ鏉垮缁佸绱掗幘瀵镐函
        if 'weight_list' in preds_dicts[0][0] and preds_dicts[0][0]['weight_list'] is not None:
            loss_weight_f_list = []
            for weight_list in preds_dicts[0][0]['weight_list']:
                weight_list = weight_list.squeeze(-1).transpose(0,1)
                # batch_size,_num_queries = weight_list.shape
                batch_size,_num_queries, _ = weight_list.shape
                weight_f_target = weight_list.new_tensor(
                    [i['modalmask'] for i in kwargs['img_metas']])
                weight_f_target_expanded = weight_f_target.unsqueeze(1).repeat(1,_num_queries,1)
                loss_weight_f = self._criterion(weight_list,weight_f_target_expanded.float())/_num_queries
                # weight_f_target_expanded = torch.zeros(batch_size, _num_queries).cuda()
                # weight_f_target_expanded[weight_f_target[:, 0] == 1, :int(_num_queries/3)] = 1
                # weight_f_target_expanded[weight_f_target[:, 1] == 1, int(_num_queries/3):int(2*_num_queries/3)] = 1
                # weight_f_target_expanded[weight_f_target[:, 2] == 1, int(2*_num_queries/3):] = 1
                # loss_weight_f = F.binary_cross_entropy(weight_list, weight_f_target_expanded)
                loss_weight_f_list.append(loss_weight_f)
            if len(loss_weight_f_list) > 0:
                weight_modality = preds_dicts[0][0].get('modalities', ['ensemble'])[0]
                self._add_loss_if_tensor(
                    loss_dict,
                    f'loss_weight_f_{weight_modality}',
                    sum(loss_weight_f_list))
        if 'loss_weight_f' in preds_dicts[0][0] and preds_dicts[0][0]['loss_weight_f'] is not None:
            self._add_loss_if_tensor(
                loss_dict,
                'loss_weight_f_ensemble',
                preds_dicts[0][0]['loss_weight_f'])
        if 'proposal_refs' in preds_dicts[0][0] and 'proposal_logits' in preds_dicts[0][0]:
            loss_proposal_cls, loss_proposal_ref = self.proposal_loss(
                preds_dicts[0][0]['proposal_refs'],
                preds_dicts[0][0]['proposal_logits'],
                gt_bboxes_3d,
                gt_labels_3d)
            loss_dict['loss_proposal_cls'] = loss_proposal_cls
            loss_dict['loss_proposal_ref'] = loss_proposal_ref
        return self._sanitize_loss_dict(loss_dict)

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):
        # 婵炴潙顑堥惁顖炲籍閸撲焦鐣遍悷娆欑悼閻?
        # 閻犲鍟伴弫?TransFusionBBoxCoder.decode() 閻忓繐妫欒啯闁搞劌顑堢欢顓㈠礄妤﹀晝鎺楁儘娴ｇ柉绀嬫俊顐熷亾婵炴潙顑囩划銊╁几?
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        # 閻犲鍟弳?z 闁秆勫姈閻栵綁鏁嶅顐ょ煠闂佹彃绉存慨蹇旂▔椤撶偟濡囬弶鐑嗗厸鐠愮喐鎯旈弴銏犲姤濞戞搩鍘肩缓?
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
