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

def pos2embed(pos, num_pos_feats=128, temperature=10000):  ### 位置编码   正余弦位置编码
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
class MultiExpertDecoding(BaseModule):   ### 定义检测头
    def __init__(self,
                 in_channels,
                 num_query=900,   ### 候选目标槽位
                 hidden_dim=128,
                 depth_num=64,
                 norm_bbox=True,
                 downsample_scale=8,
                 pc_range=[],   ### 点云空间范围
                 modalities=dict(),
                 scalar=10,   ### 表示去噪 query 的扩增倍率，或者复制 GT 生成 DN queries 的组数控制参数
                 noise_scale=1.0,
                 noise_trans=0.0,   ### 噪声平移
                 dn_weight=1.0,   ## 去噪损失权重
                 split=0.75,
                 train_cfg=None,
                 test_cfg=None,
                 common_heads=dict(
                     center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
                 ),   ### head_name = (输出维度, head层数)
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
                 loss_cls=dict(   ## 分类损失
                     type="FocalLoss",
                     use_sigmoid=True,
                     reduction="mean",
                     gamma=2, alpha=0.25, loss_weight=1.0
                 ),
                 loss_bbox=dict(  # 框回归损失
                     type="L1Loss",
                     reduction="mean",
                     loss_weight=0.25,
                 ),
                 separate_head=dict(
                     type='SeparateMlpHead', init_bias=-2.19, final_kernel=3),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None
        super(MultiExpertDecoding, self).__init__(init_cfg=init_cfg)
        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.hidden_dim = hidden_dim
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_query = num_query
        self.in_channels = in_channels
        self.depth_num = depth_num  ### 深度离散化的 bin 数   常用在 2D --> 3D、BEV的深度建模  ， 深度方向的离散分辨率
        self.norm_bbox = norm_bbox
        self.downsample_scale = downsample_scale
        ### 去噪训练参数
        self.pc_range = pc_range  # 点云感知范围
        self.modalities = modalities
        self.scalar = scalar   # DN query 的重复组数上限
        self.bbox_noise_scale = noise_scale  # 对真值框中心加噪声的缩放系数
        self.bbox_noise_trans = noise_trans  # 噪声偏移量
        self.dn_weight = dn_weight   # DN 损失的权重
        self.split = split    # 噪声幅度阈值，超过则标记为负样本 ？？？？？

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)  ### 将
        self.fp16_enabled = False

        self.shared_conv = ConvModule(
            in_channels,
            hidden_dim,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=dict(type="BN2d")
        )  # 把通道从 in_channels 映射到 hidden_dim

        # transformer
        self.transformer = build_transformer(transformer)
        # 参考点
        self.reference_points = nn.Embedding(num_query, 3)
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

    def init_weights(self):
        super(MultiExpertDecoding, self).init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

    @property
    def coords_bev(self):
        #### 生成 BEV 特征图上每个网格中心点的归一化 2D 坐标
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        ### 计算 BEV 特征图的实际尺寸
        x_size, y_size = (
            cfg['grid_size'][1] // self.downsample_scale,    ## grid[1]是x
            cfg['grid_size'][0] // self.downsample_scale     ## grid[0]是y
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]   ### 定义两个轴的参数：起点、终点、采样点数
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        ### 归一化到 [0, 1]，并偏移到每个网格的中心
        # 为什么 +0.5？因为坐标应该指向网格的中心而不是左上角
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        ## 最终输出归一化的 2D 坐标，每个对应 BEV 特征图的一个像素位置
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0)  # (H*W, 2)
        return coord_base

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        ### 为去噪训练 (Denoising Training) 准备所有数据：DN queries、attention mask、以及用于计算损失的 mask_dict
        if self.training:
            targets = [
                torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),
                          dim=1) for img_meta in img_metas]       ### 从每个样本的 img_metas 中提取真值框。gravity_center 是重力中心 (x,y,z)，tensor[:, 3:] 是 (w,l,h,yaw,vx,vy)
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas]  ## 提取标签
            ### 生成全 1 的掩码，表示"所有真值框都参与 DN"（没有遮蔽任何框）
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]  # 每批次样本有多少数据
            labels = torch.cat([t for t in labels])   #  所有标签展平
            boxes = torch.cat([t for t in targets])  # 所有框展平
            batch_idx = torch.cat([torch.full((t.size(0),), i) for i, t in enumerate(targets)])  # 标记每个框属于哪个样本

            known_indice = torch.nonzero(unmask_label + unmask_bbox)   # 将未屏蔽位置的索引取出
            known_indice = known_indice.view(-1)
            # add noise  重复 groups 次 ， known_labels_raw 是不加修改的原始标签副本，后面计算 DN 损失时需要用
            # 采样越多，覆盖越全面，模型的去噪能力越强
            groups = min(self.scalar, self.num_query // max(known_num))
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_labels_raw = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)  ## 实际属于哪个batch
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            # 加噪声  针对所有的检测框都进行平移
            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans  # 计算噪声的最大偏移范围 = 框尺寸的一半 + 偏移量
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0  #生成 [-1, 1] 之间的均匀随机数，50 个 DN query 各不相同
                known_bbox_center += torch.mul(rand_prob,
                                               diff) * self.bbox_noise_scale    ## 添加噪声
                ### 归一化到 [0, 1]，与 reference_points 的范围一致
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (
                        self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (
                        self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (
                        self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                ### 噪声太大的标记为负样本
                # 噪声加得太大，偏离真值太远，这种 query 应该被当作"这里什么都没有"来训练
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = sum(self.num_classes)

            # Padding 和参考点拼接
            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            # 将 30 个 DN 位置（初始为零）拼接在 900 个正常参考点前面
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size,
                                                                                                               1, 1)
            #填充 DN 参考点坐标
            # 构建映射索引——告诉每个 DN query 应该放在 padded_reference_points 的哪个位置
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]   # 每个样本内的局部索引
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(groups)]).long()  # 在全局的位置DN
            # 用 (batch_idx, position_idx) 索引，把加了噪声的真值中心坐标填入对应位置
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(
                    reference_points.device)
                # padding 位置放的是加噪声的坐标框

            # 构造 Attention Mask
            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            # # 不同 DN 组之间互不可见
            for i in range(groups):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == groups - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'known_labels_raw': known_labels_raw,
                'know_idx': know_idx,
                'pad_size': pad_size
            }

        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _rv_pe(self, img_feats, img_metas):
        # 图像特征的 RV 位置编码,给图像特征图的每个像素生成一个 3D 空间位置编码
        BN, C, H, W = img_feats.shape
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]   # 获取 padding 后的原始图像尺寸
        ### 将特征图坐标映射回原始图像像素坐标
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W
        ### 生成 64 个深度采样值
        coords_d = 1 + torch.arange(self.depth_num, device=img_feats[0].device).float() * (
                self.pc_range[3] - 1) / self.depth_num
        coords_h, coords_w, coords_d = torch.meshgrid([coords_h, coords_w, coords_d])  # 生成 3D 网格：每个像素 × 每个深度 = 一个采样点

        coords = torch.stack([coords_w, coords_h, coords_d, coords_h.new_ones(coords_h.shape)], dim=-1)  # 每个点: (u, v, d, 1) 的齐次坐标
        coords[..., :2] = coords[..., :2] * coords[..., 2:3]   # 要先把 (u, v) 乘以深度 d，才能用逆投影矩阵反投影到 3D 空间

        ### 计算相机→LiDAR 的逆投影矩阵
        imgs2lidars = np.concatenate([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(coords.device)
        # 核心操作：将像素坐标反投影到 LiDAR 3D 坐标系
        coords_3d = torch.einsum('hwdo, bco -> bhwdc', coords, imgs2lidars)
        # 归一化到 [0, 1]
        coords_3d = (coords_3d[..., :3] - coords_3d.new_tensor(self.pc_range[:3])[None, None, None, :]) \
                    / (coords_3d.new_tensor(self.pc_range[3:]) - coords_3d.new_tensor(self.pc_range[:3]))[None, None,
                      None, :]
        #将 64 个深度点的 3D 坐标拼接，通过 MLP 编码
        return self.rv_embedding(coords_3d.reshape(*coords_3d.shape[:-2], -1))

    def _bev_query_embed(self, ref_points, img_metas):
        #  Query 的 BEV 位置编码
        # 用 BEV 俯视角的 (x, y) 坐标来表示每个 Query 的空间位置
        bev_embeds = self.bev_embedding(pos2embed(ref_points, num_pos_feats=self.hidden_dim))
        return bev_embeds

    def _rv_query_embed(self, ref_points, img_metas):
        # Query 的 RV 位置编码
        """
        给每个 Query 的 3D 参考点生成一个相机视角的位置编码。过程是：3D → 投影到相机 → 沿射线采样深度 → 反投影回 3D → MLP 编码
        为什么不直接用 3D 坐标？因为这个编码要和图像特征做 cross-attention，需要和图像特征的 RV 位置编码（_rv_pe）在同一个表示空间中
        """
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        ### 准备正向和逆向投影矩阵
        lidars2imgs = np.stack([meta['lidar2img'] for meta in img_metas])
        lidars2imgs = torch.from_numpy(lidars2imgs).float().to(ref_points.device)
        imgs2lidars = np.stack([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(ref_points.device)

        # 将参考点从归一化坐标还原到真实 3D 坐标
        ref_points = ref_points * (ref_points.new_tensor(self.pc_range[3:]) - ref_points.new_tensor(
            self.pc_range[:3])) + ref_points.new_tensor(self.pc_range[:3])
        # 将 3D 参考点投影到 6 个相机的 2D 像素坐标
        proj_points = torch.einsum('bnd, bvcd -> bvnc',
                                   torch.cat([ref_points, ref_points.new_ones(*ref_points.shape[:-1], 1)], dim=-1),
                                   lidars2imgs)
        # 透视除法（除以深度 z）
        proj_points_clone = proj_points.clone()
        z_mask = proj_points_clone[..., 2:3].detach() > 0
        proj_points_clone[..., :3] = proj_points[..., :3] / (
                proj_points[..., 2:3].detach() + z_mask * 1e-6 - (~z_mask) * 1e-6)
        # proj_points_clone[..., 2] = proj_points.new_ones(proj_points[..., 2].shape)

        # 检查哪些参考点落在相机视野内
        mask = (proj_points_clone[..., 0] < pad_w) & (proj_points_clone[..., 0] >= 0) & (
                proj_points_clone[..., 1] < pad_h) & (proj_points_clone[..., 1] >= 0)
        mask &= z_mask.squeeze(-1)

        # 沿射线采样 64 个深度
        coords_d = 1 + torch.arange(self.depth_num, device=ref_points.device).float() * (
                self.pc_range[3] - 1) / self.depth_num
        # 将每个 2D 像素坐标 × 64 个深度，得到 64 条不同深度的齐次坐标
        proj_points_clone = torch.einsum('bvnc, d -> bvndc', proj_points_clone, coords_d)
        # 反投影回 LiDAR 3D 坐标
        proj_points_clone = torch.cat(
            [proj_points_clone[..., :3], proj_points_clone.new_ones(*proj_points_clone.shape[:-1], 1)], dim=-1)
        projback_points = torch.einsum('bvndo, bvco -> bvndc', proj_points_clone, imgs2lidars)

        # 归一化 + MLP 编码
        projback_points = (projback_points[..., :3] - projback_points.new_tensor(self.pc_range[:3])[None, None, None,
                                                      :]) \
                          / (projback_points.new_tensor(self.pc_range[3:]) - projback_points.new_tensor(
            self.pc_range[:3]))[None, None, None, :]

        rv_embeds = self.rv_embedding(projback_points.reshape(*projback_points.shape[:-2], -1))

        ## 加权求和（只保留在视野内的相机）
        rv_embeds = (rv_embeds * mask.unsqueeze(-1)).sum(dim=1)
        return rv_embeds

    def query_embed(self, ref_points, img_metas):
        ### # 原始 ref_points 可能有极端值（非常接近 0 或 1）
        # inverse_sigmoid 内部的 clamp 会截断到 [eps, 1-eps]
        # sigmoid 再映射回 [0, 1]
        # 效果: 将 ref_points 温和地约束到 (0, 1) 区间，避免极端值
        ref_points = inverse_sigmoid(ref_points.clone()).sigmoid()  # inverse_sigmoid 会对输入做 clamp（截断到 [eps, 1-eps]），然后计算 log(x / (1-x))
        bev_embeds = self._bev_query_embed(ref_points, img_metas)  ## BEV 角度的位置
        rv_embeds = self._rv_query_embed(ref_points, img_metas)  ## 相机角度的位置
        return bev_embeds, rv_embeds

    def forward_single(self, x, x_img, img_metas, points):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
            单个前向传播，包括以下重要步骤：
                特征准备（BEV特征图 + 图像特征图）：

                将 LiDAR BEV 特征图（x）降维。
                构造 QUERY 系列（900个查询点 + 去噪训练 DN queries）。
                给 BEV 和图像 Key 添加位置编码。
                Transformer 解码器（MultiExpert）：

                结合各种模态的特征，完成 cross-attention。
                预测任务：

                每个任务头（Task Head）利用 Transformer 的输出预测物体属性（如位置中心、高度、尺寸、旋转等）。
                解析/后处理：

                在训练时，对预测值进行拆分，适配多个模态和去噪训练（DN）。
                调用 visualize_queries（可选）可用于调试 Query 查询点。
                x 是 Lidar 点云生成的 BEV 特征图
        """
        ret_dicts = []
        # BEV 特征降维
        x = self.shared_conv(x)

        # QUERY 初始化
        reference_points = self.reference_points.weight

        # 去噪训练 DN queries
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(x.shape[0], reference_points, img_metas)

        rv_pos_embeds = self._rv_pe(x_img, img_metas)
        bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))

        bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)

        modalities = copy.deepcopy(self.modalities["train" if self.training else "test"])

        # 调用 MultiExpert
        ###outs_dec: Transformer 的 Decoder 最终结果
        # ca_dict: cross-attention 的中间信息（如 QUERY 路由选择损失 qmod_sel_loss）
        outs_dec, ca_dict = self.transformer(
            x, x_img, bev_query_embeds, rv_query_embeds, bev_pos_embeds, rv_pos_embeds, img_metas,
            attn_masks=attn_mask, modalities=modalities, ref_points=reference_points, pc_range=self.pc_range,
        )
        num_queries_per_modality = [m.shape[2] for m in outs_dec]
        outs_dec = torch.cat(outs_dec, dim=2)   #### 每一层的decoder的输出
        if 'zero_idx' in ca_dict and not self.training:  ## 测试
            _num_decoder, _, _, _dim = outs_dec.shape
            zero_idx = torch.cat(ca_dict['zero_idx'], dim=2)
            _zero_idx = zero_idx[-1][-1]  # 取最后一层
            outs_dec = outs_dec[-1][-1][_zero_idx]
            outs_dec = outs_dec.unsqueeze(0).repeat(_num_decoder,1,1).unsqueeze(1)
        outs_dec = torch.nan_to_num(outs_dec)
        reference = inverse_sigmoid(reference_points.clone())
        reference = reference.repeat(1, len(modalities), 1)
        if 'zero_idx' in ca_dict and not self.training:
            reference = reference[zero_idx[-1]][None,...]
        flag = 0
        def visualize_queries(points, centers, modal_idx, save_path='query_visualization.png', save_dpi=300):
            """
            Visualize point cloud and query positions on BEV (Bird's Eye View) (-60m ~ 60m range)
            提供工具函数 visualize_queries 用于可视化 QUERY 的分布
            
            Args:
                points: shape [N, 3], x,y,z coordinates of point cloud (torch.Tensor)
                centers: shape [M, 2], x,y coordinates of each query (torch.Tensor)
                modal_idx: shape [M], modality index for each query (torch.Tensor)
                save_path: str, path to save the file
                save_dpi: int, DPI (resolution) of the saved image
            """
            
            # Define colors and labels for each modality
            colors = ['red', 'green', 'blue']
            labels = ['Fusion', 'LiDAR', 'Camera']
            
            # Set graph size
            plt.figure(figsize=(12, 10))
            
            # Filter point cloud (-60m ~ 60m)
            mask_points = (points[:, 0] >= -60) & (points[:, 0] <= 60) & \
                        (points[:, 1] >= -60) & (points[:, 1] <= 60)
            filtered_points = points[mask_points]
            
            # Visualize point cloud (in gray)
            plt.scatter(filtered_points[:, 0].cpu(), filtered_points[:, 1].cpu(), 
                    c='gray', alpha=0.1, s=1)
            
            # Calculate number of queries for each modality
            modal_counts = []
            
            # Filter query positions (-60m ~ 60m)
            mask_centers = (centers[:, 0] >= -60) & (centers[:, 0] <= 60) & \
                        (centers[:, 1] >= -60) & (centers[:, 1] <= 60)
            
            # Draw scatter plot for each modality
            for idx in range(3):
                mask = (modal_idx == idx) & mask_centers
                count = torch.sum(mask).item()
                modal_counts.append(count)
                
                if torch.any(mask):
                    plt.scatter(
                        centers[mask, 0].cpu(),
                        centers[mask, 1].cpu(),
                        c=colors[idx],
                        label=f'{labels[idx]} ({count} queries)',
                        alpha=0.6,
                        s=30
                    )
            
            # Total number of queries (filtered)
            total_queries = sum(modal_counts)
            
            # Add query count information to the title
            plt.title(f'Query Positions on BEV View')
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set x, y axis range
            plt.xlim(-60, 60)
            plt.ylim(-60, 60)
            
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save the graph
            plt.tight_layout()
            plt.savefig(save_path, dpi=save_dpi, bbox_inches='tight')
            
            print(f"Graph saved to '{save_path}'.")
            print(f"\nQuery statistics (-60m ~ 60m range):")
            print(f"Total queries: {total_queries}")
            print(f"LiDAR queries: {modal_counts[0]}")
            print(f"Camera queries: {modal_counts[1]}")
            print(f"Fusion queries: {modal_counts[2]}")
            
            plt.show()
            plt.close()
        ## 任务解码
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)
            # 解码预测值: Center, Height 还原到实际物理坐标
            center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
            height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
            # 中心: 归一化 → 世界坐标
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            # 高度: 归一化 → 世界坐标
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            
            if self.training:  # 处理 DN queries 的预测结果
                ## 将 DN queries 的预测值拆分出来（dn_center, dn_height）。
                # 根据 mask_dict 设置目标
                for key in outs.keys():
                    if key in ['r_loss', 'weight_list', 'loss_weight_f']:
                        continue
                    outs[key] = list(outs[key].split(num_queries_per_modality, dim=2))

            outs["modalities"] = modalities

            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

                known_lbs_bboxes_label = task_mask_dict['known_lbs_bboxes'][0]
                known_labels_raw = task_mask_dict['known_labels_raw']

                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)

                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [
                    torch.where(known_lbs_bboxes_label == class_name.index(i) + flag)
                    for i in class_name
                ]
                task_masks_raw = [
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)

                for key in list(outs.keys()):
                    if key not in ["modalities", "r_loss",'weight_list','loss_weight_f']:
                        outs["dn_" + key] = []
                        pad_s = mask_dict['pad_size']
                        for i in range(len(outs[key])):
                            outs['dn_' + key].append(outs[key][i][:, :, :pad_s, :])
                            outs[key][i] = outs[key][i][:, :, pad_s:, :]

                outs['dn_mask_dict'] = task_mask_dict
            ret_dicts.append(outs)
        if 'qmod_sel_loss' in ca_dict:
            outs['qmod_sel_loss'] = ca_dict['qmod_sel_loss']
        return ret_dicts

    def forward(self, pts_feats, img_feats=None, img_metas=None, points=[None]):
        """
            list([bs, c, h, w])
            批量调用 forward_single 处理所有检测任务
        """
        img_metas = [img_metas for _ in range(len(pts_feats))]
        return multi_apply(self.forward_single, pts_feats, img_feats, img_metas, points)

    def _get_targets_single(self, gt_bboxes_3d, gt_labels_3d, pred_bboxes, pred_logits):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        单个样本的目标生成
        对一个样本，将真值框按 task 分组，然后用匈牙利匹配将 query 和真值框一一配对
        Args:

            gt_bboxes_3d (Tensor):  LiDARInstance3DBoxes(num_gts, 9)
            gt_labels_3d (Tensor): Ground truth class indices (num_gts, )
            pred_bboxes (list[Tensor]): num_tasks x (num_query, 10)
            pred_logits (list[Tensor]): num_tasks x (num_query, task_classes)
        Returns:
            tuple[Tensor]: a tuple containing the following.
                - labels_tasks (list[Tensor]): num_tasks x (num_query, ).
                - label_weights_tasks (list[Tensor]): num_tasks x (num_query, ).
                - bbox_targets_tasks (list[Tensor]): num_tasks x (num_query, 9).
                - bbox_weights_tasks (list[Tensor]): num_tasks x (num_query, 10).
                - pos_inds (list[Tensor]): num_tasks x Sampled positive indices.
                - neg_inds (Tensor): num_tasks x Sampled negative indices.
        """
        # 提取真值框数据
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).to(device)

        # 按 task 分组真值框
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        # 提取每个 task 的真值框，并将全局标签转为 task 内部的局部标签
        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, dim=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)

        def task_assign(bbox_pred, logits_pred, gt_bboxes, gt_labels, num_classes):
            """
            匈牙利匹配
            """
            num_bboxes = bbox_pred.shape[0]
            assign_results = self.assigner.assign(bbox_pred, logits_pred, gt_bboxes, gt_labels)
            sampling_result = self.sampler.sample(assign_results, bbox_pred, gt_bboxes)
            pos_inds, neg_inds = sampling_result.pos_inds, sampling_result.neg_inds
            # label targets
            labels = gt_bboxes.new_full((num_bboxes,),
                                        num_classes,
                                        dtype=torch.long)
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights = gt_bboxes.new_ones(num_bboxes)
            # bbox_targets
            code_size = gt_bboxes.shape[1]
            bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
            bbox_weights = torch.zeros_like(bbox_pred)
            bbox_weights[pos_inds] = 1.0

            if len(sampling_result.pos_gt_bboxes) > 0:
                bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

        labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks \
            = multi_apply(task_assign, pred_bboxes, pred_logits, task_boxes, task_classes, self.num_classes)

        return labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        批量计算匹配目标
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            pred_bboxes (list[list[Tensor]]): batch_size x num_task x [num_query, 10].
            pred_logits (list[list[Tensor]]): batch_size x num_task x [num_query, task_classes]
        Returns:
            tuple: a tuple containing the following targets.
                - task_labels_list (list(list[Tensor])): num_tasks x batch_size x (num_query, ).
                - task_labels_weight_list (list[Tensor]): num_tasks x batch_size x (num_query, )
                - task_bbox_targets_list (list[Tensor]): num_tasks x batch_size x (num_query, 9)
                - task_bbox_weights_list (list[Tensor]): num_tasks x batch_size x (num_query, 10)
                - num_total_pos_tasks (list[int]): num_tasks x Number of positive samples
                - num_total_neg_tasks (list[int]): num_tasks x Number of negative samples.
        """
        # multi_apply 对 batch 中每个样本调用 _get_targets_single
        (labels_list, labels_weight_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_targets_single, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits
        )
        task_num = len(labels_list[0])
        num_total_pos_tasks, num_total_neg_tasks = [], []
        task_labels_list, task_labels_weight_list, task_bbox_targets_list, \
            task_bbox_weights_list = [], [], [], []

        # 统计每个 task 在整个 batch 中的正负样本总数
        for task_id in range(task_num):
            num_total_pos_task = sum((inds[task_id].numel() for inds in pos_inds_list))
            num_total_neg_task = sum((inds[task_id].numel() for inds in neg_inds_list))
            num_total_pos_tasks.append(num_total_pos_task)
            num_total_neg_tasks.append(num_total_neg_task)
            task_labels_list.append([labels_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_labels_weight_list.append(
                [labels_weight_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_targets_list.append(
                [bbox_targets_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_weights_list.append(
                [bbox_weights_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])

        return (task_labels_list, task_labels_weight_list, task_bbox_targets_list,
                task_bbox_weights_list, num_total_pos_tasks, num_total_neg_tasks)

    def _loss_single_task(self,
                          pred_bboxes,
                          pred_logits,
                          labels_list,
                          labels_weights_list,
                          bbox_targets_list,
                          bbox_weights_list,
                          num_total_pos,
                          num_total_neg):
        """"Compute loss for single task.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            pred_bboxes (Tensor): (batch_size, num_query, 10)
            pred_logits (Tensor): (batch_size, num_query, task_classes)
            labels_list (list[Tensor]): batch_size x (num_query, )
            labels_weights_list (list[Tensor]): batch_size x (num_query, )
            bbox_targets_list(list[Tensor]): batch_size x (num_query, 9)
            bbox_weights_list(list[Tensor]): batch_size x (num_query, 10)
            num_total_pos: int
            num_total_neg: int
        Returns:
            loss_cls
            loss_bbox
        """
        labels = torch.cat(labels_list, dim=0)
        labels_weights = torch.cat(labels_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)

        pred_bboxes_flatten = pred_bboxes.flatten(0, 1)
        pred_logits_flatten = pred_logits.flatten(0, 1)

        # 分类损失
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * 0.1
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits_flatten, labels, labels_weights, avg_factor=cls_avg_factor
        )

        # 回归损失
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]

        loss_bbox = self.loss_bbox(
            pred_bboxes_flatten[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def loss_single(self,
                    pred_bboxes,
                    pred_logits,
                    gt_bboxes_3d,
                    gt_labels_3d):
        """"Loss function for outputs from a single decoder layer of a single
        单层 Decoder 的损失入口
        协调调用 get_targets 和 _loss_single_task，计算一层 Decoder 输出的总损失
        feature level.
        Args:
            pred_bboxes (list[Tensor]): num_tasks x [bs, num_query, 10].
            pred_logits (list(Tensor]): num_tasks x [bs, num_query, task_classes]
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_list (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        batch_size = pred_bboxes[0].shape[0]
        pred_bboxes_list, pred_logits_list = [], []
        for idx in range(batch_size):
            pred_bboxes_list.append([task_pred_bbox[idx] for task_pred_bbox in pred_bboxes])
            pred_logits_list.append([task_pred_logits[idx] for task_pred_logits in pred_logits])
        cls_reg_targets = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, pred_bboxes_list, pred_logits_list
        )
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
            num_total_neg
        )

        return sum(loss_cls_tasks), sum(loss_bbox_tasks)

    def _dn_loss_single_task(self,
                             pred_bboxes,
                             pred_logits,
                             mask_dict):
        ## 对一个 task，计算 DN queries 的分类损失和回归损失。与正常 query 的损失不同，DN queries 不需要匈牙利匹配——因为它们的目标已经在 prepare_for_dn 中确定了

        # 提取 DN 信息 ， 从 mask_dict 中取出所有 DN 相关信息
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        known_labels_raw = mask_dict['known_labels_raw']

        # for cfa transformer CFA Transformer 处理
        if 'cfa_mq_idx' in mask_dict:
            # masking whether indice in topk idx
            pad_s = self.ensemble.numq_per_modal_dn*3
            selected_rows = mask_dict['cfa_mq_idx'][bid, :pad_s]
            expanded_map_known_indice = map_known_indice.unsqueeze(1).to(pred_bboxes.device)
            mask = (selected_rows == expanded_map_known_indice).any(dim=1)

            known_labels, known_bboxs = known_labels[mask], known_bboxs[mask]
            map_known_indice, known_indice = map_known_indice[mask], known_indice[mask]
            bid, known_labels_raw = bid[mask], known_labels_raw[mask]

            selected_rows = mask_dict['cfa_mq_idx'][bid, :pad_s]
            match_indices = (selected_rows == map_known_indice.unsqueeze(1).to(selected_rows.device)).nonzero(as_tuple=False)
            # for unique_idx
            unique_values, inverse_indices = torch.unique(match_indices[:, 0], return_inverse=True)
            first_occurrences = torch.full((unique_values.size(0),), match_indices.size(0), dtype=torch.long).to(pred_bboxes.device)
            first_occurrences.scatter_(0, inverse_indices, torch.arange(match_indices.size(0), dtype=torch.long).to(pred_bboxes.device))
            first_occurrences = torch.min(first_occurrences, torch.tensor(match_indices.size(0), dtype=torch.long))
            match_indices = match_indices[first_occurrences]
            map_known_indice = match_indices[:, 1]

            assert mask.sum() == first_occurrences.shape[0], 'map_known_indice shape unmatched'

        # 提取 DN queries 的预测值 , 需要跳过padding 位置的框，因为padding位置毫无意义
        pred_logits = pred_logits[(bid, map_known_indice)]
        pred_bboxes = pred_bboxes[(bid, map_known_indice)]
        num_tgt = known_indice.numel()

        # filter task bbox 过滤当前 task 的框 , 只对属于当前 task 的 DN queries 计算回归损失 , 分类损失仍然用所有 50 个 DN queries（包括背景标签的），但回归损失只用属于当前 task 的
        task_mask = known_labels_raw != pred_logits.shape[-1]
        task_mask_sum = task_mask.sum()

        if task_mask_sum > 0:
            pred_bboxes = pred_bboxes[task_mask]
            known_bboxs = known_bboxs[task_mask]

        # classification loss  分类损失
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_tgt * 3.14159 / 6 * self.split * self.split * self.split

        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes  回归损失
        num_tgt = loss_cls.new_tensor([num_tgt])
        num_tgt = torch.clamp(reduce_mean(num_tgt), min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = torch.ones_like(pred_bboxes)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]
        loss_bbox = self.loss_bbox(
            pred_bboxes[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10],
            avg_factor=num_tgt)

        # 处理边界情况 + 返回
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        if task_mask_sum == 0:
            loss_bbox = loss_bbox * 0.0

        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox

    def dn_loss_single(self,
                       pred_bboxes,
                       pred_logits,
                       dn_mask_dict):
        # 单层 Decoder 的 DN 损失
        # 对所有 task 调用 _dn_loss_single_task，汇总损失
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._dn_loss_single_task, pred_bboxes, pred_logits, dn_mask_dict
        )
        return sum(loss_cls_tasks), sum(loss_bbox_tasks)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """"Loss function.
        总损失计算入口
        汇总所有模态、所有 Decoder 层、正常 query 和 DN query 的损失，加上 AQR 路由损失等辅助损失。
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
            for i, modality in enumerate(preds_dicts[0][0]["modalities"]):   # 对每个模态分别计算损失
                ### 正常 query 的损失
                # 组装预测结果：将分散的 center/height/dim/rot/vel 拼接为完整的 bbox 预测
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

                # 对每层 Decoder 调用 loss_single
                loss_cls, loss_bbox = multi_apply(
                    self.loss_single, all_pred_bboxes, all_pred_logits,
                    [gt_bboxes_3d for _ in range(num_decoder)],
                    [gt_labels_3d for _ in range(num_decoder)],
                )

                # 中间层监督 (Auxiliary Loss)：不仅最后一层 Decoder 有损失，前面的层也有，加速训练收敛
                loss_dict[f'loss_cls_{modality}'] = loss_cls[-1]
                loss_dict[f'loss_bbox_{modality}'] = loss_bbox[-1]

                num_dec_layer = 0
                for loss_cls_i, loss_bbox_i in zip(loss_cls[:-1],
                                                loss_bbox[:-1]):
                    loss_dict[f'd{num_dec_layer}.loss_cls_{modality}'] = loss_cls_i
                    loss_dict[f'd{num_dec_layer}.loss_bbox_{modality}'] = loss_bbox_i
                    num_dec_layer += 1

                # DN query 的损失
                dn_pred_bboxes, dn_pred_logits = collections.defaultdict(list), collections.defaultdict(list)
                dn_mask_dicts = collections.defaultdict(list)
                for task_id, preds_dict in enumerate(preds_dicts, 0):
                    for dec_id in range(num_decoder):
                        pred_bbox = [preds_dict[0]['dn_center'][i][dec_id], preds_dict[0]['dn_height'][i][dec_id],
                                    preds_dict[0]['dn_dim'][i][dec_id], preds_dict[0]['dn_rot'][i][dec_id]]
                        if 'vel' in preds_dict[0]:
                            pred_bbox.append(preds_dict[0]['dn_vel'][i][dec_id])
                        pred_bbox = torch.cat(pred_bbox, dim=-1)
                        dn_pred_bboxes[dec_id].append(pred_bbox)
                        dn_pred_logits[dec_id].append(preds_dict[0]['dn_cls_logits'][i][dec_id])
                        dn_mask_dicts[dec_id].append(preds_dict[0]['dn_mask_dict'])
                dn_pred_bboxes = [dn_pred_bboxes[idx] for idx in range(num_decoder)]
                dn_pred_logits = [dn_pred_logits[idx] for idx in range(num_decoder)]
                dn_mask_dicts = [dn_mask_dicts[idx] for idx in range(num_decoder)]
                # 与正常 query 完全对称，但用的是 dn_center/dn_height/... 等 DN 预测值
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
        # 辅助损失
        if 'r_loss' in preds_dicts[0][0]:
            loss_dict['r_loss'] = preds_dicts[0][0]['r_loss']

        #  模态选择损失
        if 'qmod_sel_loss' in preds_dicts[0][0]:
            loss_dict['qmod_sel_loss'] = preds_dicts[0][0]['qmod_sel_loss']
            
        # for the purpose not to get loss (failure_pred) 模态权重损失  针对最后的预测结果
        if 'weight_list' in preds_dicts[0][0] and preds_dicts[0][0]['weight_list'] is not None:
            loss_weight_f_list = []
            for weight_list in preds_dicts[0][0]['weight_list']:
                weight_list = weight_list.squeeze(-1).transpose(0,1)
                # batch_size,_num_queries = weight_list.shape
                batch_size,_num_queries, _ = weight_list.shape
                weight_f_target = torch.tensor([i['modalmask'] for i in kwargs['img_metas']]).cuda()
                weight_f_target_expanded = weight_f_target.unsqueeze(1).repeat(1,_num_queries,1)
                loss_weight_f = self._criterion(weight_list,weight_f_target_expanded.float())/_num_queries
                # weight_f_target_expanded = torch.zeros(batch_size, _num_queries).cuda()
                # weight_f_target_expanded[weight_f_target[:, 0] == 1, :int(_num_queries/3)] = 1
                # weight_f_target_expanded[weight_f_target[:, 1] == 1, int(_num_queries/3):int(2*_num_queries/3)] = 1
                # weight_f_target_expanded[weight_f_target[:, 2] == 1, int(2*_num_queries/3):] = 1
                # loss_weight_f = F.binary_cross_entropy(weight_list, weight_f_target_expanded)
                loss_weight_f_list.append(loss_weight_f)
            loss_dict[f'loss_weight_f_{modality}'] = sum(loss_weight_f_list)
        if 'loss_weight_f' in preds_dicts[0][0] and preds_dicts[0][0]['loss_weight_f'] is not None:
            loss_dict[f'loss_weight_f_ensemble'] = preds_dicts[0][0]['loss_weight_f']
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):
        # 测试时的解码
        # 调用 TransFusionBBoxCoder.decode() 将模型输出解码为检测结果
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        # 调整 z 坐标：从重力中心转为底部中心
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
