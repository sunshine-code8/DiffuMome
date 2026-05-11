# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
import matplotlib.pyplot as plt
import cv2

@TRANSFORMER.register_module()
class MultiExpert(BaseModule):
    def __init__(
            self,
            use_type_embed=True,    # 是否使用 BEV/RV 类型嵌入
            use_cam_embed=False,   # 是否使用相机嵌入
            window_sizes=[15,5],   # LAM 局部注意力窗口大小 [图像窗口, BEV窗口]
            encoder=None,     # AQR 路由用的 Transformer (1层)
            decoder=None,    # 三个专家共享的 Transformer (6层)
            init_cfg=None
    ):
        super(MultiExpert, self).__init__(init_cfg=init_cfg)
        # 构建 encoder（AQR 路由用）
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
            self.e_num_heads = self.encoder.layers[0].attentions[0].num_heads
            self.selected_cls = nn.Linear(256, 3)
        else:
            self.encoder = None
        # 构建 decoder（三个专家共享）
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.use_type_embed = use_type_embed
        self.use_cam_embed = use_cam_embed
        if self.use_type_embed:
            ### 当 fused 专家把 BEV 和 RV 特征拼接在一起作为 memory 时，模型无法区分哪些 token 来自 BEV、哪些来自 RV。类型嵌入就是给它们加上"身份标记"
            self.bev_type_embed = nn.Parameter(torch.randn(self.embed_dims))
            self.rv_type_embed = nn.Parameter(torch.randn(self.embed_dims))
        else:
            self.bev_type_embed = None
            self.rv_type_embed = None

        if self.use_cam_embed:  # 将 4×4 的相机外参矩阵（展平为 16 维）编码为 256 维的嵌入，让模型知道每个图像特征来自哪个相机
            self.cam_embed = nn.Sequential(
                nn.Conv1d(16, self.embed_dims, kernel_size=1),
                nn.BatchNorm1d(self.embed_dims),
                nn.Conv1d(self.embed_dims, self.embed_dims, kernel_size=1),
                nn.BatchNorm1d(self.embed_dims),
                nn.Conv1d(self.embed_dims, self.embed_dims, kernel_size=1),
                nn.BatchNorm1d(self.embed_dims)
            )
        else:
            self.cam_embed = None
        self.window_sizes = window_sizes   # 用于 LAM（局部注意力掩码） 0是图像的窗口特征   1是BEV特征即Lidar

    def init_weights(self):
        # 对权重维度 > 1 的参数（即矩阵，不是偏置向量）使用 Xavier 均匀初始化。这是 DETR 系列的标准做法，保证训练初期梯度稳定
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def AQR_with_LAM(self, ca_dict, ref_points, pc_range, x_img, img_metas, reg_branch=None):  ### AQR 路由 + 局部注意力掩码
        """
        LAM：为每个 query 构造局部注意力掩码，限制 query 只看附近的特征
        AQR：通过 encoder + 分类器，决定每个 query 分配给哪个专家
        """
        # 参考点坐标还原  , 将归一化 [0,1] 的参考点还原为真实 LiDAR 3D 坐标
        _ref_points = ref_points.new_zeros(ref_points.shape)
        _ref_points[..., 0:1] = ref_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        _ref_points[..., 1:2] = ref_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        _ref_points[..., 2:3] = ref_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        # 将 3D 参考点投影到 6 个相机的 2D 像素坐标
        _position = torch.cat([_ref_points],dim=-1)
        _matrices = np.stack([np.stack(i['lidar2img']) for i in img_metas])
        _matrices = torch.tensor(_matrices).float().cuda()
        batch, _num_queries, _ =  _position.shape

        if False: # gt visualize
            _position = img_metas[0]['gt_bboxes_3d']._data.tensor[:,:3].unsqueeze(0).cuda()
            _position[:,:,-1] += img_metas[0]['gt_bboxes_3d']._data.tensor[:,5].unsqueeze(0).cuda()*0.5
            batch, _num_queries,_ = img_metas[0]['gt_bboxes_3d']._data.tensor[:,:3].unsqueeze(0).shape
            _matrices = _matrices[0:1]

        # 加齐次坐标 → 用投影矩阵投影到每个相机
        _position_4d = torch.cat([_position, torch.ones((batch, _num_queries, 1)).cuda()], dim=-1)

        # from mmdetection3d/mmdet3d/core/visualizer/image_vis.py
        pts_2d = torch.einsum('bni,bvij->bvnj', _position_4d, _matrices.transpose(2,3))
        # 透视除法：除以深度 z 得到像素坐标 (u, v)
        pts_2d[..., 2] = torch.clip(pts_2d[..., 2], min=1e-5, max=99999)
        pts_2d[..., 0] /= pts_2d[..., 2]
        pts_2d[..., 1] /= pts_2d[..., 2]
        # 检查每个参考点投影后是否落在每个相机的视野内
        fov_inds = ((pts_2d[..., 0] < img_metas[0]['img_shape'][0][1])
                    & (pts_2d[..., 0] >= 0)
                    & (pts_2d[..., 1] < img_metas[0]['img_shape'][0][0])
                    & (pts_2d[..., 1] >= 0))

        # 找到每个 query 对应的第一个有效相机 , 在 6 个相机中找到每个 query 投影到的第一个有效相机
        _, num_views, _, _ = pts_2d.shape
        _fov_inds = torch.cat([torch.full((fov_inds.shape[0],1,fov_inds.shape[-1]),False).cuda(),fov_inds],dim=1)

        first_valid_view = _fov_inds.to(torch.float32).argmax(dim=1)
        first_valid_view[first_valid_view == 0] = -1
        first_valid_view[first_valid_view != -1] -= 1
        batch_indices = torch.arange(batch).unsqueeze(1).expand(-1, _num_queries)
        point_indices = torch.arange(_num_queries).unsqueeze(0).expand(batch, -1)
        #提取每个 query 在其有效相机中的像素坐标 (v, u)
        mask = first_valid_view != -1
        selected_pts = pts_2d[
            batch_indices[mask],
            first_valid_view[mask],
            point_indices[mask]
        ]
        pts_pers = torch.full((batch, _num_queries, 2), float('nan'), device=pts_2d.device)
        pts_pers[batch_indices[mask], point_indices[mask]] = selected_pts[...,[1,0]] # H,W
        ori_H,ori_W = img_metas[0]['img_shape'][0][:2]
        feat_H, feat_W = x_img.shape[2:]
        ratio = torch.tensor(feat_H/ori_H)
        # 拼接相机编号，得到完整的透视位置信息
        pts_pers = torch.cat([first_valid_view.unsqueeze(-1),pts_pers],dim=-1)

        if False: # visualize
            def visualize_img(pts_pers, img_metas, cam_view=0, idx='0'):
                imgfov_pts_2d = pts_pers[:1,:,:][pts_pers[:1,:,0]==cam_view][:,1:][:,[1,0]] # W, H
                img = img_metas[0]['img'][0][cam_view].permute(1,2,0).cpu().numpy()

                cmap = plt.cm.get_cmap('hsv', 256)
                cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
                mean=[103.530, 116.280, 123.675]
                std=[57.375, 57.120, 58.395]
                mean = torch.tensor(mean).view(1, 1, 3).numpy()
                std = torch.tensor(std).view(1, 1, 3).numpy()
                denormalized_image = img * std + mean
                img = np.clip(denormalized_image, 0, 255).astype(np.uint8)
                img = img.copy()
                for i in range(imgfov_pts_2d.shape[0]):
                    # depth = imgfov_pts_2d[i, 2].item()
                    depth = 70
                    color = cmap[np.clip(int(70 * 10 / depth), 0, 255), :]
                    cv2.circle(
                        img,
                        center=(int(np.round(imgfov_pts_2d[i, 0].item())),
                                int(np.round(imgfov_pts_2d[i, 1].item()))),
                        radius=1,
                        color=tuple(color),
                        thickness=3,
                    )
                output_path = f'projected_pts_img_{idx}.jpg'
                cv2.imwrite(output_path, img.astype(np.uint8))
            visualize_img(pts_pers, img_metas, 0, '0')
            visualize_img(pts_pers, img_metas, 1, '1')
            visualize_img(pts_pers, img_metas, 2, '2')
            visualize_img(pts_pers, img_metas, 3, '3')
            visualize_img(pts_pers, img_metas, 4, '4')
            visualize_img(pts_pers, img_metas, 5, '5')
        # 坐标映射到特征图尺度
        pts_pers[:,:,1:] = torch.floor(pts_pers[:,:,1:]*ratio) # H,W
        pts_pers[pts_pers[:,:,0]==-1] = 0.0
        # 计算每个 query 在 BEV 特征图上的位置
        pts_bev = torch.floor((_ref_points[..., :2] + 54.0) * (180 / 108))[:,:,[1,0]] # feature dim = x,y -> change center coordinates to y,x
        pts_idx = pts_bev[:,:,0]*180+pts_bev[:,:,1]

        # camera attn mask
        # 构造图像特征的局部注意力掩码（LAM - 图像部分）
        # 对每个 query，在其投影到的相机位置周围开一个 15×15 的窗口，窗口内的像素可见，窗口外的不可见
        pts_pers_idx = pts_pers[:,:,0]*40*100+pts_pers[:,:,1]*100+pts_pers[:,:,2]
        batch_size, num_queries, _ = pts_pers.shape
        total_elements = 6*40*100
        H,W=40,100
        stride_view = 40*100
        stride_h = 100
        window_size = self.window_sizes[0]
        # Generate window offsets
        offsets = torch.arange(-(window_size // 2), window_size // 2 + 1, device=pts_pers_idx.device).cuda()
        window_offsets = offsets.unsqueeze(1) * stride_h + offsets.unsqueeze(0)
        window_offsets = window_offsets.view(-1)

        # Apply window offsets to query indices
        indices = pts_pers_idx.unsqueeze(-1) + window_offsets.unsqueeze(0).unsqueeze(0)


        # Calculate row and column indices
        query_rows = (pts_pers_idx % (H * W)) // W
        query_cols = pts_pers_idx % W
        index_rows = (indices % (H * W)) // W
        index_cols = indices % W

        # Check row and column validity to filter patches on other views
        # # 边界检查：确保窗口不跨越不同的行或不同的相机
        valid_row = (index_rows - query_rows.unsqueeze(-1)).abs() <= window_size // 2
        valid_column = (index_cols - query_cols.unsqueeze(-1)).abs() <= window_size // 2

        # Create overall validity mask
        valid = valid_row & valid_column

        # Handle indices that are out of range  处理超出边界的
        indices = torch.clamp(indices, 0, total_elements - 1).long()

        # Generate attention mask (initialize all as True)
        img_attention_mask = torch.ones(batch_size, num_queries, total_elements, dtype=torch.bool, device=pts_pers_idx.device)  # 初始化全 True（全部不可见）
        batch_indices = torch.arange(batch_size, device=pts_pers_idx.device).long().view(-1, 1, 1)
        query_indices_range = torch.arange(num_queries, device=pts_pers_idx.device).long().view(1, -1, 1)

        valid_indices = valid.nonzero(as_tuple=True)
        img_attention_mask[
            batch_indices.expand_as(indices)[valid_indices[0], valid_indices[1], valid_indices[2]],
            query_indices_range.expand_as(indices)[valid_indices[0], valid_indices[1], valid_indices[2]],
            indices[valid_indices]
        ] = False   # 窗口内的位置设为 False（可见）

        # lidar attn mask
        #构造 BEV 特征的局部注意力掩码（LAM - BEV 部分）
        batch_size, num_queries = pts_idx.shape
        total_elements = 180*180
        row_stride = 180
        window_size = self.window_sizes[1]
        offsets = torch.arange(-(window_size // 2), window_size // 2 + 1).cuda()
        y_offsets, x_offsets = torch.meshgrid(offsets, offsets)
        window_offsets = (y_offsets * row_stride + x_offsets).reshape(-1)
        # Calculate the indices for a window for each query
        indices = pts_idx.unsqueeze(-1) + window_offsets.unsqueeze(0).unsqueeze(0)
        # Keep only valid indices (greater than or equal to 0 and less than total_elements)
        valid_indices = (indices >= 0) & (indices < total_elements)
        # Boundary check: exclude cases where rows change
        query_columns = pts_idx % row_stride
        window_columns = (indices % row_stride).float() - query_columns.unsqueeze(-1).float()
        valid_indices &= (window_columns.abs() <= window_size // 2)
        # Generate attention mask (initialize all as True)
        lidar_attention_mask = torch.ones(batch_size, num_queries, total_elements, dtype=torch.bool, device=pts_idx.device)

        batch_indices = torch.arange(batch_size, device=pts_idx.device).view(-1, 1, 1).expand_as(indices)
        query_indices_range = torch.arange(num_queries, device=pts_idx.device).view(1, -1, 1).expand_as(indices)

        # Update the mask using only valid indices
        valid_mask = valid_indices & (indices < total_elements)
        lidar_attention_mask[
            batch_indices[valid_mask],
            query_indices_range[valid_mask],
            indices[valid_mask].long()
        ] = False

        # 合并掩码
        # LAM 的意义：如果不加掩码，每个 query 要对全部 56400 个特征做注意力，计算量巨大且容易被远处无关特征干扰。加了 LAM 后，每个 query 只关注自己附近的小区域，既减少计算量，又提高路由准确性
        fusion_attention_mask = torch.cat([lidar_attention_mask, img_attention_mask], dim=-1)
        fusion_attention_mask = fusion_attention_mask.unsqueeze(1).repeat(1,self.e_num_heads,1,1).flatten(0,1)

        # AQR 路由决策
        target = torch.zeros_like(ca_dict['query_embed_l'][0])
        # encoder（1 层 Transformer）让每个 query 在 LAM 限制下看融合特征
        target = self.encoder(
            query=target,
            key=ca_dict['memory_l'][0],  # ← 融合特征 (BEV+RV)
            value=ca_dict['memory_v_l'][0],
            query_pos=ca_dict['query_embed_l'][0],
            key_pos=ca_dict['pos_embed_l'][0],
            attn_masks=[fusion_attention_mask],
            reg_branch=reg_branch
        )
        if target.shape[0] ==0:
            target = target.squeeze(0).transpose(1,0)
        else:
            target = target[-1].transpose(1,0)
        batch_size,_num_queries, num_dims = target.shape
        #  target = target.reshape(-1, num_dims)
        target = self.selected_cls(target)  # 每个 query 得到 3 个分数，对应 [fused, bev, img] 三个专家
        # AQR 路由损失（训练时）
        #modalmask 是传感器可用性标记
        if self.training:
            weight_f_target = torch.tensor([i['modalmask'] for i in img_metas]).cuda()
            weight_f_target_expanded = weight_f_target.unsqueeze(1).repeat(1,_num_queries,1)
            #  target_labels = torch.argmax(target, dim=-1)
            weight_f_target_expanded = weight_f_target_expanded.reshape(-1, 3)
            target_r = target.reshape(-1, 3)
            self._criterion = nn.CrossEntropyLoss()
            loss_weight_f = self._criterion(target_r, weight_f_target_expanded.float())
        else:
            loss_weight_f = False

        print ("AQR routing selection distribution:",  target)

        return target, loss_weight_f, _

    def forward(self, x, x_img, bev_query_embed, rv_query_embed, bev_pos_embed, rv_pos_embed, img_metas,
                attn_masks=None, modalities=None, reg_branch=None, ref_points=None, pc_range=None):
        # 特征维度重排 , 将 4D 特征图转为 Transformer 需要的序列格式 [序列长度, batch, 维度]
        bs, c, h, w = x.shape
        bev_memory = rearrange(x, "bs c h w -> (h w) bs c")  # [bs, n, c, h, w] -> [n*h*w, bs, c]
        rv_memory = rearrange(x_img, "(bs v) c h w -> (v h w) bs c", bs=bs)
        bev_pos_embed = bev_pos_embed.unsqueeze(1).repeat(1, bs, 1)  # [bs, n, c, h, w] -> [n*h*w, bs, c]
        rv_pos_embed = rearrange(rv_pos_embed, "(bs v) h w c -> (v h w) bs c", bs=bs)

        # 类型嵌入 , 给 Query 的 BEV 和 RV 位置编码加上类型标记
        if self.use_type_embed:
            bev_query_embed = bev_query_embed + self.bev_type_embed
            rv_query_embed = rv_query_embed + self.rv_type_embed

        if self.use_cam_embed:
            imgs2lidars = np.stack([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
            imgs2lidars = torch.from_numpy(imgs2lidars).float().to(x.device)
            imgs2lidars = imgs2lidars.flatten(-2).permute(0, 2, 1)
            imgs2lidars = self.cam_embed(imgs2lidars)
            imgs2lidars = imgs2lidars.permute(0, 2, 1).reshape(-1, self.embed_dims, 1, 1)
            imgs2lidars = imgs2lidars.repeat(1, 1, *x_img.shape[-2:])
            imgs2lidars = rearrange(imgs2lidars, '(bs v) c h w -> (v h w) bs c', bs=bs)

        # 准备 AQR 输入
        # 初始化输出列表和中间信息字典。ca_dict 用于存储每个阶段的 memory 等信息（给后续的 AQR 和 CFA 用）
        out_decs = []
        ca_dict = {"memory_l": [], "memory_v_l": [], "query_embed_l": [], "pos_embed_l": [], "zero_idx": []}

        # 构建融合特征，作为 AQR 的输入
        memory, pos_embed = (torch.cat([bev_memory, rv_memory], dim=0),
                             torch.cat([bev_pos_embed, rv_pos_embed], dim=0))
        memory_v = memory
        query_embed = bev_query_embed + rv_query_embed
        query_embed = query_embed.transpose(0, 1)  # [bs, num_query, dim] -> [num_query, bs, dim]

        # 把融合特征存入 ca_dict_fp，传给 AQR_with_LAM。AQR 的 encoder 会用这些作为 key/value
        ca_dict_fp = copy.deepcopy(ca_dict)
        ca_dict_fp['memory_l'].append(memory)
        ca_dict_fp['memory_v_l'].append(memory_v)
        ca_dict_fp['query_embed_l'].append(query_embed)
        ca_dict_fp['pos_embed_l'].append(pos_embed)
        qmod_sel_loss = x.new_tensor(0.0)

        # bev_query_embed 这里的形状是 [bs, num_query, dim]
        q_sel = torch.zeros(
            bev_query_embed.shape[:2],
            dtype=torch.long,
            device=bev_query_embed.device
        )

        # 三个专家分别解码（核心循环）
        for idx, modality in enumerate(modalities):
            # 根据模态准备不同的 memory
            if modality == "fused":
                memory, pos_embed = (torch.cat([bev_memory, rv_memory], dim=0),
                                     torch.cat([bev_pos_embed, rv_pos_embed], dim=0))
                memory_v = memory
                query_embed = bev_query_embed + rv_query_embed
            elif modality == "bev":
                memory, pos_embed = bev_memory, bev_pos_embed
                memory_v = memory
                query_embed = bev_query_embed
            else:
                memory, pos_embed = rv_memory, rv_pos_embed
                memory_v = memory
                if self.cam_embed is not None:
                    memory_v = memory_v * imgs2lidars
                query_embed = rv_query_embed

            query_embed = query_embed.transpose(0, 1)  # [bs, num_query, dim] -> [num_query, bs, dim]
            target = torch.zeros_like(query_embed)

            out_dec_b = []
            ref_b = []
            target_o = torch.zeros((self.decoder.num_layers, target.shape[0], target.shape[1], target.shape[2])).to(target.device)
            for b in range(bs):

                # query selection
                sel = (q_sel[b] == idx)
                if sel.sum() == 0:
                    continue
                if idx == 0 :
                    print ("lidar失效")
                    print ("选择fuse")

                target_s = target[:, b][sel].unsqueeze(1)
                query_embed_s = query_embed[:, b][sel].unsqueeze(1)
                if attn_masks is not None:
                    attn_masks_s = attn_masks[sel][:, sel]
                else:
                    attn_masks_s = None
                memory_s = memory[:, b].unsqueeze(1)
                memory_v_s = memory_v[:, b].unsqueeze(1)
                pos_embed_s = pos_embed[:, b].unsqueeze(1)
                # out_dec: [num_layers, num_query, bs, dim]
                out_dec = self.decoder(
                    query=target_s,
                    key=memory_s,
                    value=memory_v_s,
                    query_pos=query_embed_s,
                    key_pos=pos_embed_s,
                    attn_masks=[attn_masks_s, None],
                    reg_branch=reg_branch,
                )
                target_o[:, :, b][:, sel] = out_dec[:, :, 0]
            zero_idx = (target_o.sum(-1)!=0).transpose(1,2)
            out_decs.append(target_o.transpose(1, 2))
            ca_dict['memory_l'].append(memory.clone())
            ca_dict['memory_v_l'].append(memory_v.clone())
            ca_dict['query_embed_l'].append(query_embed.clone())
            ca_dict['pos_embed_l'].append(pos_embed.clone())
            ca_dict['zero_idx'].append(zero_idx.clone())
        ca_dict['qmod_sel_loss'] = qmod_sel_loss
        return out_decs, ca_dict
