# ------------------------------------------------------------------------
# DiffuMoME fused expert transformer.
# New file on purpose: the original MultiExpert stays untouched for loading
# and reproducing pretrained MoME.
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER

from .diffu_mome_transformer import timestep_embedding


@TRANSFORMER.register_module()
class DiffuMultiExpertFuse(BaseModule):
    """MoME expert decoding plus DiffuDETR-style decoder inputs.

    The feature construction is intentionally the same as MoME fused mode:
    fused uses BEV+RV memory, bev uses BEV memory, and img uses RV memory.
    Each enabled expert decodes all queries.
    """

    def __init__(self,
                 use_type_embed=True,
                 use_cam_embed=False,
                 decoder=None,
                 init_cfg=None):
        super(DiffuMultiExpertFuse, self).__init__(init_cfg=init_cfg)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        time_embed_dim = self.embed_dims * 4
        self.use_type_embed = use_type_embed
        self.use_cam_embed = use_cam_embed
        if self.use_type_embed:
            self.bev_type_embed = nn.Parameter(torch.randn(self.embed_dims))
            self.rv_type_embed = nn.Parameter(torch.randn(self.embed_dims))
        else:
            self.bev_type_embed = None
            self.rv_type_embed = None
        if self.use_cam_embed:
            self.cam_embed = nn.Sequential(
                nn.Conv1d(16, self.embed_dims, kernel_size=1),
                nn.BatchNorm1d(self.embed_dims),
                nn.Conv1d(self.embed_dims, self.embed_dims, kernel_size=1),
                nn.BatchNorm1d(self.embed_dims),
                nn.Conv1d(self.embed_dims, self.embed_dims, kernel_size=1),
                nn.BatchNorm1d(self.embed_dims))
        else:
            self.cam_embed = None
        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dims, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, x_img, bev_query_embed, rv_query_embed, bev_pos_embed,
                rv_pos_embed, img_metas, attn_masks=None, modalities=None,
                reg_branch=None, ref_points=None, pc_range=None,
                time_steps=None, query_content=None, query_embedding=None,
                no_queries=None):
        bs, c, h, w = x.shape
        bev_memory = rearrange(x, "bs c h w -> (h w) bs c")
        rv_memory = rearrange(x_img, "(bs v) c h w -> (v h w) bs c", bs=bs)
        bev_pos_embed = bev_pos_embed.unsqueeze(1).repeat(1, bs, 1)
        rv_pos_embed = rearrange(rv_pos_embed, "(bs v) h w c -> (v h w) bs c", bs=bs)

        if self.use_type_embed:
            bev_query_embed = bev_query_embed + self.bev_type_embed
            rv_query_embed = rv_query_embed + self.rv_type_embed

        rv_memory_v = rv_memory
        if self.use_cam_embed and self.cam_embed is not None:
            imgs2lidars = np.stack([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
            imgs2lidars = torch.from_numpy(imgs2lidars).float().to(x.device)
            imgs2lidars = imgs2lidars.flatten(-2).permute(0, 2, 1)
            imgs2lidars = self.cam_embed(imgs2lidars)
            imgs2lidars = imgs2lidars.permute(0, 2, 1).reshape(-1, self.embed_dims, 1, 1)
            imgs2lidars = imgs2lidars.repeat(1, 1, *x_img.shape[-2:])
            imgs2lidars = rearrange(imgs2lidars, '(bs v) c h w -> (v h w) bs c', bs=bs)
            rv_memory_v = rv_memory * imgs2lidars

        time_embed = None
        if time_steps is not None:
            time_embed = timestep_embedding(time_steps.float(), self.embed_dims)
            time_embed = self.time_embed(time_embed)

        out_decs = []
        ca_dict = dict(
            memory_l=[],
            memory_v_l=[],
            query_embed_l=[],
            pos_embed_l=[],
            zero_idx=[],
            inter_references=[])
        for modality in modalities:
            if modality == "fused":
                memory = torch.cat([bev_memory, rv_memory], dim=0)
                memory_v = memory
                pos_embed = torch.cat([bev_pos_embed, rv_pos_embed], dim=0)
                query_embed = bev_query_embed + rv_query_embed
            elif modality == "bev":
                memory = bev_memory
                memory_v = memory
                pos_embed = bev_pos_embed
                query_embed = bev_query_embed
            elif modality in ["img", "rv"]:
                memory = rv_memory
                memory_v = rv_memory_v
                pos_embed = rv_pos_embed
                query_embed = rv_query_embed
            else:
                raise ValueError(f'Unsupported DiffuMoME modality: {modality}')

            query_embed = query_embed.transpose(0, 1)
            target = (torch.zeros_like(query_embed) if query_content is None
                      else query_content.transpose(0, 1))
            decoder_kwargs = dict(
                query=target,
                key=memory,
                value=memory_v,
                query_pos=query_embed,
                key_pos=pos_embed,
                attn_masks=[attn_masks, None] if attn_masks is not None else None,
                reg_branch=reg_branch,
            )
            if query_embedding is not None:
                decoder_kwargs.update(
                    reference_points=ref_points,
                    query_embedding=query_embedding,
                    t=time_embed,
                    no_queries=no_queries)

            out_dec = self.decoder(**decoder_kwargs)
            inter_references = None
            if isinstance(out_dec, tuple):
                out_dec, inter_references = out_dec
            out_decs.append(out_dec.transpose(1, 2))
            ca_dict['memory_l'].append(memory)
            ca_dict['memory_v_l'].append(memory_v)
            ca_dict['query_embed_l'].append(query_embed)
            ca_dict['pos_embed_l'].append(pos_embed)
            ca_dict['zero_idx'].append(
                torch.ones(out_dec.shape[0], out_dec.shape[2], out_dec.shape[1],
                           dtype=torch.bool, device=out_dec.device))
            ca_dict['inter_references'].append(inter_references)

        ca_dict = dict(
            memory_l=ca_dict['memory_l'],
            memory_v_l=ca_dict['memory_v_l'],
            query_embed_l=ca_dict['query_embed_l'],
            pos_embed_l=ca_dict['pos_embed_l'],
            zero_idx=ca_dict['zero_idx'],
            inter_references=ca_dict['inter_references'])
        return out_decs, ca_dict
