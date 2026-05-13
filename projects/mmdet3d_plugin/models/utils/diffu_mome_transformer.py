# ------------------------------------------------------------------------
# Diffusion decoder utilities for MoME.
# Keeps MoME's multi-expert feature contract while adopting the
# time-conditioned reference refinement loop used by DiffuDETR/DiffuPETR.
# ------------------------------------------------------------------------

import copy
import math
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence
from mmdet.models.utils.transformer import inverse_sigmoid

REF_CLAMP_EPS = 1e-4


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x, pos_z), dim=-1)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimeStepBlock(nn.Module):
    def __init__(self,
                 channels,
                 emb_channels,
                 out_channels=256,
                 dims=1,
                 dropout=0.2,
                 use_scale_shift_norm=True):
        super(TimeStepBlock, self).__init__()
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )

    def forward(self, x, time_embed):
        if time_embed is None:
            return x
        emb_out = self.emb_layers(time_embed).type_as(x)
        if x.dim() == 3 and x.shape[1] == time_embed.shape[0]:
            emb_out = emb_out.unsqueeze(0)
        elif x.dim() == 3 and x.shape[0] == time_embed.shape[0]:
            emb_out = emb_out.unsqueeze(1)
        else:
            while emb_out.dim() < x.dim():
                emb_out = emb_out.unsqueeze(1)
        if self.use_scale_shift_norm:
            scale, shift = emb_out.chunk(2, dim=-1)
            h = x * (1 + scale) + shift
            return x + h
        return x + emb_out


@TRANSFORMER_LAYER.register_module()
class DiffuMOMETransformerDecoderLayer(BaseTransformerLayer):
    """PETR-compatible decoder layer with DiffuDETR timestep conditioning."""

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 with_cp=True,
                 **kwargs):
        super(DiffuMOMETransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.use_checkpoint = with_cp
        num_time_blocks = self.num_attn + len(self.ffns)
        time_step_embed = TimeStepBlock(
            channels=self.embed_dims,
            out_channels=self.embed_dims,
            emb_channels=self.embed_dims * 4,
            dims=1)
        self.time_step_embeds = nn.ModuleList(
            [copy.deepcopy(time_step_embed) for _ in range(num_time_blocks)])

    def apply_time_step(self, query, t, t_index, no_queries=None):
        if t is None:
            return query, t_index
        if no_queries is not None and len(no_queries) > 0:
            num_plain = int(no_queries[0])
            if num_plain > 0 and query.size(0) > num_plain:
                x = self.time_step_embeds[t_index](query[num_plain:], t)
                query = torch.cat([query[:num_plain], x], dim=0)
            else:
                query = self.time_step_embeds[t_index](query, t)
        else:
            query = self.time_step_embeds[t_index](query, t)
        return query, t_index + 1

    def _forward(self,
                 query,
                 key=None,
                 value=None,
                 query_pos=None,
                 key_pos=None,
                 t=None,
                 attn_masks=None,
                 query_key_padding_mask=None,
                 key_padding_mask=None,
                 no_queries=None,
                 **kwargs):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        t_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [attn_masks for _ in range(self.num_attn)]
            warnings.warn('Use same attn_mask in all attentions.')

        for layer in self.operation_order:
            if layer == 'self_attn':
                query, t_index = self.apply_time_step(query, t, t_index, no_queries)
                query = self.attentions[attn_index](
                    query,
                    query,
                    query,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == 'cross_attn':
                query, t_index = self.apply_time_step(query, t, t_index, no_queries)
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query
            elif layer == 'ffn':
                query, t_index = self.apply_time_step(query, t, t_index, no_queries)
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        return query

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                t=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                no_queries=None,
                **kwargs):
        if self.use_checkpoint and self.training:
            return cp.checkpoint(
                self._forward,
                query,
                key,
                value,
                query_pos,
                key_pos,
                t,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                no_queries)
        return self._forward(
            query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            t=t,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask,
            no_queries=no_queries,
            **kwargs)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DiffuMOMETransformerDecoder(TransformerLayerSequence):
    """MoME decoder with DiffuDETR-style timestep and reference refinement."""

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 ref_query_pos_scale=0.0,
                 ref_update_scale=0.0,
                 **kwargs):
        super(DiffuMOMETransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1] \
            if post_norm_cfg is not None else None
        self.ref_query_pos_scale = nn.Parameter(
            torch.tensor(float(ref_query_pos_scale), dtype=torch.float32))
        self.ref_update_scale = nn.Parameter(
            torch.tensor(float(ref_update_scale), dtype=torch.float32))

    def _make_query_pos(self, reference_points, query_embedding=None):
        if reference_points.size(-1) > 3:
            reference_points = torch.cat(
                [reference_points[..., 0:2], reference_points[..., 4:5]], dim=-1)
        reference_points = reference_points.clamp(REF_CLAMP_EPS, 1. - REF_CLAMP_EPS)
        return query_embedding(pos2posemb3d(reference_points)).transpose(0, 1)

    def _compose_query_pos(self, input_query_pos, reference_points, query_embedding=None):
        if query_embedding is None:
            return input_query_pos
        ref_query_pos = self._make_query_pos(reference_points, query_embedding)
        if input_query_pos is None:
            return ref_query_pos * self.ref_query_pos_scale.to(ref_query_pos)
        return input_query_pos + ref_query_pos * self.ref_query_pos_scale.to(ref_query_pos)

    def _update_reference(self, reference_points, reg_out):
        reference_points = reference_points.clamp(REF_CLAMP_EPS, 1. - REF_CLAMP_EPS)
        inv_reference = inverse_sigmoid(reference_points)
        reg_out = reg_out * self.ref_update_scale.to(reg_out)
        new_reference_points = reference_points.clone()
        if reference_points.size(-1) == 3:
            new_reference_points[..., 0:2] = (
                reg_out[..., 0:2] + inv_reference[..., 0:2]).sigmoid()
            new_reference_points[..., 2:3] = (
                reg_out[..., 4:5] + inv_reference[..., 2:3]).sigmoid()
        else:
            ref_dim = min(reference_points.size(-1), reg_out.size(-1))
            new_reference_points[..., :ref_dim] = (
                reg_out[..., :ref_dim] + inv_reference[..., :ref_dim]).sigmoid()
        return new_reference_points.clamp(REF_CLAMP_EPS, 1. - REF_CLAMP_EPS)

    def forward(self, query, *args, **kwargs):
        reg_branch = kwargs.pop('reg_branch', None)
        reference_points = kwargs.pop('reference_points', None)
        query_embedding = kwargs.pop('query_embedding', None)
        input_query_pos = kwargs.pop('query_pos', None)
        if reference_points is None:
            raise ValueError('DiffuMOME decoder requires reference_points.')
        if input_query_pos is None and query_embedding is None:
            raise ValueError('DiffuMOME decoder requires MOAD query_pos or query_embedding.')

        if not self.return_intermediate:
            query_pos = self._compose_query_pos(
                input_query_pos, reference_points, query_embedding)
            x = super().forward(query, *args, query_pos=query_pos, **kwargs)
            if reg_branch is not None:
                with torch.no_grad():
                    reference_points = self._update_reference(
                        reference_points, reg_branch[-1](x.transpose(0, 1))).detach()
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x, reference_points[None]

        intermediate = []
        intermediate_reference_points = []
        current_reference_points = reference_points.clamp(REF_CLAMP_EPS, 1. - REF_CLAMP_EPS)
        for layer in self.layers:
            layer_idx = len(intermediate)
            query_pos = self._compose_query_pos(
                input_query_pos, current_reference_points, query_embedding)
            query = layer(query, *args, query_pos=query_pos, **kwargs)
            output = self.post_norm(query) if self.post_norm is not None else query
            if reg_branch is not None:
                reg_out = reg_branch[layer_idx](output.transpose(0, 1))
                new_reference_points = self._update_reference(current_reference_points, reg_out)
                current_reference_points = new_reference_points.detach()
            else:
                new_reference_points = current_reference_points
            intermediate.append(output)
            intermediate_reference_points.append(new_reference_points)
        return torch.stack(intermediate), torch.stack(intermediate_reference_points)
