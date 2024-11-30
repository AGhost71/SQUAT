from pysgg.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pysgg.modeling.roi_heads.relation_head.model_msg_passing import (
    PairwiseFeatureExtractor,
)
import copy

def set_diff(a, b):
    combined = torch.cat((a, b))
    uniques, counts = combined.unique(return_counts=True)
    diff = uniques[counts == 1]
    
    return diff

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1)
        )
    
    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=0, keepdim=True).expand(z_local.shape[0], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out

    
class P2PDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, unary_output=False):
        super(P2PDecoder, self).__init__()
        
        self.layers = _get_clones(decoder_layer, num_layers) if num_layers > 0 else []
        self.num_layers = num_layers
        self.norm = norm
        self.unary_output = unary_output

    def forward(self, tgt, memory, ind, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: Target features (relationship features).
            memory: Memory containing object and initial relationship features.
            ind: Indices for selective quad attention.
        Returns:
            Final updated features and attention weights.
        """
        pair = tgt
        if self.num_layers == 0:
            return (memory[0], tgt) if self.unary_output else tgt

        # Collect attention weights from each layer
        self.attn_weights = []

        for mod in self.layers:
            unary, pair,attn = mod(pair, memory, ind, tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
            layer_combined_attn_weights = torch.cat(
                [attn[k].unsqueeze(1) for k in attn.keys()], dim=1
            )
        self.attn_weights.append(layer_combined_attn_weights)

        if self.norm is not None:
            pair = self.norm(pair)
            unary = self.norm(unary)
        
        return (unary, pair) if self.unary_output else (pair)
class P2PDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, norm_first=False):
        super(P2PDecoderLayer, self).__init__() 
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_node = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_e2e  = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_e2n = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_n2e  = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_n2n = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.linear1_unary = nn.Linear(d_model, dim_feedforward)
        self.dropout_unary = nn.Dropout(dropout)
        self.linear2_unary = nn.Linear(dim_feedforward, d_model)
        
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm1_unary = nn.LayerNorm(d_model)
        self.norm2_unary = nn.LayerNorm(d_model)
        self.norm3_unary = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout1_unary = nn.Dropout(dropout)
        self.dropout2_unary = nn.Dropout(dropout)
        self.dropout3_unary = nn.Dropout(dropout)
        
        self.activation = activation
        
    def forward(self, tgt, memory, ind, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        sparsified_pair = tgt
        sparsified_unary, entire_pair = memory 
        ind_pair, ind_e2e, ind_n2e = ind 
        
        sparsified_pair = self.norm1(sparsified_pair + self._sa_block(sparsified_pair, None, None))
        sparsified_unary = self.norm1_unary(sparsified_unary + self._sa_node_block(sparsified_unary, None, None))
        
        pair = torch.zeros_like(entire_pair)
        ind_ = torch.logical_not((torch.arange(pair.size(0), device=entire_pair.device).unsqueeze(1) == ind_pair).any(1))
        
        pair[ind_pair] = sparsified_pair
        pair[ind_] = entire_pair[ind_]
        pair_e2e = pair[ind_e2e]
        pair_n2e = pair[ind_n2e]
        attn_weights = {}
        updated_pair_e2e, attn_weights['e2e'] = self._mha_e2e(sparsified_pair, pair_e2e, None, None)
        updated_pair_e2n, attn_weights['e2n'] = self._mha_e2e(sparsified_pair, pair_e2e, None, None)
        updated_unary_n2e, attn_weights['n2e'] = self._mha_n2e(sparsified_unary, pair_n2e, None, None)
        updated_unary_n2n, attn_weights['n2n'] =self._mha_n2n(sparsified_unary, sparsified_unary, None, None)
        updated_pair = self.norm2(sparsified_pair + updated_pair_e2e \
                                                  + updated_pair_e2n) 
        updated_pair = self.norm3(updated_pair + self._ff_block_edge(updated_pair)) 
        
        updated_unary = self.norm2(sparsified_unary + updated_unary_n2e \
                                                    + updated_unary_n2n) 
        updated_unary = self.norm3(updated_unary + self._ff_block_node(updated_unary))
        for key in attn_weights:
            attn_weights[key] = F.softmax(attn_weights[key], dim=-1)
        
        return updated_unary, updated_pair,attn_weights

    def _sa_block(self, x, attn_mask, key_padding_mask): 
        x = self.self_attn(x, x, x, attn_mask=attn_mask, 
                           key_padding_mask=key_padding_mask, 
                           need_weights=False)[0]
        
        return self.dropout1(x)

    def _sa_node_block(self, x, attn_mask, key_padding_mask): 
        x = self.self_attn_node(x, x, x, attn_mask=attn_mask, 
                                key_padding_mask=key_padding_mask, 
                                need_weights=False)[0]
        
        return self.dropout1_unary(x)
    
    def _mha_e2n(self, x, mem, attn_mask, key_padding_mask):
        x,attn_weight = self.multihead_attn_e2n(x, mem, mem, 
                                      attn_mask=attn_mask, 
                                      key_padding_mask=key_padding_mask, 
                                      need_weights=True)
        
        return self.dropout2(x),attn_weight
    
    def _mha_e2e(self, x, mem, attn_mask, key_padding_mask):
        x,attn_weight = self.multihead_attn_e2e(x, mem, mem, 
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask, 
                                     need_weights=True)
        
        return self.dropout2(x),attn_weight
    
    def _mha_n2e(self, x, mem, attn_mask, key_padding_mask):
        x,attn_weight = self.multihead_attn_n2e(x, mem, mem, 
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask, 
                                     need_weights=True)
        
        return self.dropout2_unary(x),attn_weight
    
    def _mha_n2n(self, x, mem, attn_mask, key_padding_mask):
        x,attn_weight = self.multihead_attn_n2n(x, mem, mem, 
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask, 
                                     need_weights=True)
        
        return self.dropout2_unary(x),attn_weight

    def _ff_block_node(self, x): 
        x = self.linear2_unary(self.dropout_unary(self.activation(self.linear1_unary(x))))
        return self.dropout3_unary(x)
    
    def _ff_block_edge(self, x): 
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class SquatContext(nn.Module):
    def __init__(self, config, in_channels, hidden_dim=512, num_iter=3):
        super(SquatContext, self).__init__()

        self.cfg = config
        self.hidden_dim = hidden_dim
        self.num_iter = num_iter
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.pairwise_feature_extractor = PairwiseFeatureExtractor(config, in_channels)
        self.pooling_dim = self.pairwise_feature_extractor.pooling_dim

        self.obj_embedding = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.rel_embedding = nn.Sequential(
            nn.Linear(in_channels * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.pooling_dim),
            nn.ReLU()
        )
            
        norm_first = config.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.PRE_NORM
        decoder_layer = P2PDecoderLayer(self.pooling_dim, 8, self.hidden_dim * 2, norm_first=norm_first)
        num_layer = config.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.NUM_DECODER
        self.m2m_decoder = P2PDecoder(decoder_layer, num_layer)
        
        # Mask predictors for quad attention selection
        self.mask_predictor = MaskPredictor(self.pooling_dim, self.hidden_dim)
        self.mask_predictor_e2e = MaskPredictor(self.pooling_dim, self.hidden_dim)
        self.mask_predictor_n2e = MaskPredictor(self.pooling_dim, self.hidden_dim)
        self.rho = config.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.RHO
        self.beta = config.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.BETA

    def _get_map_idx(self, proposals, rel_pair_idxs):
        device = rel_pair_idxs[0].device
        rel_inds = []
        offset = 0
        obj_num = sum([len(proposal) for proposal in proposals])
        obj_obj_map = torch.zeros(obj_num, obj_num)

        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            obj_obj_map_i = (1 - torch.eye(len(proposal))).float()
            obj_obj_map[offset:offset + len(proposal), offset:offset + len(proposal)] = obj_obj_map_i
            rel_ind_i = rel_pair_idx + offset
            offset += len(proposal)
            rel_inds.append(rel_ind_i)

        rel_inds = torch.cat(rel_inds, dim=0)
        obj_obj_map = obj_obj_map.to(device)

        return rel_inds, obj_obj_map, obj_num

    def forward(self, roi_features, proposals, union_features, rel_pair_idxs):
        """
        Forward function for selective quad attention. Formats output for compatibility with Transformer.

        :param roi_features: Object (node) features.
        :param proposals: Bounding box proposals for each object.
        :param union_features: Relationship (edge) features.
        :param rel_pair_idxs: Pair indices for relationships.
        :return: Combined features in [batch_size, sequence_length, hidden_dim] format and attention weights.
        """
        
        # Obtain relationship indices, object-object map, and number of objects
        num_rels = [rel_pair_idx.size(0) for rel_pair_idx in rel_pair_idxs]
        rel_inds, obj_obj_map, obj_num = self._get_map_idx(proposals, rel_pair_idxs)
        
        # Embed object and relationship features
        feat_obj = self.obj_embedding(roi_features)
        augment_obj_feat, feat_pred = self.pairwise_feature_extractor(
            roi_features,
            union_features,
            proposals,
            rel_pair_idxs,
        )
        
        # Get mask-based selection indices
        feat_pred_batch_key = torch.split(feat_pred, num_rels)
        masks = [self.mask_predictor(k).squeeze(1) for k in feat_pred_batch_key]
        top_inds = [torch.topk(mask, int(mask.size(0) * self.rho))[1] for mask in masks]
        
        # Perform quad attention updates
        feat_pred_batch = [self.m2m_decoder(q.unsqueeze(1), (u.unsqueeze(1), p.unsqueeze(1)), (ind, ind_e2e, ind_n2e)).squeeze(1)
                           for p, u, q, ind, ind_e2e, ind_n2e in
                           zip(feat_pred_batch_key, augment_obj_feat, feat_pred_batch_key, top_inds, top_inds, top_inds)]
        
        # Concatenate object and relationship features in sequence format for Transformer
        combined_features = torch.cat([feat_obj] + feat_pred_batch, dim=0)
        
        # Reshape to [batch_size, sequence_length, hidden_dim]
        batch_size = len(proposals)
        combined_features = combined_features.view(batch_size, -1, self.hidden_dim)
        
        # Aggregate attention weights (example call, adjust based on `P2PDecoderLayer`)
        attn_weights = self.m2m_decoder.attn_weights # Assumes function exists in `P2PDecoder`
        aggregated_attn_weights = torch.stack(attn_weights, dim=0).mean(dim=0)
        return combined_features, aggregated_attn_weights


