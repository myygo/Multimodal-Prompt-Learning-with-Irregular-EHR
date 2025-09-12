#!/usr/bin/env python3
"""
[最终修正版] Baseline 方法。
- 彻底解决了所有 __init__ 和 super() 调用不匹配的问题。
- 确保所有基线模型都公平地使用与主模型一致的特征提取器。
- 清理了冗余的模型定义。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel

# 导入主模型的模块和参数，确保完全一致
from my_model import ICUHyperParams, BertForRepresentation 

class BaseBimodalModel(nn.Module):
    """
    最终版基类：特征提取部分与主模型ICUPromptModel完全看齐。
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        # 加载与主模型完全一致的超参数
        hyp_params = ICUHyperParams()

        # 定义与主模型完全一致的投影层
        self.proj_t = nn.Conv1d(hyp_params.orig_d_t, hyp_params.proj_dim, kernel_size=1, padding=0, bias=False)
        self.proj_n = nn.Conv1d(hyp_params.orig_d_n_reg, hyp_params.proj_dim, kernel_size=1, padding=0, bias=False)
        
        # 定义与主模型一致的序列长度和维度
        self.nlen = hyp_params.seq_len[1]
        self.d_n = hyp_params.proj_dim
        
        # 注意：text_encoder 将从外部注入，以保证所有模型共享同一个实例
        self.text_encoder = None

    def set_text_encoder(self, text_encoder_instance):
        """一个用于从外部注入共享的text_encoder的方法"""
        self.text_encoder = text_encoder_instance
        print("Freezing parameters of the text encoder for baseline model...")
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def _get_device(self) -> torch.device:
        return next(self.parameters()).device

    def _get_batch_size(self, *tensors: Optional[torch.Tensor]) -> int:
        for tensor in tensors:
            if torch.is_tensor(tensor) and tensor.numel() > 0:
                return tensor.size(0)
        return 0

    def _get_differentiable_placeholder(self, batch_size: int, feature_dim: int) -> torch.Tensor:
        dummy_param = next(self.parameters(), None)
        placeholder = torch.zeros(batch_size, feature_dim, device=self._get_device())
        if dummy_param is not None:
            return placeholder * dummy_param.new_tensor(0)
        return placeholder
    
    def _encode_timeseries(self, x_ts: Optional[torch.Tensor], reg_ts: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        batch_size = self._get_batch_size(x_ts, reg_ts)
        if batch_size == 0: return None
        device = self._get_device()

        if reg_ts is not None and reg_ts.numel() > 0: features = reg_ts
        elif x_ts is not None and x_ts.numel() > 0: features = x_ts
        else: return None

        _, seq_len, feature_dim = features.shape
        if feature_dim > 34: features = features[:, :, :34]
        elif feature_dim < 34:
            padding = torch.zeros(batch_size, seq_len, 34 - feature_dim, device=device)
            features = torch.cat([features, padding], dim=2)
        
        features = features.transpose(1, 2)
        if seq_len != self.nlen:
            features = F.interpolate(features, size=self.nlen, mode='linear', align_corners=False)
        
        projected_features = self.proj_n(features)
        final_vector = torch.mean(projected_features, dim=2)
        
        return final_vector

    def _encode_text(self, input_ids: Optional[torch.Tensor], attn_mask: Optional[torch.Tensor], note_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.text_encoder is None:
            raise RuntimeError("Text encoder has not been set. Please call set_text_encoder() before using.")
        if input_ids is None or input_ids.numel() == 0:
            return None

        note_vectors = self.text_encoder(input_ids.permute(1, 0, 2), attn_mask.permute(1, 0, 2))
        note_level_mask = note_mask.permute(1, 0).unsqueeze(-1).float()
        masked_note_vectors = note_vectors * note_level_mask
        summed_vectors = masked_note_vectors.sum(dim=0)
        num_real_notes = note_level_mask.sum(dim=0).clamp(min=1e-9)
        patient_vector = summed_vectors / num_real_notes
        expanded_vector = patient_vector.unsqueeze(-1)
        projected_vector = self.proj_t(expanded_vector)
        final_vector = projected_vector.squeeze(-1)
        
        return final_vector

class LowerBoundModel(BaseBimodalModel):
    def __init__(self, modality_combination: List[str], num_classes: int):
        super().__init__(num_classes=num_classes)
        self.modalities = modality_combination
        fusion_dim = sum(64 for m in ['timeseries', 'text'] if m in self.modalities)
        if fusion_dim == 0:
             raise ValueError("LowerBoundModel 必须至少有一个模态。")
        self.classifier = nn.Sequential(nn.Linear(fusion_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, num_classes))
    
    def forward(self, **kwargs: Optional[torch.Tensor]):
        batch_size = self._get_batch_size(*kwargs.values())
        if batch_size == 0: return self._get_differentiable_placeholder(0, self.num_classes)
            
        final_features = []
        if 'timeseries' in self.modalities:
            ts_features = self._encode_timeseries(kwargs.get('x_ts'), kwargs.get('reg_ts'))
            if ts_features is None: ts_features = self._get_differentiable_placeholder(batch_size, 64)
            final_features.append(ts_features)

        if 'text' in self.modalities:
            text_features = self._encode_text(kwargs.get('input_ids_sequences'), kwargs.get('attn_mask'), kwargs.get('note_mask'))
            if text_features is None: text_features = self._get_differentiable_placeholder(batch_size, 64)
            final_features.append(text_features)
        
        if not final_features: return self._get_differentiable_placeholder(batch_size, self.num_classes)
            
        concatenated_features = torch.cat(final_features, dim=1)
        return self.classifier(concatenated_features)

class ModalitySubstitutionModel(BaseBimodalModel):
    def __init__(self, num_classes: int):
        super().__init__(num_classes=num_classes)
        self.default_ts_feat = nn.Parameter(torch.randn(1, 64) * 0.02)
        self.default_text_feat = nn.Parameter(torch.randn(1, 64) * 0.02)
        self.fusion_classifier = nn.Sequential(nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, num_classes))
    
    def forward(self, **kwargs: Optional[torch.Tensor]):
        batch_size = self._get_batch_size(*kwargs.values())
        if batch_size == 0: return self._get_differentiable_placeholder(0, self.num_classes)

        ts_features = self._encode_timeseries(kwargs.get('x_ts'), kwargs.get('reg_ts'))
        if ts_features is None: ts_features = self.default_ts_feat.expand(batch_size, -1)
        
        text_features = self._encode_text(kwargs.get('input_ids_sequences'), kwargs.get('attn_mask'), kwargs.get('note_mask'))
        if text_features is None: text_features = self.default_text_feat.expand(batch_size, -1)
        
        return self.fusion_classifier(torch.cat([ts_features, text_features], dim=1))

class ModalityDropoutModel(BaseBimodalModel):
    def __init__(self, num_classes: int, dropout_rate: float = 0.7):
        super().__init__(num_classes=num_classes)
        self.dropout_rate = dropout_rate
        self.modality_weights = nn.Parameter(torch.ones(2))
        self.fusion_classifier = nn.Sequential(
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, **kwargs: Optional[torch.Tensor]):
        x_ts, reg_ts, input_ids, attn_mask, note_mask = [kwargs.get(k) for k in 
            ['x_ts', 'reg_ts', 'input_ids_sequences', 'attn_mask', 'note_mask']]
        
        if self.training:
            ts_present = x_ts is not None or reg_ts is not None
            text_present = input_ids is not None
            drop_ts = ts_present and random.random() < self.dropout_rate
            drop_text = text_present and random.random() < self.dropout_rate
            
            if ts_present and text_present and drop_ts and drop_text:
                if random.random() < 0.5: drop_ts = False
                else: drop_text = False
            
            if drop_ts: x_ts, reg_ts = None, None
            if drop_text: input_ids, attn_mask, note_mask = None, None, None
        
        ts_feat = self._encode_timeseries(x_ts, reg_ts)
        text_feat = self._encode_text(input_ids, attn_mask, note_mask)
        
        batch_size = self._get_batch_size(x_ts, reg_ts, input_ids, kwargs.get('label'))
        if batch_size == 0: return self._get_differentiable_placeholder(0, self.num_classes)
            
        features, weights_to_use = [], []
        if ts_feat is not None: 
            features.append(ts_feat)
            weights_to_use.append(self.modality_weights[0])
        if text_feat is not None: 
            features.append(text_feat)
            weights_to_use.append(self.modality_weights[1])
        
        if not features:
            fused_feat = self._get_differentiable_placeholder(batch_size, 64)
        elif len(features) == 2:
            weights = F.softmax(torch.stack(weights_to_use), dim=0)
            fused_feat = weights[0] * features[0] + weights[1] * features[1]
        else:
            fused_feat = features[0]
        
        return self.fusion_classifier(fused_feat)

def create_adapted_baselines(base_bert_model, num_classes: int = 2) -> Dict[str, nn.Module]:
    print("为所有基线模型配置共享的特征提取器...")
    hyp_params = ICUHyperParams()
    shared_text_encoder = BertForRepresentation(hyp_params, base_bert_model)
    
    baselines = {
        'LB_timeseries_only': LowerBoundModel(modality_combination=['timeseries'], num_classes=num_classes),
        'LB_text_only': LowerBoundModel(modality_combination=['text'], num_classes=num_classes),
        'LB_both_modalities': LowerBoundModel(modality_combination=['timeseries', 'text'], num_classes=num_classes),
        'MS': ModalitySubstitutionModel(num_classes=num_classes),
        'MD': ModalityDropoutModel(num_classes=num_classes, dropout_rate=0.7),
    }
    
    for model_name, model in baselines.items():
        if 'text' in model_name or 'both' in model_name or model_name in ['MS', 'MD']:
             if hasattr(model, 'set_text_encoder'):
                 model.set_text_encoder(shared_text_encoder)
    
    print("所有基线模型创建并配置完毕。")
    return baselines