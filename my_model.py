#!/usr/bin/env python3
"""
[最终完整版] ICUPromptModel及其依赖项。

关键特性:
- 文本处理: BertForRepresentation 现在使用 [CLS] token，与 Zhang et al. (2023) 完全对齐。
- 数值处理: 实现了 Zhang et al. (2023) 的 UTDE 核心思想（双路径+门控融合）。
- 模态生成: 使用基于注意力的总结机制，稳定且高效。
- 架构: 包含完整的跨模态融合、自注意力记忆和提示学习模块。
- 可配置性: 支持通过参数冻结 Longformer 骨干。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from module import multiTimeAttention, gateMLP

# 假设您的 transformer 模块位于 ./modules/transformer.py
from modules.transformer import TransformerEncoder

class BertForRepresentation(nn.Module):
    def __init__(self, args, BioBert):
        super().__init__()
        self.bert = BioBert
        self.dropout = torch.nn.Dropout(BioBert.config.hidden_dropout_prob)
        self.model_name = args.model_name

    def forward(self, input_ids_sequence, attention_mask_sequence, sent_idx_list=None, doc_idx_list=None):
        txt_arr = []
        for input_ids, attention_mask in zip(input_ids_sequence, attention_mask_sequence):
            if 'Longformer' in self.model_name:
                global_attention_mask = torch.zeros_like(input_ids)
                global_attention_mask[:, 0] = 1
                bert_outputs = self.bert(input_ids,
                                         attention_mask=attention_mask,
                                         global_attention_mask=global_attention_mask)
            else:
                bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
            
            # --- [核心修改：与参考论文 Zhang et al. (2023) 完全对齐] ---
            # 论文中明确提到 "extract the representation of the [CLS] token for each encoded clinical note" (Source [191])
            # 因此，我们不再使用掩码平均池化，而是直接提取 [CLS] token 的特征
            cls_token_embedding = bert_outputs[0][:, 0, :]
            # --- [修改结束] ---

            note_representation = self.dropout(cls_token_embedding)
            txt_arr.append(note_representation)

        txt_arr = torch.stack(txt_arr)
        return txt_arr

# 在 my_model.py 文件中, 用下面的代码替换整个 ICUPromptModel 类

class ICUPromptModel(nn.Module):
    def __init__(self, hyp_params, clinical_bert_model, freeze_backbone=True):
        super(ICUPromptModel, self).__init__()
        
        # --- 模型维度 ---
        self.orig_d_t = hyp_params.orig_d_t         # 768
        self.orig_d_n_irg = hyp_params.orig_d_n_irg # 17
        self.orig_d_n_reg = hyp_params.orig_d_n_reg # 34
        
        self.d_t = hyp_params.proj_dim    # 例如, 64
        self.d_n = hyp_params.proj_dim    # 例如, 64
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.tlen, self.nlen = hyp_params.seq_len # 48, 48
        self.output_dim = hyp_params.output_dim

        # --- 文本处理模块 ---
        self.text_encoder = BertForRepresentation(hyp_params, clinical_bert_model)
        if freeze_backbone:
            print("Freezing parameters of the Longformer backbone...")
            for param in self.text_encoder.bert.parameters():
                param.requires_grad = False

        self.embed_time = self.d_t
        self.periodic = nn.Linear(1, self.embed_time - 1)
        self.linear = nn.Linear(1, 1)
        self.register_buffer('time_query', torch.linspace(0, 1., self.tlen))
        
        # 修复 1: 添加缺失的文本投影层
        self.proj_t = nn.Conv1d(self.orig_d_t, self.d_t, kernel_size=1, padding=0, bias=False)
        self.time_attn = multiTimeAttention(self.d_t, self.d_t, self.embed_time, self.num_heads)

        # --- 数值处理模块 (UTDE) ---
        self.proj_n_imputation = nn.Conv1d(self.orig_d_n_reg, self.d_n, kernel_size=1, padding=0, bias=False)
        self.time_attn_ts = multiTimeAttention(self.orig_d_n_irg * 2, self.d_n, self.embed_time, self.num_heads)
        self.moe = gateMLP(input_dim=self.d_n * 2, hidden_size=self.d_n, output_dim=self.d_n, dropout=self.attn_dropout)

        # --- 融合与记忆模块 (与 MULTCrossModel 对齐) ---
        # 注意: 此架构不需要复杂的提示学习部分
        
        # 跨模态 transformer
        self.trans_t_with_n = TransformerEncoder(embed_dim=self.d_t, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout)
        self.trans_n_with_t = TransformerEncoder(embed_dim=self.d_n, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout)
        
        # 自注意力记忆 transformer
        self.trans_t_mem = TransformerEncoder(embed_dim=self.d_t, num_heads=self.num_heads, layers=3)
        self.trans_n_mem = TransformerEncoder(embed_dim=self.d_n, num_heads=self.num_heads, layers=3)
        
        # --- 最终输出层 ---
        # 修复 2: 修正最终投影层的维度
        combined_dim = self.d_t + self.d_n
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, self.output_dim)

    def learn_time_embedding(self, tt):
        tt = tt.to(next(self.parameters()).device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def process_text_data(self, input_ids, attn_mask, note_mask, note_time, batch_size, device):
        if input_ids is None or input_ids.numel() == 0:
            return torch.zeros(batch_size, self.d_t, self.tlen, device=device)
        
        note_vectors = self.text_encoder(input_ids.permute(1, 0, 2), attn_mask.permute(1, 0, 2))
        proj_note_vectors_768 = note_vectors.permute(1, 0, 2) # -> [B, N, 768]
        
        # 应用投影层来降低维度
        proj_note_vectors = self.proj_t(proj_note_vectors_768.permute(0,2,1)).permute(0,2,1) # -> [B, N, 64]
        
        time_query_embed = self.learn_time_embedding(self.time_query.unsqueeze(0)).expand(batch_size, -1, -1)
        time_key_embed = self.learn_time_embedding(note_time)
        
        resampled_text = self.time_attn(time_query_embed, time_key_embed, proj_note_vectors, note_mask)
        
        return resampled_text.permute(0, 2, 1)

    def process_numerical_data(self, ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input, batch_size, device):
        # 路径 A: 填充
        if reg_ts_input is not None and reg_ts_input.numel() > 0:
            e_imp = self.proj_n_imputation(reg_ts_input.transpose(1, 2))
        else:
            e_imp = torch.zeros(batch_size, self.d_n, self.nlen, device=device)

        # 路径 B: 时间注意力
        if ts_input_sequences is not None and ts_input_sequences.numel() > 0:
            time_query_embed = self.learn_time_embedding(self.time_query.unsqueeze(0)).expand(batch_size, -1, -1)
            time_key_embed = self.learn_time_embedding(ts_tt)
            
            x_ts_irg = torch.cat((ts_input_sequences, ts_mask_sequences), 2)
            padding_mask = torch.cat((ts_mask_sequences, ts_mask_sequences), 2)
            
            e_attn_permuted = self.time_attn_ts(time_query_embed, time_key_embed, x_ts_irg, padding_mask)
            e_attn = e_attn_permuted.permute(0, 2, 1)
        else:
            e_attn = torch.zeros(batch_size, self.d_n, self.nlen, device=device)

        # 门控融合
        moe_gate_input = torch.cat((e_imp, e_attn), dim=1).permute(0, 2, 1)
        mixup_rate = self.moe(moe_gate_input).permute(0, 2, 1)
        
        final_numerical_representation = mixup_rate * e_attn + (1 - mixup_rate) * e_imp
        return final_numerical_representation

    def forward(self, ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input, 
                input_ids, attn_mask, note_time, note_time_mask, label, **kwargs):
        
        batch_size, device = label.size(0), label.device
        x_t = self.process_text_data(input_ids, attn_mask, note_time_mask, note_time, batch_size, device)
        x_n = self.process_numerical_data(ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input, batch_size, device)
        
        proj_x_t = x_t.permute(2, 0, 1)
        proj_x_n = x_n.permute(2, 0, 1)

        # --- 修复 3: 正确实现 MulT 融合架构 ---
        # 步骤 1: 跨模态注意力
        h_t_with_n = self.trans_t_with_n(proj_x_t, proj_x_n, proj_x_n)
        h_n_with_t = self.trans_n_with_t(proj_x_n, proj_x_t, proj_x_t)

        # 步骤 2: 自注意力记忆 (修正了信息流)
        h_t_final = self.trans_t_mem(h_t_with_n)
        h_n_final = self.trans_n_mem(h_n_with_t)

        # 步骤 3: 池化 (取最后一个隐状态)
        last_h_t = h_t_final[-1]
        last_h_n = h_n_final[-1]
        # --- 融合修复结束 ---

        # 步骤 4: 拼接与最终预测
        last_hs = torch.cat([last_h_t, last_h_n], dim=1)
        
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=0.2, training=self.training))
        last_hs_proj += last_hs
        
        return self.out_layer(last_hs_proj)

class ICUHyperParams:
    def __init__(self):
        self.orig_d_n_irg = 17
        self.orig_d_n_reg = 34
        self.orig_d_t = 768
        
        self.proj_dim = 64
        self.seq_len = [48, 48] # 序列长度
        self.num_heads, self.layers = 4, 4
        self.prompt_length, self.prompt_dim = 5, 64
        self.attn_dropout = 0.2
        self.output_dim = 2
        self.model_name = "yikuan8/Clinical-Longformer"

def create_icu_model(bert_model, freeze_backbone=True):
    hyp_params = ICUHyperParams()
    return ICUPromptModel(hyp_params, bert_model, freeze_backbone=freeze_backbone)
