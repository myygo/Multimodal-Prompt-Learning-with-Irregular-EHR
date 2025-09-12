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

class ICUPromptModel(nn.Module):
    def __init__(self, hyp_params, clinical_bert_model, freeze_backbone=True):
        super(ICUPromptModel, self).__init__()
        
        self.orig_d_t = hyp_params.orig_d_t
        self.orig_d_n_irg = hyp_params.orig_d_n_irg
        self.orig_d_n_reg = hyp_params.orig_d_n_reg
        
        self.d_t = hyp_params.proj_dim    
        self.d_n = hyp_params.proj_dim    
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.prompt_length = hyp_params.prompt_length
        self.prompt_dim = hyp_params.prompt_dim
        self.tlen, self.nlen = hyp_params.seq_len
        self.output_dim = hyp_params.output_dim

        # 文本处理模块
        self.text_encoder = BertForRepresentation(hyp_params, clinical_bert_model)
        if freeze_backbone:
            print("Freezing parameters of the Longformer backbone...")
            for param in self.text_encoder.bert.parameters():
                param.requires_grad = False

        self.embed_time = self.d_t
        self.periodic = nn.Linear(1, self.embed_time - 1)
        self.linear = nn.Linear(1, 1)
        self.register_buffer('time_query', torch.linspace(0, 1., self.tlen))
        self.time_attn = multiTimeAttention(self.orig_d_t, self.d_t, self.embed_time, self.num_heads)

        # 数值处理模块 (UTDE 实现)
        self.proj_n_imputation = nn.Conv1d(self.orig_d_n_reg, self.d_n, kernel_size=1, padding=0, bias=False)
        self.proj_n_attn = nn.Linear(self.orig_d_n_irg, self.d_n)
        #self.numerical_resample_attn = nn.MultiheadAttention(embed_dim=self.d_n, num_heads=self.num_heads, dropout=self.attn_dropout, batch_first=True)
        self.time_attn_ts = multiTimeAttention(self.orig_d_n_irg * 2, self.d_n, self.embed_time, self.num_heads)
        # 门控模块 (moe: Mixture of Experts)
        self.moe = gateMLP(input_dim=self.d_n * 2, hidden_size=self.d_n, output_dim=1, dropout=self.attn_dropout)

        # 模态生成与 Prompt 模块
        self.t2n = nn.Conv1d(self.d_t, self.prompt_dim, kernel_size=1, padding=0)
        self.n2t = nn.Conv1d(self.d_n, self.prompt_dim, kernel_size=1, padding=0)
        self.t_resample_query = nn.Parameter(torch.randn(1, self.tlen, self.prompt_dim))
        self.t_resample_attn = nn.MultiheadAttention(embed_dim=self.prompt_dim, num_heads=self.num_heads, dropout=self.attn_dropout, batch_first=True)
        self.n_resample_query = nn.Parameter(torch.randn(1, self.nlen, self.prompt_dim))
        self.n_resample_attn = nn.MultiheadAttention(embed_dim=self.prompt_dim, num_heads=self.num_heads, dropout=self.attn_dropout, batch_first=True)

        self.generative_prompt = nn.Parameter(torch.zeros(2, self.prompt_dim, self.prompt_length))
        self.promptt_m = nn.Parameter(torch.zeros(self.d_t, self.tlen))
        self.promptn_m = nn.Parameter(torch.zeros(self.d_n, self.nlen))
        self.promptt_nm = nn.Parameter(torch.zeros(self.d_t, self.tlen))
        self.promptn_nm = nn.Parameter(torch.zeros(self.d_n, self.nlen))
        self.missing_type_prompt = nn.Parameter(torch.zeros(2, self.prompt_length, self.prompt_dim))
        self.m_t = nn.Parameter(torch.zeros(self.tlen, 2 * self.prompt_dim))
        self.m_n = nn.Parameter(torch.zeros(self.nlen, 2 * self.prompt_dim))
        
        combined_dim = self.d_t + self.d_n # combined_dim = 128
        self.trans_t_with_n = TransformerEncoder(embed_dim=self.d_t, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout)
        self.trans_n_with_t = TransformerEncoder(embed_dim=self.d_n, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout)
        self.trans_t_mem = TransformerEncoder(embed_dim=self.d_t, num_heads=self.num_heads, layers=3)
        self.trans_n_mem = TransformerEncoder(embed_dim=self.d_n, num_heads=self.num_heads, layers=3)
        
        #final_combined_dim = combined_dim * 2 # final_combined_dim = 256
        
        # 让投影层保持维度不变 (256 -> 256)
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, self.output_dim)

    def learn_time_embedding(self, tt):
        tt = tt.to(next(self.parameters()).device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def determine_missing_modes(self, input_ids, ts_input, reg_ts_input, batch_size, device):
        missing_modes = []
        for i in range(batch_size):
            has_text = input_ids is not None and (input_ids[i] > 1).any()
            has_num = (ts_input is not None and torch.abs(ts_input[i]).sum() > 1e-6) or \
                      (reg_ts_input is not None and torch.abs(reg_ts_input[i]).sum() > 1e-6)
            
            if has_text and has_num: missing_modes.append(2) # 完整
            elif has_text: missing_modes.append(1) # 缺数值
            elif has_num: missing_modes.append(0) # 缺文本
            else: missing_modes.append(2) # 默认都缺失时为完整（应在数据加载时过滤）
        return torch.tensor(missing_modes, dtype=torch.long, device=device)

    def get_proj_matrix(self):
        tm_n = (self.promptt_m @ self.m_t + self.promptn_nm @ self.m_n).unsqueeze(0)
        t_nm = (self.promptt_nm @ self.m_t + self.promptn_m @ self.m_n).unsqueeze(0)
        t_n = (self.promptt_nm @ self.m_t + self.promptn_nm @ self.m_n).unsqueeze(0)
        self.mp = torch.cat([tm_n, t_nm, t_n], dim=0)

    def process_text_data(self, input_ids, attn_mask, note_mask, note_time, batch_size, device):
        if input_ids is None or input_ids.numel() == 0:
            return torch.zeros(batch_size, self.d_t, self.tlen, device=device)
        
        note_vectors = self.text_encoder(input_ids.permute(1, 0, 2), attn_mask.permute(1, 0, 2))
        proj_note_vectors_768 = note_vectors.permute(1, 0, 2) # -> [B, N, 768]
        #proj_note_vectors = self.proj_t(proj_note_vectors_768.permute(0,2,1)).permute(0,2,1) # -> [B, N, 64]
        
        time_query_embed = self.learn_time_embedding(self.time_query.unsqueeze(0)).expand(batch_size, -1, -1)
        time_key_embed = self.learn_time_embedding(note_time)
        resampled_text = self.time_attn(time_query_embed, time_key_embed, proj_note_vectors_768, note_mask)
        
        return resampled_text.permute(0, 2, 1)

    def process_numerical_data(self, ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input, batch_size, device):
        # 路径 A: 简单填充路径
        if reg_ts_input is not None and reg_ts_input.numel() > 0:
            features_imp = reg_ts_input.transpose(1, 2)
            if features_imp.shape[2] != self.nlen:
                features_imp = F.interpolate(features_imp, size=self.nlen, mode='linear', align_corners=False)
            e_imp = self.proj_n_imputation(features_imp)
        else:
            e_imp = torch.zeros(batch_size, self.d_n, self.nlen, device=device)

        # 路径 B: 时间注意力路径
        if ts_input_sequences is not None and ts_input_sequences.numel() > 0:
            time_query_embed = self.learn_time_embedding(self.time_query.unsqueeze(0)).expand(batch_size, -1, -1)
            time_key_embed = self.learn_time_embedding(ts_tt)

            x_ts_irg = torch.cat((ts_input_sequences, ts_mask_sequences), 2)
            padding_mask = torch.cat((ts_mask_sequences, ts_mask_sequences), 2)
            


           # value_ts = self.proj_n_attn(ts_input_sequences)
            padding_mask = (ts_mask_sequences.sum(dim=2) == 0)
            e_attn_permuted = self.time_attn_ts(time_query_embed, time_key_embed, x_ts_irg, padding_mask)
            #e_attn_permuted, _ = self.numerical_resample_attn(query=time_query_embed, key=time_key_embed, value=value_ts, key_padding_mask=padding_mask)
            e_attn = e_attn_permuted.permute(0, 2, 1)
        else:
            e_attn = torch.zeros(batch_size, self.d_n, self.nlen, device=device)

        # 门控融合
        moe_gate_input = torch.cat((e_imp, e_attn), dim=1).permute(0, 2, 1) # 准备输入
        mixup_rate = self.moe(moe_gate_input).permute(0, 2, 1) # 调用新模块

        final_numerical_representation = mixup_rate * e_attn + (1.0 - mixup_rate) * e_imp # 调整顺序以匹配
       # final_numerical_representation = g * e_imp + (1.0 - g) * e_attn

        return final_numerical_representation

    def get_complete_data(self, x_t, x_n, missing_modes):
        processed_t, processed_n = [], []
        for i in range(x_t.size(0)):
            mode = missing_modes[i].item()
            sample_t, sample_n = x_t[i].unsqueeze(0), x_n[i].unsqueeze(0)
            
            if mode == 0: # 文本缺失
                n2t_out = self.n2t(sample_n)
                gen_prompt = self.generative_prompt[0].unsqueeze(0)
                generated_sequence = torch.cat([gen_prompt, n2t_out], dim=2).permute(0, 2, 1)
                query = self.t_resample_query.expand(1, -1, -1)
                resampled_t, _ = self.t_resample_attn(query, generated_sequence, generated_sequence)
                final_t = resampled_t.permute(0, 2, 1) + self.promptt_m
                final_n = sample_n + self.promptn_nm
            elif mode == 1: # 数值缺失
                t2n_out = self.t2n(sample_t)
                gen_prompt = self.generative_prompt[1].unsqueeze(0)
                generated_sequence = torch.cat([gen_prompt, t2n_out], dim=2).permute(0, 2, 1)
                query = self.n_resample_query.expand(1, -1, -1)
                resampled_n, _ = self.n_resample_attn(query, generated_sequence, generated_sequence)
                final_n = resampled_n.permute(0, 2, 1) + self.promptn_m
                final_t = sample_t + self.promptt_nm
            else: # 模态完整
                final_t = sample_t + self.promptt_nm
                final_n = sample_n + self.promptn_nm
            
            processed_t.append(final_t)
            processed_n.append(final_n)
            
        return torch.cat(processed_t, dim=0), torch.cat(processed_n, dim=0)

    '''def forward(self, ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input, 
                input_ids, attn_mask, note_time, note_time_mask, label, intra_missing_ratio=0.0):
        
        batch_size, device = label.size(0), label.device
        
        x_t = self.process_text_data(input_ids, attn_mask, note_time_mask, note_time, batch_size, device)
        x_n = self.process_numerical_data(ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input, batch_size, device)
        
        missing_mod = self.determine_missing_modes(input_ids, ts_input_sequences, reg_ts_input, batch_size, device)
        xx_t, xx_n = self.get_complete_data(x_t, x_n, missing_mod)

        
        
        proj_x_t = xx_t.permute(2, 0, 1)
        proj_x_n = xx_n.permute(2, 0, 1)

        self.get_proj_matrix()
        
        batch_prompt = torch.cat([
            torch.matmul(self.missing_type_prompt, self.mp[missing_mod[i]]).unsqueeze(0)
            for i in range(batch_size)
        ], dim=0).transpose(0, 1)
        
        h_t_with_n = self.trans_t_with_n(proj_x_t, proj_x_n, proj_x_n)
        h_n_with_t = self.trans_n_with_t(proj_x_n, proj_x_t, proj_x_t)
        
        h_ts = torch.cat([h_t_with_n, h_n_with_t], dim=2)
        h_ns = torch.cat([h_n_with_t, h_t_with_n], dim=2)
        
        text_prompt = batch_prompt[0].transpose(0, 1)
        num_prompt = batch_prompt[1].transpose(0, 1)
        
        h_ts_with_prompt = torch.cat([h_ts, text_prompt], dim=0)
        h_ns_with_prompt = torch.cat([h_ns, num_prompt], dim=0)

        h_ts_final= self.trans_t_mem(h_ts_with_prompt)
        last_h_t = torch.mean(h_ts_final, dim=0)  # 池化操作发生在这里

        h_ns_final= self.trans_n_mem(h_ns_with_prompt)
        last_h_n = torch.mean(h_ns_final, dim=0)  # 池化操作发生在这里
        
        last_hs = torch.cat([last_h_t, last_h_n], dim=1)
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=0.2, training=self.training))
        last_hs_proj += last_hs
        
        return self.out_layer(last_hs_proj)'''
    def forward(self, ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input, 
                input_ids, attn_mask, note_time, note_time_mask, label, intra_missing_ratio=0.0):
        
        # 1. 数据预处理
        #    调用我们之前讨论过的、与参考代码对齐的预处理方法
        batch_size, device = label.size(0), label.device
        x_t = self.process_text_data(input_ids, attn_mask, note_time_mask, note_time, batch_size, device)
        x_n = self.process_numerical_data(ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input, batch_size, device)
        
        # 2. 准备Transformer输入
        #    将维度从 [Batch, Dim, Seq] 转换为 [Seq, Batch, Dim]
        proj_x_t = x_t.permute(2, 0, 1)
        proj_x_n = x_n.permute(2, 0, 1)

        # -------------------------------------------------------------------------
        # 核心修改：严格遵循参考代码 (MULTCrossModel) 的 MulT 融合架构
        # -------------------------------------------------------------------------

        # 3. 步骤一: 跨模态注意力融合 (Cross-Modal Attention)
        #    文本序列(t)作为Query，关注数值序列(n)
        h_t_with_n= self.trans_t_with_n(proj_x_t, proj_x_n, proj_x_n)
        #    数值序列(n)作为Query，关注文本序列(t)
        h_n_with_t = self.trans_n_with_t(proj_x_n, proj_x_t, proj_x_t)

        # 4. 步骤二: 自注意力记忆模块 (Self-Attention Memory)
        #    注意：这里输入的对应关系与参考代码一致，是交叉的
        #    文本记忆模块(trans_t_mem)处理的是被文本增强过的数值序列(h_n_with_t)
        h_t_final= self.trans_t_mem(h_n_with_t)
        #    数值记忆模块(trans_n_mem)处理的是被数值增强过的文本序列(h_t_with_n)
        h_n_final = self.trans_n_mem(h_t_with_n)

        # 5. 步骤三: 池化 (Pooling)
        #    使用参考代码的策略，取序列的最后一个时间步的输出
        last_h_t = h_t_final[-1]
        last_h_n = h_n_final[-1]

        # -------------------------------------------------------------------------
        # 融合与输出 (这部分逻辑与我们之前修复后的一致)
        # -------------------------------------------------------------------------

        # 6. 拼接最终的多模态特征向量
        last_hs = torch.cat([last_h_t, last_h_n], dim=1)
        
        # 7. 通过残差块和输出层进行最终预测
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