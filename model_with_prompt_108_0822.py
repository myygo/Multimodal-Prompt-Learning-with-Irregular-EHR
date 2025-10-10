#!/usr/bin/env python3
"""

修改：MLPLayer

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
            
            cls_token_embedding = bert_outputs[0][:, 0, :]
            note_representation = self.dropout(cls_token_embedding)
            txt_arr.append(note_representation)

        txt_arr = torch.stack(txt_arr)
        return txt_arr

class MiniTransformerEncoder(nn.Module):
    """使用 TransformerEncoder 进行模态转换。
    输入/输出形状: [B, D, L]
    """
    def __init__(self, in_dim, out_dim, num_heads, layers, attn_dropout):
        super().__init__()
        
        # 如果输入和输出维度不同，需要一个线性层或Conv1d进行维度适配
        if in_dim != out_dim:
             self.dim_proj = nn.Conv1d(in_dim, out_dim, kernel_size=1, padding=0)
        else:
             self.dim_proj = nn.Identity()

        # TransformerEncoder 期望的输入维度是 out_dim
        self.transformer = TransformerEncoder(
            embed_dim=out_dim, 
            num_heads=num_heads, 
            layers=layers, 
            attn_dropout=attn_dropout
        )

    def forward(self, x):
        # 1. 维度适配: [B, D_in, L] -> [B, D_out, L]
        x_proj = self.dim_proj(x)
        
        # 2. 形状调整: [B, D, L] -> [L, B, D] (Transformer 标准输入)
        x_trans = x_proj.permute(2, 0, 1) 
        
        # 3. 运行 Transformer
        h_trans = self.transformer(x_trans) # [L, B, D]
        
        # 4. 形状调整回原格式: [L, B, D] -> [B, D, L]
        return h_trans.permute(1, 2, 0)

class MLPLayer(nn.Module):
    """用于模态转换的深度MLP网络，通过1x1 Conv实现深度全连接。
    输入/输出形状: [B, D, L]
    """
    def __init__(self, in_dim, out_dim, hidden_dim=None, num_layers=3, dropout=0.1):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = max(in_dim, out_dim)
        
        layers = []
        
        # 第一层：输入维度 -> 隐藏维度
        layers.append(nn.Conv1d(in_dim, hidden_dim, kernel_size=1, padding=0))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # 中间层：隐藏维度 -> 隐藏维度
        for _ in range(num_layers - 2):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, padding=0))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            
        # 最后一层：隐藏维度 -> 输出维度
        layers.append(nn.Conv1d(hidden_dim, out_dim, kernel_size=1, padding=0))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ICUPromptModel(nn.Module):
    def __init__(self, hyp_params, clinical_bert_model, freeze_backbone=True,backbone_weights_path=None):
        super(ICUPromptModel, self).__init__()
        
        # --- 1. 维度与超参数定义 ---
        self.orig_d_t = hyp_params.orig_d_t
        self.orig_d_n_irg = hyp_params.orig_d_n_irg
        self.orig_d_n_reg = hyp_params.orig_d_n_reg
        self.d_t = hyp_params.proj_dim
        self.d_n = hyp_params.proj_dim
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.tlen, self.nlen = hyp_params.seq_len
        self.output_dim = hyp_params.output_dim
        self.prompt_length = hyp_params.prompt_length
        self.prompt_dim = self.d_t  # Prompt的维度与投影后维度一致

        # --- 2. 前端特征工程模块 (来自 MUL_model) ---
        # 文本处理
        self.text_encoder = BertForRepresentation(hyp_params, clinical_bert_model)
        if freeze_backbone:
            for param in self.text_encoder.bert.parameters():
                param.requires_grad = False
        self.embed_time = self.d_t
        self.periodic = nn.Linear(1, self.embed_time - 1)
        self.linear = nn.Linear(1, 1)
        self.register_buffer('time_query', torch.linspace(0, 1., self.tlen))
        self.proj_t = nn.Conv1d(self.orig_d_t, self.d_t, kernel_size=1, padding=0, bias=False)
        self.time_attn = multiTimeAttention(self.d_t, self.d_t, self.embed_time, self.num_heads)
        # 数值处理 (UTDE)
        self.proj_n_imputation = nn.Conv1d(self.orig_d_n_reg, self.d_n, kernel_size=1, padding=0, bias=False)
        self.time_attn_ts = multiTimeAttention(self.orig_d_n_irg * 2, self.d_n, self.embed_time, self.num_heads)
        self.moe = gateMLP(input_dim=self.d_n * 2, hidden_size=self.d_n, output_dim=self.d_n, dropout=self.attn_dropout)

        # --- 3. Prompting 相关模块 (来自 prompt_model) ---
        # 生成式 Prompts
        self.generative_prompt = nn.Parameter(torch.zeros(2, self.prompt_dim, self.prompt_length))
        # 模态转换层 (注意输入维度是d_n/d_t，因为在投影后使用)
        self.n2t = MiniTransformerEncoder(
            in_dim=self.d_n, 
            out_dim=self.prompt_dim, 
            num_heads=self.num_heads, 
            layers=2, # 建议只用2-3层，避免过大
            attn_dropout=self.attn_dropout
        )
        self.t2n = MiniTransformerEncoder(
            in_dim=self.d_t, 
            out_dim=self.prompt_dim, 
            num_heads=self.num_heads, 
            layers=2,
            attn_dropout=self.attn_dropout
        )
        # 序列长度融合层
        self.t_np = MLPLayer(self.prompt_length + self.nlen, self.tlen)
        self.n_tp = MLPLayer(self.prompt_length + self.tlen, self.nlen)
        # 模态信号 Prompts
        self.promptt_m = nn.Parameter(torch.zeros(self.prompt_dim, self.tlen))
        self.promptn_m = nn.Parameter(torch.zeros(self.prompt_dim, self.nlen))
        self.promptt_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.tlen))
        self.promptn_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.nlen))
        # 缺失类型 Prompts
        self.missing_type_prompt = nn.Parameter(torch.zeros(3, self.prompt_length, self.prompt_dim))
        self.m_t = nn.Parameter(torch.zeros(self.tlen,  self.prompt_dim))
        self.m_n = nn.Parameter(torch.zeros(self.nlen,  self.prompt_dim))
    
        # --- 4. 后端融合与记忆模块 (prompt_model架构) ---
        self.trans_t_with_n = self.get_network(self_type="tn")
        self.trans_n_with_t = self.get_network(self_type="nt")
        self.trans_t_mem = self.get_network(self_type="t_mem", layers=3)
        self.trans_n_mem = self.get_network(self_type="n_mem", layers=3)
        
        # --- 5. 最终输出层 (prompt_model架构) ---
        combined_dim = (self.d_t + self.d_n) # 维度翻倍
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, self.output_dim)

                # --- 新增：在所有层定义之后，加载预训练权重 ---
        if backbone_weights_path:
            print(f"--- 正在从 {backbone_weights_path} 加载预训练骨干权重... ---")
            backbone_state_dict = torch.load(backbone_weights_path)['model_state_dict']

            # 使用 strict=False，因为混合模型有额外的Prompt层，而骨干权重中没有
            # 这会加载所有匹配的层，并忽略不匹配的层（即您的Prompt相关层）
            self.load_state_dict(backbone_state_dict, strict=False)
            print("--- 骨干权重加载成功 ---")

    # --- 前端模块的辅助函数 (来自 MUL_model) ---
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
        proj_note_vectors_768 = note_vectors.permute(1, 0, 2)
        proj_note_vectors = self.proj_t(proj_note_vectors_768.permute(0,2,1)).permute(0,2,1)
        
        time_query_embed = self.learn_time_embedding(self.time_query.unsqueeze(0)).expand(batch_size, -1, -1)
        time_key_embed = self.learn_time_embedding(note_time)
        
        resampled_text = self.time_attn(time_query_embed, time_key_embed, proj_note_vectors, note_mask)
        
        return resampled_text.permute(0, 2, 1)

    def process_numerical_data(self, ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input, batch_size, device):
        if reg_ts_input is not None and reg_ts_input.numel() > 0:
            e_imp = self.proj_n_imputation(reg_ts_input.transpose(1, 2))
        else:
            e_imp = torch.zeros(batch_size, self.d_n, self.nlen, device=device)

        if ts_input_sequences is not None and ts_input_sequences.numel() > 0:
            time_query_embed = self.learn_time_embedding(self.time_query.unsqueeze(0)).expand(batch_size, -1, -1)
            time_key_embed = self.learn_time_embedding(ts_tt)
            
            x_ts_irg = torch.cat((ts_input_sequences, ts_mask_sequences), 2)
            padding_mask = torch.cat((ts_mask_sequences, ts_mask_sequences), 2)
            
            e_attn_permuted = self.time_attn_ts(time_query_embed, time_key_embed, x_ts_irg, padding_mask)
            e_attn = e_attn_permuted.permute(0, 2, 1)
        else:
            e_attn = torch.zeros(batch_size, self.d_n, self.nlen, device=device)

        moe_gate_input = torch.cat((e_imp, e_attn), dim=1).permute(0, 2, 1)
        mixup_rate = self.moe(moe_gate_input).permute(0, 2, 1)
        
        final_numerical_representation = mixup_rate * e_attn + (1 - mixup_rate) * e_imp
        return final_numerical_representation

    # --- Prompting 相关的辅助函数 (来自 prompt_model) ---
    def get_network(self, self_type="t", layers=-1):
        # 关键：t_mem 和 n_mem 的 embed_dim 是 2*d_t 和 2*d_n
        if self_type in ["t", "nt"]:
            embed_dim, attn_dropout = self.d_t, self.attn_dropout
        elif self_type in ["n", "tn"]:
            embed_dim, attn_dropout = self.d_n, self.attn_dropout
        # --- 核心修改：将这里的 2*d_t 和 2*d_n 改回 d_t 和 d_n ---
        elif self_type == "t_mem":
            embed_dim, attn_dropout = self.d_t, self.attn_dropout # 不再是 2 * self.d_t
        elif self_type == "n_mem":
            embed_dim, attn_dropout = self.d_n, self.attn_dropout # 不再是 2 * self.d_n
        # --- 修改结束 ---
        else:
            raise ValueError("Unknown network type")
        return TransformerEncoder(embed_dim=embed_dim, num_heads=self.num_heads, layers=max(self.layers, layers), attn_dropout=attn_dropout)

    def determine_missing_modes(self, input_ids, reg_ts_input):
        batch_size = input_ids.size(0) if input_ids is not None else reg_ts_input.size(0)
        missing_modes = []
        for i in range(batch_size):
            has_text = input_ids is not None and (input_ids[i] > 1).any()
            has_num = reg_ts_input is not None and torch.abs(reg_ts_input[i]).sum() > 1e-6
            if has_text and has_num: missing_modes.append(2)
            elif has_text: missing_modes.append(0)
            elif has_num: missing_modes.append(1)
            else: missing_modes.append(2)
        return torch.tensor(missing_modes, dtype=torch.long, device=next(self.parameters()).device)
    
    def get_proj_matrix(self):
        t_nm = (self.promptt_nm @ self.m_t + self.promptn_m @ self.m_n).unsqueeze(dim=0)
        tm_n = (self.promptt_m @ self.m_t + self.promptn_nm @ self.m_n).unsqueeze(dim=0)
        t_n = (self.promptt_nm @ self.m_t + self.promptn_nm @ self.m_n).unsqueeze(dim=0)
        self.mp = torch.cat([t_nm, tm_n, t_n], dim=0)
        
    def get_complete_data(self, x_t, x_n, missing_mode):
        """
        [已恢复重建功能]
        接收已经由 process_text_data 和 process_numerical_data 处理好的、
        低维度的 x_t 和 x_n。当模态缺失时，会主动进行重建。
        形状为 [1, proj_dim, seq_len]
        """
        
        if missing_mode == 0:  # 数值缺失，需要生成 x_n
            x_n_gen_part = self.t2n(x_t)

            x_n_cat = torch.cat(
                [self.generative_prompt[1, :, :].unsqueeze(0), x_n_gen_part],
                dim=2, # 沿序列维度拼接
            )
            
            x_n = self.n_tp(x_n_cat.transpose(1, 2)).transpose(1, 2) + self.promptn_m.unsqueeze(0)
            
            x_t = x_t + self.promptt_nm.unsqueeze(0)

        elif missing_mode == 1:  # 文本缺失，需要生成 x_t
            # 1. [已激活] 使用 n2t 将数值特征转换为文本域的特征
            # 输入 x_n: [1, d_n, nlen] -> 输出 x_t_gen_part: [1, prompt_dim, nlen]
            x_t_gen_part = self.n2t(x_n)

            # 2. 拼接生成式prompt
            # [1, prompt_dim, prompt_length] cat [1, prompt_dim, nlen] -> [1, prompt_dim, prompt_length + nlen]
            x_t_cat = torch.cat(
                [self.generative_prompt[0, :, :].unsqueeze(0), x_t_gen_part],
                dim=2,
            )

            # 3. 使用 MLPLayer (t_np) 和 transpose 技巧改变序列长度
            # (transpose) -> [1, prompt_length + nlen, prompt_dim]
            # (t_np) -> [1, tlen, prompt_dim]
            # (transpose) -> [1, prompt_dim, tlen]
            x_t = self.t_np(x_t_cat.transpose(1, 2)).transpose(1, 2) + self.promptt_m.unsqueeze(0)

            # 4. 对未缺失的 x_n 直接添加 "非缺失" prompt
            x_n = x_n + self.promptn_nm.unsqueeze(0)
        else:  # 模态完整
            # 两个模态都存在，直接添加 "非缺失" prompt
            x_t = x_t + self.promptt_nm.unsqueeze(0)
            x_n = x_n + self.promptn_nm.unsqueeze(0)

        return x_t, x_n

    # --- 最终的混合 Forward 函数 ---
    def forward(self, ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input, 
                input_ids, attn_mask, note_time, note_time_mask, label, **kwargs):
        
        batch_size, device = label.size(0), label.device

        # 步骤 1: 使用 MUL_model 的强大前端进行特征提取
        x_t = self.process_text_data(input_ids, attn_mask, note_time_mask, note_time, batch_size, device)
        x_n = self.process_numerical_data(ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input, batch_size, device)

        # 步骤 2: 确定缺失模式
        missing_modes = self.determine_missing_modes(input_ids, reg_ts_input)

        # 步骤 3: 逐样本应用数据补全逻辑
        xx_t_list, xx_n_list = [], []
        for idx in range(batch_size):
            x_t_temp, x_n_temp = self.get_complete_data(
                x_t[idx].unsqueeze(0), x_n[idx].unsqueeze(0), missing_modes[idx].item()
            )
            xx_t_list.append(x_t_temp)
            xx_n_list.append(x_n_temp)
        xx_t = torch.cat(xx_t_list, dim=0)
        xx_n = torch.cat(xx_n_list, dim=0)
        
        # 步骤 4: 准备Transformer输入并生成缺失类型Prompt
        proj_x_t = xx_t.permute(2, 0, 1)
        proj_x_n = xx_n.permute(2, 0, 1)
        
        self.get_proj_matrix()
        batch_prompts = [torch.matmul(self.missing_type_prompt, self.mp[int(min(max(mode.item(), 0), 2))]) for mode in missing_modes]
        batch_prompt = torch.stack(batch_prompts, dim=0).transpose(0, 1)

        # 步骤 5: 融合流程
        # 5.1 跨模态注意力
        h_t_with_n = self.trans_t_with_n(proj_x_t, proj_x_n, proj_x_n)
        h_n_with_t = self.trans_n_with_t(proj_x_n, proj_x_t, proj_x_t)
        
        # 5.2 [核心修改] 采用策略三：与原始特征拼接
  #      h_ts = torch.cat([proj_x_t, h_t_with_n], dim=2)
  #      h_ns = torch.cat([proj_x_n, h_n_with_t], dim=2)
        h_ts = h_t_with_n # 直接使用跨模态注意力的结果
        h_ns = h_n_with_t # 直接使用跨模态注意力的结果
        # 5.3 注入缺失类型Prompt
        batch_prompt_t = batch_prompt[0].transpose(0, 1)
        batch_prompt_n = batch_prompt[1].transpose(0, 1)
        
        h_ts = torch.cat([h_ts, batch_prompt_t], dim=0)
        h_ns = torch.cat([h_ns, batch_prompt_n], dim=0)
        
        # 5.4 自注意力记忆网络
        h_ts = self.trans_t_mem(h_ts)
        h_ns = self.trans_n_mem(h_ns)
            
        # 步骤 6: 池化与最终输出
        last_h_t = h_ts[-1]
        last_h_n = h_ns[-1]
        
        last_hs = torch.cat([last_h_t, last_h_n], dim=1)
        
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.attn_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output

class ICUHyperParams:
    def __init__(self):
        self.orig_d_n_irg = 17
        self.orig_d_n_reg = 34
        self.orig_d_t = 768
        self.proj_dim = 64
        self.seq_len = [48, 48]
        self.num_heads, self.layers = 4, 4
        self.prompt_length = 16 # Prompt 长度可以调整
        self.attn_dropout = 0.2
        self.output_dim = 2
        self.model_name = "yikuan8/Clinical-Longformer"

def create_icu_model(bert_model, freeze_backbone=True,backbone_weights_path=None):
    hyp_params = ICUHyperParams()
    return ICUPromptModel(hyp_params, bert_model, freeze_backbone=freeze_backbone,backbone_weights_path=backbone_weights_path)
