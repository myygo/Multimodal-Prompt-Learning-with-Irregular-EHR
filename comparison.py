#!/usr/bin/env python3
"""
使用真实数据对 ICUPromptModel 和所有 baseline 模型进行综合性能测试。
支持完整的缺失模态情景和模态内部缺失情景的测试。
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from torch.utils.data import DataLoader, Dataset
import warnings
import random

# 动态导入
from data import data_perpare
from my_model import create_icu_model # 导入你的模型
from baseline_methods import create_adapted_baselines # 导入 baseline

# 尝试导入 transformers
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers 库加载成功")
except ImportError:
    TRANSFORMERS_AVAILABLE = False

warnings.filterwarnings('ignore')

# --- NEW: 实现模态内部缺失的函数 ---
def apply_intra_modal_missing(batch_data, missing_rate, random_seed=42):
    """
    对一个数据批次应用模态内部的随机缺失。
    这会随机遮蔽掉一部分时序数据点和文本词元。
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    (ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input,
     input_ids, attn_mask, note_time, note_time_mask, label) = batch_data
    
    # 1. 对文本数据应用缺失 (通过修改 attention mask)
    if attn_mask is not None:
        new_attn_mask = attn_mask.clone()
        for b in range(attn_mask.shape[0]):
            for n in range(attn_mask.shape[1]):
                valid_positions = (attn_mask[b, n] == 1).nonzero(as_tuple=False).squeeze(-1)
                if len(valid_positions) > 0:
                    num_to_mask = int(len(valid_positions) * missing_rate)
                    if num_to_mask > 0:
                        mask_indices = torch.randperm(len(valid_positions))[:num_to_mask]
                        positions_to_mask = valid_positions[mask_indices]
                        new_attn_mask[b, n, positions_to_mask] = 0
        attn_mask = new_attn_mask
    
    # 2. 对规则时序数据应用缺失 (直接将值设为0)
    if reg_ts_input is not None:
        new_reg_ts = reg_ts_input.clone()
        batch_size, seq_len, num_features = new_reg_ts.shape
        
        num_steps_to_mask = int(seq_len * missing_rate)
        if num_steps_to_mask > 0:
            for b in range(batch_size):
                mask_indices = torch.randperm(seq_len)[:num_steps_to_mask]
                new_reg_ts[b, mask_indices, :] = 0
        reg_ts_input = new_reg_ts

    return (ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input,
            input_ids, attn_mask, note_time, note_time_mask, label)


# --- MODIFIED: 更新测试情景 ---
def create_test_scenarios():
    """创建与图片匹配的测试情景"""
    return {
        'Complete_Data': {'text_missing': False, 'numerical_missing': False, 'intra_missing': False, 'missing_rate': 0.0, 'description': '完整数据'},
        'Missing_Text': {'text_missing': True, 'numerical_missing': False, 'intra_missing': False, 'missing_rate': 0.0, 'description': '缺失文本 (仅时序)'},
        'Missing_Numerical': {'text_missing': False, 'numerical_missing': True, 'intra_missing': False, 'missing_rate': 0.0, 'description': '缺失时序 (仅文本)'},
        'Intra_Missing_20': {'text_missing': False, 'numerical_missing': False, 'intra_missing': True, 'missing_rate': 0.2, 'description': '内部缺失20%'},
        'Intra_Missing_40': {'text_missing': False, 'numerical_missing': False, 'intra_missing': True, 'missing_rate': 0.4, 'description': '内部缺失40%'},
        'Intra_Missing_60': {'text_missing': False, 'numerical_missing': False, 'intra_missing': True, 'missing_rate': 0.6, 'description': '内部缺失60%'},
        'Intra_Missing_80': {'text_missing': False, 'numerical_missing': False, 'intra_missing': True, 'missing_rate': 0.8, 'description': '内部缺失80%'},
    }

# --- 辅助函数：加载模型 ---
def load_my_trained_model(weight_path, bert_model): # <--- 修改：接收 bert_model
    print(f"正在加载你的模型权重: {weight_path}")
    if not os.path.exists(weight_path):
        print(f"❌ 警告: 找不到权重文件 '{weight_path}'。")
        return None
    try:
        model = create_icu_model(bert_model) # <--- 修改：将 bert_model 传进去
        checkpoint = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ 你的模型加载成功")
        return model
    except Exception as e:
        print(f"❌ 加载你的模型权重时出错: {e}。")
        return None

# --- FIXED: 将 Args 类定义移到函数外部 ---
class Args:
    def __init__(self, file_path, batch_size, debug, task):
        self.file_path = file_path
        self.eval_batch_size = batch_size
        self.debug = debug
        self.task = task
        self.max_length = 128
        self.num_of_notes = 3
        self.tt_max = 24
        self.pad_to_max_length = True
        self.notes_order = "Last"
        self.modeltype = "TS_Text"
        self.model_name = "yikuan8/Clinical-Longformer"
        self.chunk = False
        self.ratio_notes_order = None

# --- 主测试流程 ---
def run_comprehensive_comparison(file_path, task='ihm', mode='test', batch_size=8, debug=False):
    """
    运行综合模型对比测试
    """
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # --- FIXED: 正确地实例化 Args 类 ---
    args = Args(file_path, batch_size, debug, task)
    
# --- 修改后 ---
    # 准备数据加载器和基础BERT模型
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers 库未安装，无法进行测试。")
        return
        
    from transformers import AutoModel # 导入 AutoModel

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("🧠 正在加载基础BERT模型...")
    bert_model = AutoModel.from_pretrained(args.model_name) # <--- 新增：加载基础BERT模型
    print("✅ 基础BERT模型加载成功")
    
    _, _, dataloader = data_perpare(args, mode, tokenizer)

    if dataloader is None: return

    # --- 加载你的模型 ---
    my_model_weight_path = "icu_model_weights_final/icu_model_epoch_13_best.pth" # !!! 确保路径正确
    my_model = load_my_trained_model(my_model_weight_path,bert_model)
    
    models_to_test = {}
    if my_model:
        models_to_test['ICU_PromptModel'] = my_model
    else:
        print("你的模型未能加载，测试无法进行。")
        return

    scenarios = create_test_scenarios()
    results = []

    for model_name, model in models_to_test.items():
        print(f"\n{'='*20} 正在测试: {model_name} {'='*20}")
        model.to(device).eval()
        
        for scenario_name, config in scenarios.items():
            print(f"  🔬 情景: {scenario_name} - {config['description']}")
            
            all_labels = []
            all_preds = []
            all_probs = [] # 用于 AUPRC

            for i, batch_data in enumerate(dataloader):
                if batch_data is None: continue

                # 应用模态缺失或内部缺失
                if config['intra_missing']:
                    batch_data = apply_intra_modal_missing(batch_data, config['missing_rate'], random_seed=42+i)
                
                (ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input,
                 input_ids, attn_mask, note_time, note_time_mask, label) = batch_data

                if config['text_missing']:
                    input_ids, attn_mask, note_time, note_time_mask = [None] * 4
                if config['numerical_missing']:
                    ts_input_sequences, ts_mask_sequences, reg_ts_input = [None] * 3

                # 将数据移动到设备
                batch_on_device = [
                    d.to(device).float() if torch.is_tensor(d) and d.dtype == torch.float32 else
                    d.to(device).long() if torch.is_tensor(d) else None
                    for d in (ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input,
                              input_ids, attn_mask, note_time, note_time_mask, label)
                ]
                
                # 前向传播
                with torch.no_grad():
                    outputs = model(*batch_on_device, intra_missing_ratio=0.0)

                # 收集结果
                if outputs is not None:
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_labels.extend(label.cpu().numpy())
                    all_preds.extend(preds)
                    all_probs.extend(probs)

            # --- MODIFIED: 计算新指标 ---
            if all_labels:
                f1 = f1_score(all_labels, all_preds, average='binary' if task == 'ihm' else 'macro')
                micro_f1 = f1_score(all_labels, all_preds, average='micro')
                auprc = average_precision_score(all_labels, all_probs)
                
                results.append({
                    'Model': model_name,
                    'Scenario': scenario_name,
                    'Missing_Rate': f"{config['missing_rate']*100:.0f}%" if config['intra_missing'] else "N/A",
                    'F1-Score': f1,
                    'Micro-F1': micro_f1,
                    'AUPRC': auprc,
                    'Samples': len(all_labels)
                })

    # --- 结果展示 ---
    if results:
        df = pd.DataFrame(results)
        print(f"\n{'📊'*10} 详细测试结果 {'📊'*10}")
        # 格式化输出
        df_display = df.copy()
        for col in ['F1-Score', 'Micro-F1', 'AUPRC']:
            df_display[col] = df_display[col].map('{:.4f}'.format)
        print(df_display.to_string(index=False))

if __name__ == "__main__":
    DATA_PATH = "data/ihm"
    run_comprehensive_comparison(DATA_PATH, task='ihm', batch_size=16)