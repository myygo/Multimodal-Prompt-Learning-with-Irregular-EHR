#!/usr/bin/env python3
"""
使用真实数据对所有 baseline 模型进行综合性能测试。
该脚本是 comparison.py 的适配版本，专为 baseline_methods.py 中的模型设计。
支持完整的缺失模态情景和模态内部缺失情景的测试。
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score, average_precision_score
from torch.utils.data import DataLoader
import warnings
import random

# 动态导入
from data import data_perpare
from baseline_methods import create_adapted_baselines # 导入 baseline 模型创建函数

# 尝试导入 transformers
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers 库加载成功")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers 库未找到，将无法处理文本数据")

warnings.filterwarnings('ignore')

# --- 复用：实现模态内部缺失的函数 (与 comparison.py 相同) ---
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
    if attn_mask is not None and TRANSFORMERS_AVAILABLE:
        # 对于 baseline，直接修改 input_ids 也可以，但修改 attn_mask 更通用
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

# --- 复用：创建测试情景 (与 comparison.py 相同) ---
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

# --- NEW: 辅助函数：加载所有 Baseline 模型 ---
# --- NEW: 辅助函数：加载所有 Baseline 模型 (适配新的目录结构) ---
def load_baseline_models(weights_dir, num_classes=2):
    """
    创建所有 baseline 模型实例，并从各自的子目录中加载最新的最优权重。
    会自动寻找 '..._best.pth' 文件，并选择 epoch 编号最大的一个。
    """
    print(f"正在从主目录 '{weights_dir}' 加载所有 Baseline 模型权重...")
    baseline_models = create_adapted_baselines(num_classes)
    loaded_models = {}

    for name, model in baseline_models.items():
        model_subdir = os.path.join(weights_dir, name)
        
        if not os.path.isdir(model_subdir):
            print(f"⚠️ 警告: 找不到模型 '{name}' 的权重目录 '{model_subdir}'。将跳过此模型。")
            continue

        # 寻找目录中最新的 "_best.pth" 文件
        best_file = None
        highest_epoch = -1
        try:
            for filename in os.listdir(model_subdir):
                if "_best.pth" in filename:
                    # 从文件名中提取 epoch 数值，例如 'epoch_5_best.pth' -> 5
                    epoch_num = int(filename.split('_')[1])
                    if epoch_num > highest_epoch:
                        highest_epoch = epoch_num
                        best_file = filename
        except (ValueError, IndexError):
             print(f"⚠️ 警告: 目录 '{model_subdir}' 中的文件名格式不规范，无法自动确定最新权重。将跳过 '{name}'。")
             continue

        if best_file is None:
            print(f"⚠️ 警告: 在 '{model_subdir}' 中未找到任何 '*_best.pth' 权重文件。将跳过 '{name}'。")
            continue
            
        weight_path = os.path.join(model_subdir, best_file)
        print(f"  - 正在为 '{name}' 加载权重: {weight_path}")
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            loaded_models[name] = model
            print(f"✅ 模型 '{name}' 加载成功")
        except Exception as e:
            print(f"❌ 加载模型 '{name}' 权重时出错: {e}。将跳过此模型。")
            
    return loaded_models
# --- 复用：Args 类 ---
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
        self.model_name = "bert-base-uncased"
        self.chunk = False
        self.ratio_notes_order = None

# --- 主测试流程 ---
def run_baseline_comparison(file_path, task='ihm', mode='test', batch_size=16, debug=False):
    """
    运行所有 Baseline 模型的综合对比测试
    """
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    args = Args(file_path, batch_size, debug, task)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name) if TRANSFORMERS_AVAILABLE else None
    _, _, dataloader = data_perpare(args, mode, tokenizer)

    if dataloader is None: return

    # --- 加载所有训练好的 Baseline 模型 ---
    baseline_weights_path = "baseline_weights/" # !!! 确保此路径下存放了训练好的模型权重
    models_to_test = load_baseline_models(baseline_weights_path)
    
    if not models_to_test:
        print("❌ 未能加载任何 Baseline 模型，测试无法进行。请检查权重路径和文件。")
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
            all_probs = []

            for i, batch_data in enumerate(dataloader):
                if batch_data is None: continue

                # 应用模态缺失或内部缺失
                if config['intra_missing']:
                    batch_data = apply_intra_modal_missing(batch_data, config['missing_rate'], random_seed=42+i)
                
                (ts_input_sequences, _, _, reg_ts_input,
                 input_ids, _, note_time, _, label) = batch_data

                if config['text_missing']:
                    input_ids, note_time = None, None
                if config['numerical_missing']:
                    ts_input_sequences, reg_ts_input = None, None
                
                # --- 适配 Baseline 输入：使用关键字参数 ---
                model_inputs = {
                    'x_ts': ts_input_sequences,
                    'reg_ts': reg_ts_input,
                    'input_ids_sequences': input_ids,
                    'note_time': note_time,
                    'label': label
                }

                # 将数据移动到设备
                for key, value in model_inputs.items():
                    if torch.is_tensor(value):
                        model_inputs[key] = value.to(device)
                
                with torch.no_grad():
                    outputs = model(**model_inputs)

                if outputs is not None:
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_labels.extend(label.cpu().numpy())
                    all_preds.extend(preds)
                    all_probs.extend(probs)

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

    if results:
        df = pd.DataFrame(results)
        print(f"\n{'📊'*10} Baseline 模型详细测试结果 {'📊'*10}")
        df_display = df.copy()
        for col in ['F1-Score', 'Micro-F1', 'AUPRC']:
            df_display[col] = df_display[col].map('{:.4f}'.format)
        print(df_display.to_string(index=False))

if __name__ == "__main__":
    # 假设您的数据路径和任务与原文件相同
    DATA_PATH = "data/ihm" 
    run_baseline_comparison(DATA_PATH, task='ihm', batch_size=16)