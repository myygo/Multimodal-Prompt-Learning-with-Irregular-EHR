#!/usr/bin/env python3
"""
[最终完整版] 用于训练和评估所有 Baseline 模型的脚本。

功能亮点:
- 通过修改一个列表来选择要训练的基线模型。
- 使用与主模型训练脚本一致的 tqdm 进度条设置 (按25%进度更新)。
- 使用固定的类别权重进行训练。
- 在每个epoch结束后为每个基线模型保存检查点，并单独保存最佳模型权重。
- 正确加载和共享预训练的 Hugging Face 模型，并冻结其参数。
- 适配了基线模型期望的 kwargs 输入格式，并解决了所有调用错误。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModel, AutoTokenizer

# 确保所有需要的模块都已导入
# 假设这些文件与此脚本在同一目录下
from baseline_methods import create_adapted_baselines
from my_model import ICUHyperParams 
from data import data_perpare

class BaselineTrainer:
    """为 Baseline 模型定制的训练器，处理kwargs输入并包含完整功能"""
    def __init__(self, model, device, class_weights=None, learning_rate=0.0001):
        self.model = model.to(device)
        self.device = device
        # 只优化未被冻结的参数
        params_to_train = [p for p in self.model.parameters() if p.requires_grad]
        print(f"  > BaselineTrainer: 模型总参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  > BaselineTrainer: 可训练参数量: {sum(p.numel() for p in params_to_train):,}")
        self.optimizer = optim.Adam(params_to_train, lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=3, factor=0.5, verbose=True)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    def save_checkpoint(self, filepath, epoch, is_best=False):
        """保存模型检查点"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, filepath)
        # print(f"Checkpoint saved: {filepath}") # 注释掉以保持日志整洁
        
        if is_best:
            best_path = os.path.join(os.path.dirname(filepath), 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"🎉 Best model for this run saved to: {best_path}")

    def _prepare_kwargs(self, batch_data):
        """将dataloader的元组输出转换为模型期望的kwargs字典"""
        (ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input,
         input_ids, attn_mask, note_time, note_time_mask, label) = [d.to(self.device) if torch.is_tensor(d) else None for d in batch_data]
        return {
            "x_ts": ts_input_sequences, "reg_ts": reg_ts_input,
            "input_ids_sequences": input_ids, "attn_mask": attn_mask,
            "note_time": note_time, "note_mask": note_time_mask,
            "label": label
        }

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, all_preds, all_labels = 0, [], []
        
        update_interval = max(1, len(dataloader) // 4)
        progress_bar = tqdm(
            dataloader, 
            desc="Training", 
            miniters=update_interval,
            mininterval=float('inf')
        )
        
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None: continue
            kwargs = self._prepare_kwargs(batch_data)
            label = kwargs.pop("label")
            if label is None: continue

            try:
                self.optimizer.zero_grad()
                outputs = self.model(**kwargs)
                if outputs is None: continue
                
                loss = self.criterion(outputs, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'}, refresh=False)
            except Exception as e:
                print(f"\n训练基线模型批次 {batch_idx} 发生错误: {e}")
                continue
        
        avg_loss = total_loss / len(all_labels) if all_labels else 0
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0
        return avg_loss, accuracy

    def validate(self, dataloader):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        
        update_interval = max(1, len(dataloader) // 4)
        progress_bar = tqdm(
            dataloader, 
            desc="Validating", 
            miniters=update_interval,
            mininterval=float('inf')
        )
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(progress_bar):
                if batch_data is None: continue
                kwargs = self._prepare_kwargs(batch_data)
                label = kwargs.pop("label")
                if label is None: continue
                
                try:
                    outputs = self.model(**kwargs)
                    if outputs is None: continue
                    
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    predicted = torch.argmax(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                except Exception as e:
                    print(f"\n验证基线模型批次发生错误: {e}")
                    continue
        
        if all_labels:
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='binary')
            auprc = average_precision_score(all_labels, all_probs)
            return accuracy, f1, auprc
        return 0.0, 0.0, 0.0

class ArgsPlaceholder:
    """用于模拟args参数，方便管理和传递"""
    def __init__(self, data_path, batch_size, bert_model_name):
        self.file_path = data_path
        self.train_batch_size, self.eval_batch_size = batch_size, batch_size
        self.debug = False
        self.max_length, self.num_of_notes, self.tt_max = 512, 5, 48
        self.pad_to_max_length, self.notes_order = True, "Last"
        self.modeltype, self.model_name = "TS_Text", bert_model_name
        self.chunk, self.ratio_notes_order = False, None

def train_and_evaluate_baselines(data_path, bert_model_name, models_to_run, epochs, batch_size, learning_rate, save_dir):
    print("开始训练指定的 Baseline 模型")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n[步骤 1] 准备数据加载器...")
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    args = ArgsPlaceholder(data_path, batch_size, bert_model_name)
    _, _, train_dataloader = data_perpare(args, 'train', tokenizer)
    _, _, val_dataloader = data_perpare(args, 'val', tokenizer)
    
    fixed_weights = [0.57515432, 3.82648871]
    class_weights = torch.FloatTensor(fixed_weights)
    print(f"正在为所有模型使用固定的类别权重: {class_weights.numpy()}")
    
    print("\n[步骤 2] 创建所有基线模型...")
    print(f"正在从Hugging Face加载共享的预训练模型: {bert_model_name}")
    base_bert_model = AutoModel.from_pretrained(bert_model_name)
    all_baseline_models = create_adapted_baselines(base_bert_model, num_classes=2)
    
    final_results = []

    for model_name in models_to_run:
        if model_name not in all_baseline_models:
            print(f"警告：在 baseline_methods.py 中找不到名为 '{model_name}' 的模型，已跳过。")
            continue
        
        model = all_baseline_models[model_name]
        print(f"\n{'='*25} 正在训练: {model_name} {'='*25}")
        
        trainer = BaselineTrainer(model, device, class_weights=class_weights, learning_rate=learning_rate)
        best_val_metric, best_f1, best_auprc, best_epoch = 0, 0, 0, 0
        patience, patience_counter = 5, 0

        for epoch in range(epochs):
            print(f"--- Epoch {epoch+1}/{epochs} for {model_name} ---")
            
            trainer.train_epoch(train_dataloader)
            val_acc, val_f1, val_auprc = trainer.validate(val_dataloader)
            
            print(f"验证 - {model_name} - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUPRC: {val_auprc:.4f}")
            
            current_metric_sum = val_f1 + val_auprc
            is_best = current_metric_sum > best_val_metric
            
            if is_best:
                best_val_metric = current_metric_sum
                best_f1, best_auprc, best_epoch = val_f1, val_auprc, epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            model_save_dir = os.path.join(save_dir, model_name)
            checkpoint_path = os.path.join(model_save_dir, f"epoch_{epoch+1}.pth")
            trainer.save_checkpoint(checkpoint_path, epoch + 1, is_best=is_best)
            
            if patience_counter >= patience:
                print(f"早停：指标连续 {patience} 个epoch无提升。")
                break

        final_results.append({"Model": model_name, "Best F1": best_f1, "Best AUPRC": best_auprc, "Best Epoch": best_epoch})
        del model, trainer
        torch.cuda.empty_cache()

    print(f"\n{'📊'*10} 指定基线模型训练完成总结 {'📊'*10}")
    if final_results:
        df = pd.DataFrame(final_results)
        df_display = df.copy()
        for col in ['Best F1', 'Best AUPRC']:
            df_display[col] = df_display[col].map('{:.4f}'.format)
        print(df_display.to_string(index=False))
    else:
        print("没有训练任何模型。")

if __name__ == "__main__":
    # --- 主要参数配置区 ---
    DATA_PATH = "data/ihm"
    BERT_MODEL_NAME = "yikuan8/Clinical-Longformer"
    SAVE_DIR = "baseline_weights"
    EPOCHS = 15
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4 # 适用于冻结模式下训练小型分类器

    # --- 模型选择配置区 ---
    # 在这里列出您想要训练的所有基线模型的名称
    # 可用名称: 'LB_timeseries_only', 'LB_text_only', 'LB_both_modalities', 'MS', 'MD'
    MODELS_TO_RUN = [
        'LB_text_only',
        'LB_both_modalities',
        'MS',
        'MD',
    ]
    
    # --- 运行区 ---
    train_file = os.path.join(DATA_PATH, 'trainp2x_data.pkl')
    if not os.path.exists(train_file):
        print(f"错误：找不到训练数据文件: '{train_file}'")
    else:
        train_and_evaluate_baselines(
            data_path=DATA_PATH,
            bert_model_name=BERT_MODEL_NAME,
            models_to_run=MODELS_TO_RUN,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            save_dir=SAVE_DIR
        )