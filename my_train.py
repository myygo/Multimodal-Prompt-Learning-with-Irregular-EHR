#!/usr/bin/env python3
"""
[最终适配版] 用于训练 my_model.py 中的 ICUPromptModel。
实现了与原始 train.py 一致的精细化微调策略，并加入了梯度累积以解决显存问题。
"""
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from my_model import create_icu_model
from data import data_perpare

class ICUTrainer:
    """封装了训练、验证和检查点保存的逻辑，并集成了AMP"""
    def __init__(self, model, device, optimizer, scheduler, class_weights=None, use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = use_amp and torch.cuda.is_available()

        self.update_param_count()
        
        if class_weights is not None:
            class_weights = class_weights.to(device)
            print(f"正在使用带权重的损失函数，权重为: {class_weights.cpu().numpy()}")
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("  > 自动混合精度训练 (AMP) 已启用。")
            
    def update_param_count(self):
        params_to_train = [p for p in self.model.parameters() if p.requires_grad]
        print(f"  > 模型总参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  > 当前可训练参数量: {sum(p.numel() for p in params_to_train):,}")

    # --- 核心修改：添加 gradient_accumulation_steps 参数 ---
    def train_epoch(self, dataloader, gradient_accumulation_steps=1):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc="Training")
        
        # 将 zero_grad 移到循环外
        self.optimizer.zero_grad()
        
        for step, batch_data in enumerate(progress_bar):
            if batch_data is None: continue
            
            (ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input,
             input_ids, attn_mask, note_time, note_time_mask, label) = [
                d.to(self.device) if torch.is_tensor(d) else None for d in batch_data
            ]

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    ts_input_sequences=ts_input_sequences, ts_mask_sequences=ts_mask_sequences, ts_tt=ts_tt,
                    reg_ts_input=reg_ts_input, input_ids=input_ids, attn_mask=attn_mask,
                    note_time=note_time, note_time_mask=note_time_mask, label=label
                )
                if outputs is None: continue
                loss = self.criterion(outputs, label)
                
                # 对损失进行缩放
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 累加原始大小的损失
            total_loss += loss.item() * gradient_accumulation_steps
            
            # --- 核心修改：每 N 步更新一次 ---
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()

            progress_bar.set_postfix({'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}'}, refresh=True)

        avg_loss = total_loss / len(progress_bar) if len(progress_bar) > 0 else 0
        return avg_loss

    def validate(self, dataloader):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Validating"):
                if batch_data is None: continue
                
                (ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input,
                 input_ids, attn_mask, note_time, note_time_mask, label) = [
                    d.to(self.device) if torch.is_tensor(d) else None for d in batch_data
                ]

                outputs = self.model(
                    ts_input_sequences=ts_input_sequences, ts_mask_sequences=ts_mask_sequences, ts_tt=ts_tt,
                    reg_ts_input=reg_ts_input, input_ids=input_ids, attn_mask=attn_mask,
                    note_time=note_time, note_time_mask=note_time_mask, label=label
                )
                
                if outputs is None: continue
                probs = torch.softmax(outputs, dim=1)[:, 1]
                predicted = torch.argmax(outputs, dim=1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        if all_labels:
            f1 = f1_score(all_labels, all_preds, average='binary')
            auprc = average_precision_score(all_labels, all_probs)
            return f1, auprc
        return 0.0, 0.0

    def save_checkpoint(self, filepath, epoch, is_best=False):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict()}
        if is_best:
            best_path = filepath.replace(f'_epoch_{epoch}.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

def train_icu_model(data_path, epochs, batch_size, grad_accum_steps, learning_rate, backbone_lr, fine_tune_epochs, save_dir, bert_model_name, use_amp):
    print("开始训练 ICUPromptModel (使用精细化微调策略)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    torch.manual_seed(42); np.random.seed(42)
    
    class DataArgs:
        def __init__(self):
            self.file_path = data_path
            self.train_batch_size, self.eval_batch_size = batch_size, batch_size
            self.debug = False; self.max_length = 1024; self.num_of_notes = 5
            self.tt_max = 48; self.pad_to_max_length = True; self.notes_order = "Last"
            self.modeltype = "TS_Text"; self.model_name = bert_model_name
            self.chunk = False; self.ratio_notes_order = None
    
    data_args = DataArgs()
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    train_dataset, _, train_dataloader = data_perpare(data_args, 'train', tokenizer)
    _, _, val_dataloader = data_perpare(data_args, 'val', tokenizer)
    
    class_weights = torch.FloatTensor([0.57515432, 3.82648871])
    
    print("\n[步骤 2] 创建模型 (Longformer 初始为冻结状态)...")
    clinical_bert_model = AutoModel.from_pretrained(bert_model_name)
    # 初始创建时，总是先冻结
    model = create_icu_model(clinical_bert_model, freeze_backbone=True)
    
    trainer = None # trainer 将在循环内部初始化和更新
    
    best_val_metric_sum, best_f1, best_auprc, best_epoch = 0, 0, 0, 0
    patience, patience_counter = 8, 0
    
    print(f"\n[步骤 5] 开始训练，共 {epochs} 个 epoch...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        start_time = time.time()
        
        # --- 核心修改：在每个 epoch 开始时决定是否微调 ---
        is_fine_tuning_epoch = epoch < fine_tune_epochs

        if is_fine_tuning_epoch:
            print(f"** [策略] 本 Epoch ({epoch+1}) 将微调 Longformer **")
            # 解冻
            for param in model.text_encoder.bert.parameters():
                param.requires_grad = True
            
            # 创建带差分学习率的优化器
            optimizer_grouped_parameters = [
                {"params": model.text_encoder.bert.parameters(), "lr": backbone_lr},
                {"params": [p for n, p in model.named_parameters() if 'text_encoder.bert' not in n and p.requires_grad], "lr": learning_rate}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, weight_decay=1e-4)
        else:
            if epoch == fine_tune_epochs: # 只在第一次转换时打印
                print(f"** [策略] 从本 Epoch ({epoch+1}) 开始，冻结 Longformer **")
            # 冻结
            for param in model.text_encoder.bert.parameters():
                param.requires_grad = False
            
            # 创建只包含非骨干参数的优化器
            params_to_optimize = [p for p in model.parameters() if p.requires_grad]
            optimizer = AdamW(params_to_optimize, lr=learning_rate, weight_decay=1e-4)

        # 每次优化器变化后，都需要重新创建调度器和训练器
        if trainer is None or is_fine_tuning_epoch != (epoch-1 < fine_tune_epochs):
            print("  > 优化策略改变，重新初始化 Trainer...")
            num_training_steps = epochs * len(train_dataloader)
            num_warmup_steps = int(0.1 * num_training_steps) # 预热只在开始时有效
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
            trainer = ICUTrainer(model, device, optimizer, scheduler, class_weights=class_weights, use_amp=use_amp)
        else:
             # 如果策略不变，只需更新优化器和调度器
             trainer.optimizer = optimizer
             trainer.scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=(epochs-epoch)*len(train_dataloader))
        
        train_loss = trainer.train_epoch(train_dataloader, gradient_accumulation_steps=grad_accum_steps)
        val_f1, val_auprc = trainer.validate(val_dataloader)
        
        print(f"训练 - 平均损失: {train_loss:.4f}")
        print(f"验证 - F1分数: {val_f1:.4f}, AUPRC: {val_auprc:.4f}")
            
        current_metric_sum = val_f1 + val_auprc
        if current_metric_sum > best_val_metric_sum:
            best_val_metric_sum = current_metric_sum
            best_f1, best_auprc, best_epoch = val_f1, val_auprc, epoch + 1
            patience_counter = 0
            print(f"🎉 新的最佳指标: F1+AUPRC = {best_val_metric_sum:.4f}")
            is_best = True
        else:
            patience_counter += 1
            print(f"验证集指标无提升 ({patience_counter}/{patience})")
            is_best = False
            
        checkpoint_path = os.path.join(save_dir, f"icu_model_epoch_{epoch+1}.pth")
        trainer.save_checkpoint(checkpoint_path, epoch + 1, is_best=is_best)
            
        if patience_counter >= patience:
            print(f"早停：指标连续 {patience} 个epoch无提升。")
            break
        
        print(f"Epoch耗时: {time.time() - start_time:.2f}秒")

    print("\n" + "="*60)
    print("训练完成总结:")
    if best_epoch > 0:
        print(f"最佳验证集指标 (在 Epoch {best_epoch}):")
        print(f"  - F1 Score: {best_f1:.4f}")
        print(f"  - AUPRC: {best_auprc:.4f}")
        best_model_path = os.path.join(save_dir, 'icu_model_best.pth')
        print(f"最佳模型权重已保存至: {best_model_path}")
    else:
        print("未能在训练过程中找到更优的模型。")
    print("=" * 60)

if __name__ == "__main__":
    DATA_PATH = "Data/ihm"
    BERT_MODEL_NAME = "yikuan8/Clinical-Longformer"
    SAVE_DIR = "my_model_weights_finetuned"
    
    EPOCHS = 25
    
    # --- 核心修改：引入梯度累积配置 ---
    EFFECTIVE_BATCH_SIZE = 32
    PHYSICAL_BATCH_SIZE = 2  # 设置一个非常小的物理批次
    GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // PHYSICAL_BATCH_SIZE
    
    print(f"将使用有效批次: {EFFECTIVE_BATCH_SIZE} (物理批次: {PHYSICAL_BATCH_SIZE}, 累积步数: {GRAD_ACCUM_STEPS})")
    
    LEARNING_RATE = 5e-5       
    BACKBONE_LR = 2e-5         
    FINE_TUNE_EPOCHS = 3       
    
    USE_AMP = True
    
    train_icu_model(
        data_path=DATA_PATH,
        epochs=EPOCHS,
        batch_size=PHYSICAL_BATCH_SIZE, # <--- 传递物理批次
        grad_accum_steps=GRAD_ACCUM_STEPS, # <--- 传递累积步数
        learning_rate=LEARNING_RATE,
        backbone_lr=BACKBONE_LR,
        fine_tune_epochs=FINE_TUNE_EPOCHS,
        save_dir=SAVE_DIR,
        bert_model_name=BERT_MODEL_NAME,
        use_amp=USE_AMP,
    )
