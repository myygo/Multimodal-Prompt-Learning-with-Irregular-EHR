#!/usr/bin/env python3
"""
[最终完整版] 用于训练 my_model.py 中的 ICUPromptModel。

功能亮点:
- 实现了精细化微调策略 (前N个epoch微调，后续冻结)。
- 实现了梯度累积，以在有限显存下模拟大批量训练。
- 实现了子集训练模式，用于快速调试和验证。
- **新增了模态丢弃 (Modality Dropout) 功能，以提升模型鲁棒性。**
"""
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from torch.utils.data import DataLoader, Subset, RandomSampler, SequentialSampler
import os
import time
from tqdm import tqdm
import warnings
import random

warnings.filterwarnings('ignore')

# 导入您项目中的必要模块
from model_prompt import create_icu_model
from data import data_perpare, TextTSIrgcollate_fn

class ICUTrainer:
    """封装了训练、验证和检查点保存的逻辑"""
    def __init__(self, model, device, optimizer, scheduler, class_weights=None, use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = use_amp and torch.cuda.is_available()

        self.update_param_count()
        
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def update_param_count(self):
        params_to_train = [p for p in self.model.parameters() if p.requires_grad]
        print(f"  > 模型总参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  > 当前可训练参数量: {sum(p.numel() for p in params_to_train):,}")

    # --- 核心修改 1: 为 train_epoch 添加 modality_dropout_rate 参数 ---
    def train_epoch(self, dataloader, gradient_accumulation_steps=1, modality_dropout_rate=0.0):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc="Training")
        self.optimizer.zero_grad()
        
        for step, batch_data in enumerate(progress_bar):
            if batch_data is None: continue
            
            (ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input,
             input_ids, attn_mask, note_time, note_time_mask, label) = batch_data

            # --- 核心修改 2: 在这里实现模态丢弃逻辑 ---
            if self.model.training and modality_dropout_rate > 0:
                if torch.rand(1).item() < modality_dropout_rate:
                    # 决定丢弃哪个模态 (50/50 概率)
                    if torch.rand(1).item() < 0.5:
                        # 丢弃文本模态
                        input_ids, attn_mask, note_time, note_time_mask = [None] * 4
                    else:
                        # 丢弃数值模态
                        ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input = [None] * 4
            
            # 将数据移动到设备
            (ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input,
             input_ids, attn_mask, note_time, note_time_mask, label) = [
                d.to(self.device) if torch.is_tensor(d) else None for d in 
                (ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input,
                 input_ids, attn_mask, note_time, note_time_mask, label)
            ]

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    ts_input_sequences=ts_input_sequences, ts_mask_sequences=ts_mask_sequences, ts_tt=ts_tt,
                    reg_ts_input=reg_ts_input, input_ids=input_ids, attn_mask=attn_mask,
                    note_time=note_time, note_time_mask=note_time_mask, label=label
                )
                if outputs is None: continue
                loss = self.criterion(outputs, label)
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            progress_bar.set_postfix({'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}'}, refresh=True)

        return total_loss / len(progress_bar) if len(progress_bar) > 0 else 0

    def validate(self, dataloader):
        # ... (validate 函数保持不变) ...
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
        # ... (save_checkpoint 函数保持不变) ...
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict()}
        if is_best:
            # --- 在这里修改为您想要的最佳模型文件名 ---
            # 我们直接使用 os.path.join 来构建一个全新的路径，更安全
            dir_name = os.path.dirname(filepath)
            best_path = os.path.join(dir_name, "best_model_50.pth")
            # ------------------------------------------
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

def train_icu_model(args, train_dataloader, val_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n[步骤 2] 创建模型 (Longformer 初始为冻结状态)...")
    clinical_bert_model = AutoModel.from_pretrained(args.bert_model_name)
    model = create_icu_model(clinical_bert_model, freeze_backbone=(args.fine_tune_epochs == 0))
    
    trainer = None
    best_val_metric_sum, best_f1, best_auprc, best_epoch = 0, 0, 0, 0
    patience, patience_counter = 8, 0
    
    print(f"\n[步骤 5] 开始训练，共 {args.epochs} 个 epoch...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        start_time = time.time()
        
        is_fine_tuning_epoch = epoch < args.fine_tune_epochs

        if is_fine_tuning_epoch:
            if epoch == 0: print(f"** [策略] 前 {args.fine_tune_epochs} 个 Epoch 将微调 Longformer **")
            for param in model.text_encoder.bert.parameters():
                param.requires_grad = True
            optimizer_grouped_parameters = [
                {"params": model.text_encoder.bert.parameters(), "lr": args.backbone_lr},
                {"params": [p for n, p in model.named_parameters() if 'text_encoder.bert' not in n and p.requires_grad], "lr": args.learning_rate}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, weight_decay=1e-4)
        else:
            if epoch == args.fine_tune_epochs:
                print(f"** [策略] 从本 Epoch ({epoch+1}) 开始，冻结 Longformer **")
            for param in model.text_encoder.bert.parameters():
                param.requires_grad = False
            params_to_optimize = [p for p in model.parameters() if p.requires_grad]
            optimizer = AdamW(params_to_optimize, lr=args.learning_rate, weight_decay=1e-4)

        if trainer is None or is_fine_tuning_epoch != (epoch-1 < args.fine_tune_epochs):
            print("  > 优化策略改变，重新初始化 Trainer...")
            num_training_steps = args.epochs * len(train_dataloader)
            num_warmup_steps = int(0.1 * num_training_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
            trainer = ICUTrainer(model, device, optimizer, scheduler, class_weights=args.class_weights, use_amp=args.use_amp)
        else:
             trainer.optimizer = optimizer
             trainer.scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=(args.epochs-epoch)*len(train_dataloader))
        
        # --- 核心修改 3: 将模态丢弃率传递给 trainer ---
        train_loss = trainer.train_epoch(
            train_dataloader, 
            gradient_accumulation_steps=args.grad_accum_steps,
            modality_dropout_rate=args.modality_dropout_rate
        )
        
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
            
        checkpoint_path = os.path.join(args.save_dir, f"icu_model_epoch_{epoch+1}.pth")
        trainer.save_checkpoint(checkpoint_path, epoch + 1, is_best=is_best)
            
        if patience_counter >= patience:
            print(f"早停：指标连续 {patience} 个epoch无提升。")
            break
        
        print(f"Epoch耗时: {time.time() - start_time:.2f}秒")

    print("\n" + "="*60 + "\n训练完成总结:")
    if best_epoch > 0:
        print(f"最佳验证集指标 (在 Epoch {best_epoch}): F1 Score: {best_f1:.4f}, AUPRC: {best_auprc:.4f}")
        best_model_path = os.path.join(args.save_dir, 'icu_model_best_drop.pth')
        print(f"最佳模型权重已保存至: {best_model_path}")
    else:
        print("未能在训练过程中找到更优的模型。")
    print("=" * 60)
    return model

if __name__ == "__main__":
    # --- 核心开关 ---
    USE_SUBSET_MODE = False

    # --- 参数配置 ---
    class TrainingArgs:
        def __init__(self):
            self.data_path = "Data/ihm"
            self.bert_model_name = "yikuan8/Clinical-Longformer"
            self.save_dir = "my_model_weights_finetuned"
            self.epochs = 25
            self.learning_rate = 5e-5
            self.backbone_lr = 2e-5
            self.fine_tune_epochs = 3
            self.use_amp = True
            self.class_weights = torch.FloatTensor([0.57515432, 3.82648871])
            self.seed = 42
            self.effective_batch_size = 32
            self.physical_batch_size = 2 if not USE_SUBSET_MODE else 4
            self.grad_accum_steps = self.effective_batch_size // self.physical_batch_size
            
            # --- 核心修改 4: 在这里设置模态丢弃率 ---
            # 论文建议值为 0.7 (70%)
            self.modality_dropout_rate = 0.7 if not USE_SUBSET_MODE else 0.2
    
    args = TrainingArgs()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    
    print(f"--- 模式: {'快速调试 (子集)' if USE_SUBSET_MODE else '正式训练 (全量)'} ---")
    print(f"将使用有效批次: {args.effective_batch_size} (物理批次: {args.physical_batch_size}, 累积步数: {args.grad_accum_steps})")
    if args.modality_dropout_rate > 0:
        print(f"** 模态丢弃已启用，丢弃率: {args.modality_dropout_rate:.1f} **")
    
    # ... (后续的数据加载和测试逻辑保持不变) ...
    class DataArgs:
        def __init__(self):
            self.file_path = args.data_path
            self.train_batch_size, self.eval_batch_size = args.physical_batch_size, args.physical_batch_size * 2
            self.debug = False; self.max_length = 1024; self.num_of_notes = 5
            self.tt_max = 48; self.pad_to_max_length = True; self.notes_order = "Last"
            self.modeltype = "TS_Text"; self.model_name = args.bert_model_name
            self.chunk = False; self.ratio_notes_order = None

    data_args = DataArgs()
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
    
    print("--- 正在加载完整数据集 ---")
    full_train_dataset, _, _ = data_perpare(data_args, 'train', tokenizer)
    full_val_dataset, _, _ = data_perpare(data_args, 'val', tokenizer)
    full_test_dataset, _, _ = data_perpare(data_args, 'test', tokenizer)
    
    if USE_SUBSET_MODE:
        train_size = int(0.2 * len(full_train_dataset))
        train_indices = np.random.choice(len(full_train_dataset), train_size, replace=False)
        train_dataset = Subset(full_train_dataset, train_indices)
        print(f"使用训练子集: {len(train_dataset)} / {len(full_train_dataset)} 个样本")

        val_size = int(0.5 * len(full_val_dataset))
        val_indices = np.random.choice(len(full_val_dataset), val_size, replace=False)
        val_dataset = Subset(full_val_dataset, val_indices)
        print(f"使用验证子集: {len(val_dataset)} / {len(full_val_dataset)} 个样本")
    else:
        train_dataset, val_dataset = full_train_dataset, full_val_dataset
        print(f"使用全部训练数据: {len(train_dataset)} 个样本")
        print(f"使用全部验证数据: {len(val_dataset)} 个样本")

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=data_args.train_batch_size, collate_fn=TextTSIrgcollate_fn)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=data_args.eval_batch_size, collate_fn=TextTSIrgcollate_fn)
    
    trained_model = train_icu_model(args, train_dataloader, val_dataloader)
    
    print("\n--- 开始最终测试 ---")
    if USE_SUBSET_MODE:
        test_size = int(0.5 * len(full_test_dataset))
        test_indices = np.random.choice(len(full_test_dataset), test_size, replace=False)
        test_dataset = Subset(full_test_dataset, test_indices)
        print(f"使用测试子集: {len(test_dataset)} / {len(full_test_dataset)} 个样本")
    else:
        test_dataset = full_test_dataset
        print(f"使用全部测试数据: {len(test_dataset)} 个样本")
        
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=data_args.eval_batch_size, collate_fn=TextTSIrgcollate_fn)

    trained_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    test_f1, test_auprc = ICUTrainer(trained_model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), None, None).validate(test_dataloader)
    print("\n--- 测试集最终结果 ---")
    print(f"  - F1 Score: {test_f1:.4f}")
    print(f"  - AUPRC: {test_auprc:.4f}")