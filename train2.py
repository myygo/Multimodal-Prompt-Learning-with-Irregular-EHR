#!/usr/bin/env python3
"""
[最终完整版] 用于训练 ICUPromptModel 的脚本。

功能亮点:
- 新增：完整的自动混合精度训练 (AMP) 支持，以提升速度、降低显存。
- 新增：更详细的训练信息输出，包括启动时的参数总览和学习率变化提示。
- 使用加权损失函数处理类别不平衡问题。
- 根据验证集上 F1-Score 和 AUPRC 的总和来选择并保存最佳模型。
- 包含早停（Early Stopping）机制。
- 支持“子集训练”模式，用于快速调试。
- 集成了模态丢弃（Modality Dropout）作为核心正则化策略。
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

# 确保 data.py 和 my_model.py 在同一目录下或Python路径中
from my_model import create_icu_model, ICUHyperParams
from data import data_perpare

class ICUTrainer:
    """封装了训练、验证和检查点保存的逻辑，并集成了AMP"""
    def __init__(self, model, device, optimizer, scheduler, class_weights=None, use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = use_amp and torch.cuda.is_available() # 只有在CUDA可用时才开启AMP
        
        # 打印参数量的逻辑可以保留
        params_to_train = [p for p in self.model.parameters() if p.requires_grad]
        print(f"  > 模型总参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  > 可训练参数量: {sum(p.numel() for p in params_to_train):,}")
        
        if class_weights is not None:
            class_weights = class_weights.to(device)
            print(f"正在使用带权重的损失函数，权重为: {class_weights.cpu().numpy()}")
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("  > 自动混合精度训练 (AMP) 已启用。")

    def train_epoch(self, dataloader, use_subset=False, modality_missing_rate=0.0):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        
        num_batches_to_use = len(dataloader)
        if use_subset:
            num_batches_to_use = int(len(dataloader) * 0.1)
            print(f"  [子集训练模式] 每个Epoch只使用 {num_batches_to_use} / {len(dataloader)} 个训练批次 (10%)")

        update_interval = max(1, len(dataloader) // 4)
        progress_bar = tqdm(
            dataloader, 
            desc="Training", 
            miniters=update_interval,
            mininterval=float('inf')
        )
        
        for batch_idx, batch_data in enumerate(progress_bar):
            if use_subset and batch_idx >= num_batches_to_use:
                break
            if batch_data is None: continue
            
            try:
                (ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input,
                 input_ids, attn_mask, note_time, note_time_mask, label) = [d.to(self.device) if torch.is_tensor(d) else None for d in batch_data]

                if self.model.training and torch.rand(1).item() < modality_missing_rate:
                    if torch.rand(1).item() < 0.5:
                        input_ids, attn_mask, note_time, note_time_mask = None, None, None, None
                    else:
                        ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input = None, None, None, None
                
                self.optimizer.zero_grad(set_to_none=True) # 使用 set_to_none=True 略微提升性能

                # 使用 AMP
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(
                        ts_input_sequences=ts_input_sequences, ts_mask_sequences=ts_mask_sequences,
                        ts_tt=ts_tt, reg_ts_input=reg_ts_input, input_ids=input_ids,
                        attn_mask=attn_mask, note_time=note_time, note_time_mask=note_time_mask,
                        label=label, intra_missing_ratio=0.1
                    )
                    if outputs is None: continue
                    loss = self.criterion(outputs, label)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                self.scheduler.step() # 在每个step后更新学习率

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'}, refresh=False)
                
            except Exception as e:
                print(f"\n训练批次 {batch_idx} 发生错误: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = total_loss / len(all_labels) if all_labels else 0
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0
        return avg_loss, accuracy

    def validate(self, dataloader, use_subset=False):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        
        num_batches_to_use = len(dataloader)
        if use_subset:
            num_batches_to_use = int(len(dataloader) * (1/3))
            print(f"  [子集验证模式] 每个Epoch只使用 {num_batches_to_use} / {len(dataloader)} 个验证批次 (1/3)")
        
        with torch.no_grad():
            update_interval = max(1, len(dataloader) // 4)
            progress_bar = tqdm(dataloader, desc="Validating", miniters=update_interval, mininterval=float('inf'))
            for batch_idx, batch_data in enumerate(progress_bar):
                if use_subset and batch_idx >= num_batches_to_use:
                    break
                if batch_data is None: continue
                
                try:
                    (ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input,
                     input_ids, attn_mask, note_time, note_time_mask, label) = [d.to(self.device) if torch.is_tensor(d) else None for d in batch_data]
                    
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        outputs = self.model(
                            ts_input_sequences=ts_input_sequences, ts_mask_sequences=ts_mask_sequences,
                            ts_tt=ts_tt, reg_ts_input=reg_ts_input, input_ids=input_ids,
                            attn_mask=attn_mask, note_time=note_time, note_time_mask=note_time_mask,
                            label=label, intra_missing_ratio=0.0
                        )
                    
                    if outputs is None: continue
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    predicted = torch.argmax(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

                except Exception as e:
                    print(f"\n验证批次发生错误: {e}")
                    continue
        
        if all_labels:
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='binary')
            auprc = average_precision_score(all_labels, all_probs)
            return accuracy, f1, auprc
        return 0.0, 0.0, 0.0

    def save_checkpoint(self, filepath, epoch, is_best=False):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict()}
        torch.save(checkpoint, filepath)
        # print(f"Checkpoint saved: {filepath}")
        if is_best:
            best_path = filepath.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

class ArgsPlaceholder:
    def __init__(self, data_path, batch_size, bert_model_name):
        self.file_path = data_path
        self.train_batch_size, self.eval_batch_size = batch_size, batch_size
        self.debug = False
        self.max_length, self.num_of_notes, self.tt_max = 512, 5, 48
        self.pad_to_max_length, self.notes_order = True, "Last"
        self.modeltype, self.model_name = "TS_Text", bert_model_name
        self.chunk, self.ratio_notes_order = False, None

def train_icu_model(data_path, epochs, batch_size, learning_rate, modality_dropout_rate, save_dir, bert_model_name, use_subset, use_amp, freeze_backbone):
    print("开始训练 ICUPromptModel")
    print("=" * 60)
    
    print("训练参数总览:")
    print(f"  - 数据路径: {data_path}")
    print(f"  - BERT模型: {bert_model_name}")
    print(f"  - 保存目录: {save_dir}")
    print(f"  - 训练轮数 (Epochs): {epochs}")
    print(f"  - 批处理大小 (Batch Size): {batch_size}")
    print(f"  - 学习率 (Learning Rate): {learning_rate}")
    print(f"  - 模态丢弃率: {modality_dropout_rate}")
    print(f"  - 使用混合精度 (AMP): {use_amp}")
    print(f"  - 冻结BERT骨干: {freeze_backbone}")
    print(f"  - 使用数据子集: {use_subset}")
    print("-" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    torch.manual_seed(42); np.random.seed(42)
    
    print("\n[步骤 1] 准备数据加载器...")
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    args = ArgsPlaceholder(data_path, batch_size, bert_model_name)
    train_dataset, _, train_dataloader = data_perpare(args, 'train', tokenizer)
    _, _, val_dataloader = data_perpare(args, 'val', tokenizer)
    
    fixed_weights = [0.57515432, 3.82648871]
    class_weights = torch.FloatTensor(fixed_weights)
    
    print("\n[步骤 2] 创建模型...")
    print(f"正在从Hugging Face加载预训练模型: {bert_model_name}")
    clinical_bert_model = AutoModel.from_pretrained(bert_model_name)
    model = create_icu_model(clinical_bert_model, freeze_backbone=freeze_backbone)
    
    print("\n[步骤 3] 初始化训练器...")

    backbone_lr = 2e-5  # 预训练主干的低学习率
    head_lr = learning_rate # 自定义模块使用您设置的较高学习率

    print(f"  > 差分学习率: Backbone LR = {backbone_lr}, Fusion Head LR = {head_lr}")

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "text_encoder" in n],
            "lr": backbone_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "text_encoder" not in n],
            "lr": head_lr,
            "weight_decay": 1e-4
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    num_training_steps = epochs * len(train_dataloader)
    num_warmup_steps = int(0.1 * num_training_steps) # 使用10%的步数进行预热
    print(f"  > 总训练步数: {num_training_steps}, 预热步数: {num_warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    num_training_steps = epochs * len(train_dataloader)
    num_warmup_steps = int(0.1 * num_training_steps) # 使用10%的步数进行预热
    print(f"  > 总训练步数: {num_training_steps}, 预热步数: {num_warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    # --- [修改结束] ---


    print("\n[步骤 3.5] 初始化训练器...")
    # --- [修改代码：传入新的optimizer和scheduler] ---

    trainer = ICUTrainer(model, device, optimizer, scheduler, class_weights=class_weights, use_amp=use_amp)
    
    best_val_metric_sum, best_f1, best_auprc, best_epoch = 0, 0, 0, 0
    patience, patience_counter = 8, 0
    
    print(f"\n[步骤 4] 开始训练，共 {epochs} 个 epoch...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        start_time = time.time()
        
        train_loss, train_acc = trainer.train_epoch(train_dataloader, use_subset=use_subset, modality_missing_rate=modality_dropout_rate)
        
        if val_dataloader:
            val_acc, val_f1, val_auprc = trainer.validate(val_dataloader, use_subset=use_subset)
            print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
            print(f"验证 - 准确率: {val_acc:.4f}, F1分数: {val_f1:.4f}, AUPRC: {val_auprc:.4f}")
            
            trainer.scheduler.step(val_f1 + val_auprc)
            current_metric_sum = val_f1 + val_auprc
            is_best = current_metric_sum > best_val_metric_sum
            
            if is_best:
                best_val_metric_sum = current_metric_sum
                best_f1, best_auprc, best_epoch = val_f1, val_auprc, epoch + 1
                patience_counter = 0
                print(f"🎉 新的最佳指标: F1+AUPRC = {best_val_metric_sum:.4f} (F1: {best_f1:.4f}, AUPRC: {best_auprc:.4f})")
            else:
                patience_counter += 1
                print(f"验证集指标无提升 ({patience_counter}/{patience})")
            
            checkpoint_path = os.path.join(save_dir, f"icu_model_epoch_{epoch+1}.pth")
            trainer.save_checkpoint(checkpoint_path, epoch + 1, is_best=is_best)
            
            if patience_counter >= patience:
                print(f"早停：指标连续 {patience} 个epoch无提升。")
                break
        
        print(f"Epoch耗时: {time.time() - start_time:.2f}秒")

    print(f"\n{'='*60}")
    print("训练完成总结:")
    if best_epoch > 0:
        print(f"最佳验证集指标 (在 Epoch {best_epoch}):")
        print(f"  - F1 Score: {best_f1:.4f}")
        print(f"  - AUPRC: {best_auprc:.4f}")
        print(f"  - F1 + AUPRC: {best_val_metric_sum:.4f}")
        best_model_path = os.path.join(save_dir, f'icu_model_epoch_{best_epoch}_best.pth')
        print(f"最佳模型权重已保存至: {best_model_path}")
    else:
        print("未能在训练过程中找到更优的模型。")
    print("=" * 60)

if __name__ == "__main__":
    # --- 主要参数配置区 ---
    # 您可以在这里轻松地调整所有关键超参数
    
    # 数据和模型路径
    DATA_PATH = "data/ihm"
    BERT_MODEL_NAME = "yikuan8/Clinical-Longformer"
    SAVE_DIR = "icu_model_weights_final"
    
    # 训练控制参数
    EPOCHS = 25
    BATCH_SIZE = 16       # 4090显卡 + AMP，可以尝试更大的batch size
    LEARNING_RATE = 4e-4  # 冻结模式下，可以使用更高的学习率
    MODALITY_DROPOUT_RATE = 0.5
    
    # 功能开关
    USE_AMP = True         # 设置为 True 来开启混合精度训练
    FREEZE_BACKBONE = True # 设置为 True 来冻结BERT/Longformer的参数
    
    # 快速测试模式
    USE_SUBSET_MODE = False
    
    # --- 运行区 ---
    train_file = os.path.join(DATA_PATH, 'trainp2x_data.pkl')
    if not os.path.exists(train_file):
        print(f"错误：找不到训练数据文件: '{train_file}'")
    else:
        train_icu_model(
            data_path=DATA_PATH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            modality_dropout_rate=MODALITY_DROPOUT_RATE,
            save_dir=SAVE_DIR,
            bert_model_name=BERT_MODEL_NAME,
            use_subset=USE_SUBSET_MODE,
            use_amp=USE_AMP,
            freeze_backbone=FREEZE_BACKBONE
        )