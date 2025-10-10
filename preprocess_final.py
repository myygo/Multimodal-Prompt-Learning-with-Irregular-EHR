import os
import pickle
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
# 从你原始的 data.py 中导入需要的函数
from data import F_impute, load_data as load_original_data
import numpy as np

def preprocess_and_save(original_data_path, output_dir, mode, tokenizer, max_len=1024, tt_max=48):
    """
    执行最终的预处理步骤 (impute 和 tokenize) 并将结果保存到新目录。
    """
    print(f"--- 开始处理 '{mode}' 数据 ---")
    
    # 1. 从原始路径加载半成品数据
    try:
        # 使用原始的 load_data 函数
        original_data = load_original_data(file_path=original_data_path, mode=mode, debug=False)
    except FileNotFoundError as e:
        print(f"❌ 错误：找不到 '{mode}' 模式的原始数据，已跳过。请检查路径 '{original_data_path}'。")
        print(f"   详细错误: {e}")
        return

    processed_data = []
    for item in tqdm(original_data, desc=f"正在处理 {mode} 数据"):
        
        # 跳过可能不完整的数据项
        if 'irg_ts' not in item or 'text_data' not in item:
            print(f"警告：跳过一个不完整的数据项。")
            continue

        # 2. 执行 F_impute 计算
        reg_ts = F_impute(item['irg_ts'], item['ts_tt'], item['irg_ts_mask'], 1, tt_max)
        
        # 3. 执行 Tokenization
        text_token = []
        atten_mask = []
        for t in item['text_data']:
            inputs = tokenizer.encode_plus(
                t,
                padding="max_length",
                max_length=max_len,
                add_special_tokens=True,
                return_attention_mask=True,
                truncation=True
            )
            text_token.append(inputs['input_ids'])
            
            attention_mask = inputs['attention_mask']
            # 兼容 Longformer 的特殊 attention mask 处理
            if "Longformer" in tokenizer.name_or_path:
                attention_mask[0] += 1
            atten_mask.append(attention_mask)

        # 4. 组装所有处理好的数据到一个新的字典中
        new_item = item.copy()
        # 存储为 numpy array 以便后续快速加载
        new_item['reg_ts_imputed'] = np.array(reg_ts, dtype=np.float32)
        new_item['input_ids_tokenized'] = [np.array(t, dtype=np.int32) for t in text_token]
        new_item['attention_mask_tokenized'] = [np.array(a, dtype=np.int32) for a in atten_mask]
        
        processed_data.append(new_item)

    # 5. 保存到新的 Data_Final 文件夹
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{mode}_final_data.pkl")
    
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
        
    print(f"✅ '{mode}' 数据处理完成，已保存至: {output_path}\n")

if __name__ == "__main__":
    # --- 配置 ---
    ORIGINAL_DATA_PATH = "Data/ihm"
    FINAL_DATA_PATH = "Data_Final"
    BERT_MODEL_NAME = "yikuan8/Clinical-Longformer"
    
    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # --- 对所有数据模式执行预处理 ---
    preprocess_and_save(ORIGINAL_DATA_PATH, FINAL_DATA_PATH, 'train', tokenizer)
    preprocess_and_save(ORIGINAL_DATA_PATH, FINAL_DATA_PATH, 'val', tokenizer)
    preprocess_and_save(ORIGINAL_DATA_PATH, FINAL_DATA_PATH, 'test', tokenizer)
    
    print("🎉 所有数据预处理完成！现在你可以使用 data_final.py 来进行训练了。")
