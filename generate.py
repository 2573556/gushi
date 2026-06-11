# generate.py - 古诗生成脚本（修复最后一行多字问题）

import torch
import torch.nn as nn
import pickle
import random

from config import *
from model import PoetryLSTM

def format_poem(poem_text):
    """给生成的文本添加标点，按五言格式排版"""
    if not poem_text:
        return ""
    
    result = []
    length = len(poem_text)
    
    for i, ch in enumerate(poem_text):
        result.append(ch)
        
        # 判断位置（从1开始计数）
        pos = i + 1
        
        # 每5个字加逗号（但不是10的倍数位置）
        if pos % 5 == 0 and pos % 10 != 0:
            result.append('，')
        # 每10个字加句号和换行
        elif pos % 10 == 0:
            result.append('。\n')
        # 如果是最后一个字且不是句号结尾，加句号
        elif pos == length and length % 10 != 0:
            result.append('。')
    
    return ''.join(result)

def generate_poem(model, start_chars, idx_to_char, char_to_idx, max_len=40, temperature=0.8):
    device = next(model.parameters()).device
    
    generated = list(start_chars)
    input_seq = [char_to_idx.get(c, char_to_idx.get('<UNK>', 1)) for c in start_chars]
    
    hidden = model.init_hidden(1, device)
    
    for _ in range(max_len):
        seq = input_seq[-MAX_SEQ_LEN:] if len(input_seq) > MAX_SEQ_LEN else input_seq
        x = torch.tensor([seq]).to(device)
        
        output, hidden = model(x, hidden)
        logits = output[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()
        
        if next_idx == 0:
            continue
        
        next_char = idx_to_char.get(next_idx, '?')
        
        # 遇到句号且长度足够就停止
        if next_char in ['。', '！', '？'] and len(generated) > 20:
            generated.append(next_char)
            break
        
        generated.append(next_char)
        input_seq.append(next_idx)
        
        # 限制生成长度
        if len(generated) >= max_len:
            break
    
    return ''.join(generated)

def load_model(model_path, vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    char_to_idx = vocab['char_to_idx']
    idx_to_char = vocab['idx_to_char']
    vocab_size = len(char_to_idx)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = PoetryLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    import os
    import glob
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    elif os.path.exists('checkpoints/model_final.pth'):
        model.load_state_dict(torch.load('checkpoints/model_final.pth', map_location=device, weights_only=False))
    elif os.path.exists('checkpoints/mode1_final.pth'):
        model.load_state_dict(torch.load('checkpoints/mode1_final.pth', map_location=device, weights_only=False))
    else:
        models = glob.glob('checkpoints/*.pth')
        if models:
            latest = max(models, key=os.path.getctime)
            print(f"加载最新模型: {latest}")
            model.load_state_dict(torch.load(latest, map_location=device, weights_only=False))
        else:
            raise FileNotFoundError("找不到模型文件")
    
    model.eval()
    return model, idx_to_char, char_to_idx, device

def main():
    print("加载模型中...")
    model_path = "checkpoints/model_final.pth"
    vocab_path = "checkpoints/vocab.pkl"
    
    try:
        model, idx_to_char, char_to_idx, device = load_model(model_path, vocab_path)
        print("模型加载成功！")
    except Exception as e:
        print(f"错误：{e}")
        return
    
    start_words_list = ["春", "秋月", "江", "山", "夜", "白", "青"]
    
    print("\n" + "="*50)
    print("古诗生成结果（五言诗）")
    print("="*50 + "\n")
    
    for start_word in start_words_list:
        poem = generate_poem(
            model, start_word, idx_to_char, char_to_idx,
            max_len=40, temperature=0.8
        )
        formatted = format_poem(poem)
        print(f"《{start_word}》")
        print(formatted)
        print()
    
    print("\n" + "="*50)
    print("交互模式（输入起始字，如：床前明月光）")
    print("="*50)
    
    while True:
        start = input("\n起始字: ").strip()
        if start.lower() == 'q':
            break
        if start:
            poem = generate_poem(model, start, idx_to_char, char_to_idx, max_len=40, temperature=0.8)
            formatted = format_poem(poem)
            print(f"\n{formatted}\n")

if __name__ == "__main__":
    main()
