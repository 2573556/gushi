# dataset.py - 数据加载和预处理

import json
import torch
from torch.utils.data import Dataset
from collections import Counter

class PoetryDataset(Dataset):
    """古诗数据集类"""
    
    def __init__(self, data_path, max_seq_len=64):
        # 1. 加载 JSON 数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.poems = json.load(f)
        
        # 2. 提取诗句文本（处理不同的 JSON 格式）
        self.poetry_lines = []
        for poem in self.poems:
            # 处理全唐诗的数据格式
            if 'paragraphs' in poem:
                for para in poem['paragraphs']:
                    # 清洗：去除空格、标点等
                    line = para.replace(' ', '').replace('，', '').replace('。', '')
                    if len(line) >= 4:  # 只保留长度>=4的诗句
                        self.poetry_lines.append(line)
            elif 'content' in poem:
                line = poem['content'].replace(' ', '').replace('，', '').replace('。', '')
                if len(line) >= 4:
                    self.poetry_lines.append(line)
        
        # 3. 构建字符到索引的映射
        all_chars = []
        for line in self.poetry_lines:
            all_chars.extend(list(line))
        
        char_counts = Counter(all_chars)
        self.chars = [char for char, _ in char_counts.most_common()]
        self.char_to_idx = {char: idx+1 for idx, char in enumerate(self.chars)}  # 0 留给填充
        self.idx_to_char = {idx+1: char for idx, char in enumerate(self.chars)}
        
        # 添加填充符和未知字符
        self.char_to_idx['<PAD>'] = 0
        self.idx_to_char[0] = '<PAD>'
        self.char_to_idx['<UNK>'] = len(self.chars) + 1
        self.idx_to_char[len(self.chars) + 1] = '<UNK>'
        
        self.vocab_size = len(self.char_to_idx)
        self.max_seq_len = max_seq_len
        
        # 4. 将所有诗句转换为索引序列
        self.data = []
        for line in self.poetry_lines:
            indices = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in list(line)]
            self.data.append(indices)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取一个序列
        seq = self.data[idx]
        
        # 如果序列太长，截断
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]
        
        # 填充到 max_seq_len
        padded_seq = seq + [0] * (self.max_seq_len - len(seq))
        
        # 输入：[:-1] 输出：[1:]
        x = torch.tensor(padded_seq[:-1], dtype=torch.long)
        y = torch.tensor(padded_seq[1:], dtype=torch.long)
        
        return x, y

def get_vocab_size(dataset):
    """获取词汇表大小"""
    return dataset.vocab_size
