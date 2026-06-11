# model.py - LSTM 古诗生成模型

import torch
import torch.nn as nn
import torch.nn.functional as F

class PoetryLSTM(nn.Module):
    """基于 LSTM 的古诗生成模型"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super(PoetryLSTM, self).__init__()
        
        # 词嵌入层：将字符索引转换为稠密向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层：将隐藏状态映射到词汇表大小
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout 层（防止过拟合）
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        """
        x: [batch_size, seq_len] 输入序列
        hidden: LSTM 的初始隐藏状态
        返回: [batch_size, seq_len, vocab_size] 预测概率
        """
        # 嵌入层
        x = self.embedding(x)  # [batch, seq_len, embedding_dim]
        
        # Dropout
        x = self.dropout(x)
        
        # LSTM
        x, hidden = self.lstm(x, hidden)  # x: [batch, seq_len, hidden_dim]
        
        # Dropout
        x = self.dropout(x)
        
        # 全连接层
        x = self.fc(x)  # [batch, seq_len, vocab_size]
        
        return x, hidden
    
    def init_hidden(self, batch_size, device):
        """初始化隐藏状态为全零"""
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        return (h0, c0)

class PoetryLSTMWithAttention(nn.Module):
    """带注意力机制的 LSTM 模型（进阶版）"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super(PoetryLSTMWithAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = self.dropout(x)
        x, hidden = self.lstm(x, hidden)
        
        # 注意力权重
        attention_weights = F.softmax(self.attention(x), dim=1)
        context = torch.sum(attention_weights * x, dim=1)
        
        x = self.fc(x)
        return x, hidden
