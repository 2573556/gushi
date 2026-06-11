# train.py - 模型训练脚本

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from config import *
from dataset import PoetryDataset, get_vocab_size
from model import PoetryLSTM

def train():
    # 1. 设置设备（CPU 或 GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 2. 加载数据集
    print("加载数据中...")
    dataset = PoetryDataset(DATA_PATH, MAX_SEQ_LEN)
    vocab_size = get_vocab_size(dataset)
    print(f"词汇表大小: {vocab_size}")
    print(f"诗句数量: {len(dataset)}")
    
    # 3. 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. 初始化模型
    model = PoetryLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充符 <PAD>
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 6. 创建保存目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 7. 训练循环
    print("开始训练...")
    train_losses = []
    
    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            # 将数据移到设备
            x, y = x.to(device), y.to(device)
            
            # 初始化隐藏状态
            hidden = model.init_hidden(x.size(0), device)
            
            # 前向传播
            output, hidden = model(x, hidden)
            
            # 计算损失
            loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 打印进度
            if batch_idx % PRINT_EVERY == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}] 完成, 平均损失: {avg_loss:.4f}")
        
        # 每 10 个 epoch 保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')
            print(f"模型已保存: checkpoints/model_epoch_{epoch+1}.pth")
    
    # 8. 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Training Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/loss_curve.png')
    plt.show()
    print("损失曲线已保存: logs/loss_curve.png")
    
    # 9. 保存最终模型
    torch.save(model.state_dict(), 'checkpoints/model_final.pth')
    print("训练完成！最终模型已保存: checkpoints/model_final.pth")
    
    # 10. 保存词汇表（供生成脚本使用）
    import pickle
    with open('checkpoints/vocab.pkl', 'wb') as f:
        pickle.dump({
            'char_to_idx': dataset.char_to_idx,
            'idx_to_char': dataset.idx_to_char
        }, f)
    print("词汇表已保存: checkpoints/vocab.pkl")

if __name__ == "__main__":
    train()
