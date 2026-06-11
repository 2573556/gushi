# 深度学习实验3：中文古诗生成

## 项目简介
基于 LSTM 的字符级古诗生成模型，使用全唐诗数据集训练。

## 文件说明
- `config.py` - 超参数配置
- `dataset.py` - 数据加载与预处理
- `model.py` - LSTM 模型定义
- `train.py` - 训练脚本
- `generate.py` - 古诗生成脚本（运行这个）

## 环境配置
```bash
# 安装依赖
pip install -r requirements.txt
使用方法
生成古诗（直接使用）
bash
python generate.py
训练模型（可选）
bash
python train.py
交互操作
操作	说明
输入起始字符	输入1-4个汉字，如：春江、床前明月
生成诗句	自动生成后续内容
退出程序	输入 q 后按回车
GitHub 仓库
https://github.com/2573556/gushi

作者：张星琪
学号：202400400179
