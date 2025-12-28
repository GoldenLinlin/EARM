# EARM - Edit-Aware Reward Model

## 文件结构

```
earm_project/
├── build_earm_data.py    # 数据构建脚本
├── earm_model.py         # EARM 模型定义
├── earm_dataset.py       # 数据集和数据加载器
├── train_earm.py         # 训练脚本
├── requirements.txt      # 依赖
├── scripts/
│   ├── run_build_data.sh # 数据构建脚本 (待填充)
│   └── run_train_earm.sh # 训练脚本 (待填充)
├── data/                 # 数据目录
└── outputs/              # 输出目录
```

## 使用方法 (示例)

1. 安装依赖
```bash
pip install spacy
python -m spacy download en_core_web_sm
pip install -r requirements.txt
```

2. 构建数据 (填充 scripts/run_build_data.sh 或直接运行 build_earm_data.py)
```bash
./scripts/run_build_data.sh
# or:
python build_earm_data.py --merge_folder /path/to/folder --output_dir ./data/earm_train_data.json --model_name your/tokenizer
```

3. 训练模型 (填充 scripts/run_train_earm.sh 或直接运行 train_earm.py)
```bash
./scripts/run_train_earm.sh
# or:
accelerate launch --num_processes 1 train_earm.py --train_data ./data/earm_train_data.json --model_name your/model --output_dir ./outputs/earm
```

## 说明

- 将代码内容复制到对应的 Python 文件中：build_earm_data.py, earm_model.py, earm_dataset.py, train_earm.py
- 根据需要完善 scripts 目录下的 shell 脚本
- data/ 用于放置原始及处理后数据；outputs/ 用于保存模型与日志
