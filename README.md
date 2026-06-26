# Gesture Glove + Vision → Unity Hand Tracking

> D435i → ViTPose + HAMER → UDP → Unity 手部实时追踪系统

## 仓库结构

`
gesture-glove-vision-unity/
├── python/                          ← Python 推理脚本
│   ├── run_vitpose_v3.py            ← 主推理 (ViTPose + HAMER → UDP)
│   ├── run_mediapipe_hand.py        ← 备用方案 (MediaPipe → UDP)
│   ├── d435i_hamer_fusion.py        ← HAMER + IMU 本地可视化
│   └── requirements.txt             ← Python 依赖
├── unity/Assets/Scenes/             ← Unity C# 接收端
│   ├── VisionBridge.cs              ← UDP 接收 + IMU 切换桥梁
│   ├── HandMotionManager.cs         ← IMU 手势解算与校准
│   ├── FingerSolver.cs              ← 手指骨骼求解器
│   └── ...其他脚本
├── hamer_source/                    ← HAMER 源码 (权重需要自行下载)
│   ├── hamer/                       ← HAMER Python 包
│   ├── vitpose_model.py             ← ViTPose 封装
│   ├── third-party/ViTPose/         ← ViTPose 配置
│   └── setup.py
├── docs/
│   └── PROJECT_REPRODUCTION.md      ← 完整复现文档
├── .gitignore
└── README.md
`

## 快速开始

**完整复现步骤请参阅** [docs/PROJECT_REPRODUCTION.md](docs/PROJECT_REPRODUCTION.md)

### 1. 环境安装

`ash
conda create -n hamer python=3.10 -y
conda activate hamer
pip install -r python/requirements.txt
`

### 2. 权重下载

| 模型 | 文件 | 位置 | 下载 |
|------|------|------|------|
| HAMER | hamer.ckpt (~300MB) | hamer_source/_DATA/hamer_ckpts/checkpoints/ | [Hugging Face](https://huggingface.co/geopavlakos/hamer) |
| ViTPose+ | wholebody.pth (~2.3GB) | hamer_source/_DATA/vitpose_ckpts/vitpose+_huge/ | [GitHub Release](https://github.com/ViTAE-Transformer/ViTPose/releases) |

### 3. 启动

`ash
# 先启动 Unity → Play
# 再启动 Python
cd python
python run_vitpose_v3.py
`

## 关键修复记录

- CPU 1790% → OMP/MKL 线程池硬限制
- 7 秒视频卡死 → JIT 编译禁用
- 手部标记残留 → hands_confirmed 状态机
- 双手重叠检测 → IoU 去重

**完整踩坑记录**: [docs/PROJECT_REPRODUCTION.md](docs/PROJECT_REPRODUCTION.md)

## 硬件要求

- NVIDIA GPU 8GB+ (RTX 4080 Laptop 12GB 已验证)
- Intel RealSense D435i (或其他 USB 摄像头)
- Windows 10 / Ubuntu 20.04+
