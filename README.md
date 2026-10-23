# 智行先锋 · 高速服务区车位智能解析与优化生态系统

> 长安大学大学生创新训练计划项目成果
> 基于改进 **YOLOv8** 的高速服务区车位检测与管理系统

---

## 项目简介

我国机动车保有量已突破 4.4 亿辆、高速公路总里程达 18.4 万公里，高速服务区「停车难、找位难、管理粗放」的问题日益突出。传统地感线圈、超声波与人工巡查方案在**夜间低光、强光污染、雨雾雪、大俯角图像畸变、车辆密集遮挡**等复杂场景下精度骤降、延迟高、运维成本大。

本项目以 **YOLOv8n** 为基线模型，针对高速服务区开放场景进行结构化改进，融合车位状态识别、违停检测、车流统计与短时流量预测，构建了一套可直接接入既有监控摄像头的**车位智能解析与优化生态系统**，无需额外专用传感器，部署成本低、落地性强。

## 主要工作

### 1. 改进 YOLOv8 检测网络

| 改进模块 | 作用 |
| --- | --- |
| **DSPPA**（膨胀空间金字塔池化） | 增强多尺度、畸变、模糊车位特征提取能力，减少深层语义信息丢失 |
| **ECA**（通道注意力） | 强化车辆/车位特征响应、抑制背景干扰，提升光照鲁棒性 |
| **BiFPN**（双向特征金字塔） | 替换 PANet，加强遮挡目标残缺特征的双向融合 |
| **DIoU-NMS** | 替代硬 NMS，对重叠框做置信度衰减，保留被遮挡车辆的检测框 |

### 2. 加权 IoU + 匈牙利全局车位匹配

- 将车位划分为**中心区域（高权重）**与**边缘区域（低权重）**，计算加权占用率，降低不同拍摄角度带来的匹配误差；
- 以加权 IoU 构建代价矩阵，使用**匈牙利算法**全局搜索最优「车位—车辆」匹配组合，解决**错配**与**重复分配**问题；
- 结合车位类型与车型判断违停行为（如小型客车占用货车位），实时更新空闲 / 占用 / 违停三态。

### 3. 可视化管理系统

基于 Python + PyTorch + OpenCV + Streamlit 的模块化系统，包含视频采集、图像预处理、车位检测、违停判断、智能分配、可视化展示与数据存储七大模块，支持：

- 实时视频流接入与车位状态动态显示（绿 / 红 / 黄三色标注）
- 违停自动检测、高亮标注、弹窗预警与记录留存
- 车流统计、车位占用率与高峰时段分析
- 历史数据多维度查询、图表展示与报表导出
- 基于 LSTM 的 1–3 小时短时车流量预测
- 边缘设备适配（INT8 量化，Jetson Nano / 树莓派，单张推理 ≤ 5 ms）

### 4. 场景化数据集与标注流程

采集城市近郊、高速主干道、山区路段 3 类典型服务区，覆盖晴天、夜间、雨天、雾天、强光、大俯角畸变、密集遮挡 7 类场景。经筛选与统一分辨率处理后采用**多边形标注**（对比矩形标注准确率由 65% 提升至 82%），标注类别为 `car` / `parking lot` / `p`，并通过 `json_to_txt.py` 将 LabelMe JSON 批量转换为 YOLO TXT 格式。

## 实验结果

| 指标 | 结果 |
| --- | --- |
| mAP@0.5（复杂场景） | **0.89** |
| 车辆检测准确率 | 0.92 |
| 车位识别准确率 | 0.87 |
| 夜间 / 雨天 mAP | ≥ 0.81 |
| 极端视角 mAP | ≥ 0.85 |
| 稳定性 | 7×24 小时连续运行无崩溃、无性能衰减 |

关键训练参数（对照实验确定）：**epochs = 100、batch_size = 4**。

## 项目结构

```
yolov8-main/
├── ultralytics/            # YOLOv8 源码（v8.0.49），模型定义、训练/推理引擎、工具函数
│   ├── yolo/               #   引擎、数据处理、损失函数、指标计算
│   ├── models/v8/          #   网络结构配置（yolov8n/s/m/l/x.yaml）
│   └── datasets/           #   公开数据集配置文件（coco.yaml、VOC.yaml 等）
├── json_to_txt.py          # LabelMe JSON → YOLO TXT 标注格式批量转换脚本
├── label.txt               # 类别映射（0:car 1:parking lot 2:p）
├── Test.py                 # 单张图片推理冒烟测试
├── 66.py                   # CUDA 环境可用性检查
├── requirements.txt        # 依赖清单
└── datasets/               # 【不随仓库发布】自建高速服务区数据集与标注
```

## 快速开始

```bash
# 1. 安装依赖（建议 Python 3.8+ / PyTorch 1.7+）
pip install -r requirements.txt
```

**训练**

```python
from ultralytics import YOLO

model = YOLO("yolov8n.yaml").load("yolov8n.pt")   # 从预训练权重初始化
model.train(data="datasets/Dong/my_data.yaml", epochs=100, batch=4, imgsz=640)
```

或使用命令行：

```bash
yolo detect train model=yolov8n.pt data=datasets/Dong/my_data.yaml epochs=100 batch=4 imgsz=640
```

**推理**

```python
from ultralytics import YOLO

yolo = YOLO("runs/detect/train/weights/best.pt")
yolo(source="test.jpg", save=True)
```

**标注转换**

修改 `json_to_txt.py` 中的 `json_floder_path` 与 `txt_outer_path` 后运行：

```bash
python json_to_txt.py
```

## 关于数据集

`datasets/` 目录下的数据来自实地采集与人工标注，**涉及未开源授权，不随本仓库发布**，已在 `.gitignore` 中排除。请自行准备数据，并按如下结构组织与修改 `my_data.yaml` 中的路径与类别：

```
datasets/Dong/
├── images/train2025/    # 训练集图像
├── images/val2025/      # 验证集图像
├── labels/train2025/    # 训练集标签（YOLO TXT）
└── labels/val2025/      # 验证集标签
```

## 团队成员

长安大学信息工程学院本科生项目组（负责人 1 名、成员 3 名），指导教师为学院计算机视觉方向副教授。

## 成果

- 计算机软件著作权 1 项：《高速服务区车位智能解析系统 V1.0》
- 发明专利申请 1 项（在审）
- 高速服务区场景化数据集、改进 YOLOv8 模型权重与全套训练/测试脚本

## 许可证与致谢

本项目代码基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)（v8.0.49，AGPL-3.0）修改与扩展，遵循其许可证要求，详见 [LICENSE](./LICENSE)。

感谢长安大学教务处、创新创业学院与信息工程学院的支持。
