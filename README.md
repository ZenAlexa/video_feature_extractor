# 视频特征提取与分析系统

本项目提供了一套完整的多视角视频特征提取、分析和可视化工具，用于动作分割和识别任务。

## 功能概述

- 从视频中提取深度特征（使用ResNeXt-101预训练模型）
- 多视角视频的特征对齐和融合
- 特征可视化和分析（热力图、PCA降维、时序变化等）
- 潜在的行为边界检测

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- CUDA支持（可选，用于GPU加速）
- 其他依赖包：numpy, matplotlib, scikit-learn, tqdm

安装依赖：
```bash
pip install numpy matplotlib scikit-learn torch torchvision tqdm
```

## 工作流程

### 1. 特征提取

使用`video_feature_extractor.py`从视频中提取特征：

```bash
python video_feature_extractor.py --input_dir ./videos --output_dir ./features
```

参数说明：
- `--input_dir`: 包含视频文件的目录
- `--output_dir`: 保存提取特征的目录
- `--fps`: 特征提取的帧率（默认：24）
- `--resnext101_model_path`: 预训练模型路径（可选）

注意：视频文件命名遵循`{组名}_{视角名}.mov`格式，例如`Reducer_3_view1.mov`。

### 2. 多视角特征分析

对提取的特征进行分析和融合：

```bash
python simple_analyzer.py --feature_dir ./features
```

参数说明：
- `--feature_dir`: 包含特征文件的目录
- `--output_dir`: 分析结果保存目录（默认：`{feature_dir}/analysis`）

这一步会：
1. 自动识别并分组同一对象的不同视角
2. 对齐不同视角的时间轴
3. 计算视角间的相似度
4. 生成融合特征（平均融合和最大值融合）

### 3. 特征可视化

使用`view_features.py`查看和分析特征文件：

```bash
# 查看单个特征文件
python view_features.py ./features/Reducer_3_view1.npy

# 比较多个特征文件
python view_features.py ./features/Reducer_3_view1.npy ./features/Reducer_3_view2.npy

# 将可视化结果保存到指定目录
python view_features.py ./features/Reducer_3_*.npy --output ./visualizations

# 使用3D PCA可视化
python view_features.py ./features/Reducer_3_fused_mean.npy --pca 3
```

参数说明：
- `files`: 要分析的特征文件（支持多个文件）
- `--output`或`-o`: 可视化结果保存目录
- `--pca`: PCA降维维度（2或3）

可视化工具会展示：
- 特征基本信息（形状、类型、均值、标准差等）
- 特征热力图
- PCA降维后的特征分布
- 时序特征变化
- 潜在行为边界
- 多特征文件比较（如有多个文件）

## 常见问题及解决方案

### 1. 特征提取问题

如果遇到PIL相关的DLL加载错误，可使用`simple_analyzer.py`替代`multi_view_analyzer.py`进行分析，避免PIL依赖。

### 2. 无法加载特征文件

确保特征提取步骤已成功完成，并检查`./features`目录中是否存在`.npy`文件。特征文件命名应与视频文件名相匹配，如`Reducer_3_view1.npy`。

### 3. Scipy警告

运行过程中可能会出现Scipy的二进制兼容性警告，这通常不影响程序功能，可以忽略。

## 文件说明

核心文件：
- `video_feature_extractor.py`: 视频特征提取工具
- `simple_analyzer.py`: 简化版特征分析工具（不依赖PIL）
- `view_features.py`: 特征可视化工具
- `model.py`: 模型加载与特征提取核心
- `video_loader.py`: 视频加载工具
- `preprocessing.py`: 预处理功能
- `videocnn/`: 模型定义目录
- `model/`: 预训练模型目录

## 下一步工作

完成特征提取与融合后，可将融合特征文件（`*_fused_mean.npy`或`*_fused_max.npy`）用于后续的动作分割任务。这些融合特征结合了多视角信息，通常能够提供更好的性能。 
