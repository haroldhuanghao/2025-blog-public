# 背景

- 在复杂的环境中找到仪表，控制摄像头移动、放大，得到特写图，便于后续进一步判断（若容易判断则可直接判断）

- 缺乏人工标注数据，希望系统自学习/自适应

# 痛点

* 数据样本少（尤其是负面样本），且与0样本或低样本冲突
* 类别名字较专业，通用 prompt 不好写
* 目标外观和通用数据集差很多
* 小目标、密集目标、遮挡目标多
* 同类内部差异大，异类之间又很像

# 一、目标

做一个“**专业领域主动视觉检测与追踪系统**”：

1. 用 **少量人工标注样本**，把 Open-GroundingDINO 适配到你的专业领域。
2. 用它去 **自动标注大量未标注图像/视频帧**。
3. 用 **SAM / SAM2** 精化 box 或 mask。
4. 用 **DINOv3** 过滤伪标签、做重识别特征。
5. 用清洗后的数据训练 **YOLO 学生模型**，满足实时部署。
6. 在线阶段用 **YOLO + 跟踪器 + Le-WorldModel** 控制摄像头转向、放大、持续跟踪。

# 二、模块级技术路线

阶段 A：专业领域少样本适配教师模型
------------------

### A1. 构建 few-shot 种子集

每个类别先准备：

* 1-shot 验证集
* 4-shot 起步集
* 8-shot 稳定集

数据要求：

* 不同尺度
* 不同光照
* 不同遮挡
* 不同背景
* 类间易混样本也要覆盖

### A2. 用 Open-GroundingDINO 做领域适配

作用：

* 把通用 open-set detector 变成“你的专业领域教师模型”
* 后续负责自动打框

为什么放这里：  
Open-GroundingDINO 仓库明确写了它可以在你自己的数据集上 fine-tune，也可以从头 pretrain。

### A3. 教师输出

输出内容建议统一成：

* `bbox`
* `class_name`
* `score`
* `text_prompt`
* `image_id`
* `teacher_version`

* * *

阶段 B：自动标注与精化
------------

### B1. 教师自动打框

输入：

* 未标注图片
* 视频抽帧
* 长视频关键帧

输出：

* 原始检测框

### B2. 用 Grounded-SAM / SAM2 精化

你这里分两种用法：

#### 静态图精化

* GroundingDINO 打框
* SAM 细化目标轮廓
* 用 mask 反推更紧的 box

#### 视频追踪精化

* Grounded-SAM 2 已明确面向开放世界视频 tracking 场景
* 可用于把单帧检测延伸为视频对象跟踪与 mask propagation。

### B3. 标注输出格式

统一存成两份：

* `COCO JSON`：方便分析与可视化
* `YOLO txt`：方便训练学生模型

* * *

阶段 C：DINOv3 伪标签质量控制
-------------------

DINOv3 官方定位是通用高质量视觉表征模型，尤其强调 dense features。对你最有价值的不是“直接当检测器”，而是做表征质量控制。

### C1. 伪标签过滤

对每个检测框 crop：

* 提取 DINOv3 embedding
* 类内聚类
* 离群样本剔除
* 重复样本合并
* 类间近邻冲突标记

### C2. 难例挖掘

把下面这些样本优先送人工复核：

* 教师高分但 DINOv3 特征离群
* 同图中多个高度相似框
* 邻近帧预测跳变大
* SAM 精化前后面积差异异常大

### C3. 在线 ReID 特征

给每个 track 保留：

* `appearance_embedding`
* `last_seen_ts`
* `gallery_features`

作用：

* 短时遮挡后找回
* 画面外重入时重关联
* 降低 ID switch

* * *

阶段 D：YOLO 学生模型训练
----------------

Ultralytics 当前文档仍推荐通过 `pip install -U ultralytics` 安装，并支持用 CLI / Python 训练自定义数据。训练模式面向单机和多 GPU，自定义数据集 YAML 是标准入口。

### D1. 数据输入

来源：

* 人工真值
* 教师高置信伪标签
* DINOv3 清洗后样本
* 视频扩充帧

建议分层：

* `gold`: 人工精标
* `silver`: 教师 + SAM + DINOv3 清洗
* `bronze`: 教师高召回但未人工确认

### D2. 学生训练策略

建议：

* 第一阶段只用 `gold + silver`
* 第二阶段再加入少量 `bronze`
* 每一轮训练后回流难例

### D3. 产物

* `best.pt`
* `last.pt`
* `metrics.json`
* `classwise_ap.csv`
* `error_cases/`

* * *

阶段 E：在线跟踪层
----------

### E1. 基础在线链路

**YOLO → 跟踪器 → 相机控制器**

推荐先用：

* `ByteTrack`：适合处理低分检测框恢复，对遮挡目标更友好。官方 README 强调其核心优势就是把低分框也纳入关联，减少漏跟与轨迹碎裂。
* 或 `BoT-SORT`：基于 ByteTrack 和 FastReID，更适合外观特征更重要的场景，但原始仓库测试环境较旧。

### E2. 为什么这里仍保留跟踪器

因为 Le-WorldModel 更适合做“动作决策”，不是直接替代经典 MOT。  
所以在线状态估计仍然建议靠：

* 检测器：YOLO
* 关联器：ByteTrack / BoT-SORT
* 外观特征：DINOv3

* * *

阶段 F：Le-WorldModel 主动控制层
------------------------

Le-WorldModel 官方代码说明它基于 stable-worldmodel 做环境管理、规划与评估，训练代码在 `jepa.py`，配置走 Hydra；安装示例使用 Python 3.10 和 `uv pip install stable-worldmodel[train,env]`。

### F1. 状态定义

输入状态建议：

* 当前帧图像或图像 latent
* 目标框 `(cx, cy, w, h)`
* 跟踪速度 `(vx, vy, vw, vh)`
* DINOv3 外观 embedding
* 当前云台状态 `(yaw, pitch, zoom)`
* 最近 N 帧历史
* 当前目标置信度 / 丢失状态

### F2. 动作定义

输出动作：

* `delta_yaw`
* `delta_pitch`
* `delta_zoom`

可离散化为：

* 左/右/上/下/放大/缩小/保持

### F3. 奖励函数

建议：

* 目标中心越接近画面中心越好
* 目标面积接近理想尺度越好
* 丢失惩罚
* 抖动惩罚
* 高频变焦惩罚
* 大动作能耗惩罚

### F4. 部署定位

Le-WorldModel 不要一开始就直连真实云台大范围探索。  
先做：

1. 离线视频回放环境
2. 仿真云台环境
3. 安全约束下的小范围实机联调

这一步是工程上的稳妥推断，主要因为它是研究型世界模型训练代码，不是现成 PTZ 相机控制 SDK。这个判断是基于仓库安装与训练说明做出的工程推断。

# 三、环境架构与数据流

## 1.双环境总架构

### a.服务器侧

做这些事：

* Open-GroundingDINO 少样本微调
* 大批量自动标注
* SAM / SAM2 精化标注
* DINOv3 特征提取、伪标签过滤、ReID 特征库更新
* YOLO 学生训练与蒸馏
* Le-WorldModel 训练、离线评估、策略导出
* 模型版本管理与数据闭环

服务器侧之所以放这些，是因为它们要么算力重，要么是低频任务。Open-GroundingDINO 仓库明确强调了训练能力；Le-WorldModel 仓库给出的安装和训练方式也是围绕训练栈、Hydra 配置和 stable-worldmodel / stable-pretraining 生态展开的。

### b.端侧（边缘侧）

做这些事：

* 摄像头视频接入
* YOLO 实时检测
* ByteTrack 跟踪
* DINOv3 轻量 ReID 或缓存特征匹配
* PID / 规则控制
* 可选：加载 Le-WorldModel 导出的轻量策略
* 向服务器回传难例、丢失片段、低置信帧

端侧这样设计的原因是，Ultralytics 明确支持导出 ONNX、TensorRT 等部署格式，适合 NVIDIA 边缘盒子做高吞吐推理；ByteTrack 的核心就是把低分检测框也纳入关联，适合你这种相机移动、目标可能暂时变小或被遮挡的场景。 

## 2.最合理的数据流

我建议你按这个闭环跑：

**服务器**  
少样本标注 → Open-GroundingDINO 微调 → 自动标注 → SAM 精化 → DINOv3 过滤 → 训练 YOLO → 导出 ONNX / TensorRT → 下发到边缘

**边缘**  
摄像头输入 → YOLO 检测 → ByteTrack 跟踪 → 控制器输出云台动作 → 记录异常片段 → 回传服务器

**再闭环**  
服务器拿边缘回传的难例继续做：

* 伪标签修正
* 学生再训练
* Le-WorldModel 策略再训练

这套闭环的最大好处是：  
**边缘永远只跑最必要的在线模块，服务器持续负责“变聪明”。**
