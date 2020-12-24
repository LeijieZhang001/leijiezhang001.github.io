---
title: Deep Multi-modal Object Detection and Semantic Segmentation for Autonomous Driving - Datasets, Methods, and Challenges
date: 2020-12-17 09:28:40
updated: 2020-11-23 09:34:12
tags: ["paper reading", "Segmentation", "Deep Learning", "Autonomous Driving", "Point Cloud", "3D Detection", "Multi-modal Fusion"]
categories:
- Deep Learning
- Review
mathjax: true
---

　　由于相机，激光雷达，毫米波雷达等传感器各有优劣，所以深度多模态数据融合在自动驾驶感知中非常重要。本文<a href="#1" id="1ref"><sup>[1]</sup></a>以目标检测及语义分割为例，详细阐述了深度多模态数据融合的发展及挑战。  
　　多模传感器融合的目标检测及语义分割任务，可分解为三大问题：What to Fuse，When to Fuse，How to Fuse。以下就从这三个方面进行分析归纳。

## 1.&ensp;What to Fuse
　　自动驾驶中用于全范围感知的有激光雷达，毫米波雷达，相机。{%post_link paper-reading-RadarNet RadarNet%} 中比较详细得介绍了激光雷达与毫米波雷达的优劣，并融合二者作目标检测跟踪；{%post_link paper-reading-CenterFusion CenterFusion%} 则融合毫米波雷达与相机二者的优势，作目标检测与速度测量；激光雷达与相机的融合，研究已经较多，这里不作举例。同时融合三个传感器的算法暂时没看到。  
　　激光点云的处理方法主要有三种: 1. 将点云物理空间 3D Voxel 化处理；2. 直接在点云连续空间内进行点级别的学习；3. 将点云投影到 2D 空间，如 Bird-View，Apherical-View，Cylinder-View 等，然后作 2D 卷积处理。  
　　毫米波雷达数据 \\(x,y,v\\) 可表示为 2D 特征图，然后用 2D 卷积来处理；也可表示为点云的形式，然后用点云的操作来处理。

## 2.&ensp;How to Fuse
　　考虑两个不同的传感器数据源 \\(M _ i, M _ j\\)，对应的第 \\(l\\) 层网络特征 \\(f _ l ^ {M _ i}, f _ l ^ {M _ j}\\)，以及操作 \\( G _ l(\\cdot)\\)。融合方式有以下几种：

- Addition or Average Mean:  
将两个特征图相加或者取平均，\\(f _ l=G _ {l-1}\\left(f _ {l-1} ^ {M _ i}+f _ {l-1} ^ {M _ j}\\right)\\)。
- Concatenation:  
将两个特征图在深度维度进行串联，\\(f _ l=G _ {l-1}\\left(f _ {l-1} ^ {M _ i}\\frown f _ {l-1} ^ {M _ j}\\right)\\)。
- Ensemble:  
在目标检测任务中，对 ROI 内的特征进行整合，\\(f _ l=G _ {l-1}\\left(f _ {l-1} ^ {M _ i}\\right)\\cup G _ {l-1}\\left( f _ {l-1} ^ {M _ j}\\right)\\)。
- Mixture of Experts:  
用 experts 网络预测带融合特征的权重，然后作权重融合，\\(f _ l=G _ {l}\\left(w ^ {M _ i}\\cdot f _ {l-1} ^ {M _ i}+w ^ {M _ j}\\cdot f _ {l-1} ^ {M _ j}\\right)\\)，其中 \\(w ^ {M _ i}+ w ^ {M _ j} = 1\\)。

## 3.&ensp;When to Fuse
<img src="fusion-methods.png" width="90%" height="90%" title="图 1. Fusion Methods">
　　如图 1 所示，融合的时间点可分为 early，middle，late 三种，**本文归纳发现并没有哪一种融合是最优的，这与传感器类型，数据，网络结构等相关**。设融合操作为 \\(f _ l = f _ {l-1} ^ {M _ i}\\oplus f _ {l-1} ^ {M _ j}\\)，那么各融合方式可归纳为：

- Early Fusion  
在传感器原始数据阶段进行数据融合:
$$f _ L = G _ L\left(G _ {L-1}\left(\dots G _ l\left(\dots G _ 2\left(G _ 1\left(f _ 0 ^ {M _ i}\oplus f _ 0 ^ {M _ j}\right)\right)\right)\right)\right)\tag{1}$$
前融合的优势是深度整合传感器数据信息，理论上能挖掘最全的特征信息，以及计算量较小；劣势是模型灵活性较差，以及对多模态数据的空间对齐准确度非常敏感，其空间对齐的精度受传感器之间参数标定，采样频率，传感器缺陷等因素影响。
- Late Fusion  
在网络输出后进行融合：
$$f _ L=G _ L ^ {M _ i}\left(G _ {L-1} ^ {M _ i}\left(\dots G _ 1 ^ {M _ i}(f _ 0 ^ {M _ i})\right)\right) \oplus G _ L ^ {M _ j}\left(G _ {L-1} ^ {M _ j}\left(\dots G _ 1 ^ {M _ j}(f _ 0 ^ {M _ j})\right)\right)\tag{2}$$
后融合是模块化的，所以有很强的灵活性；但是需要较多的计算资源，以及没有在特征层面对数据进行融合，可能丧失一定的信息量。

- Middle Fusion  
中融合变种非常多，如图 1. 所示，可以是 deep fusion 模式，也可以是 short-cut fusion 模式。网络结构上，还是比较难断定哪种结构是最优的。

<img src="fusion-arch.png" width="100%" height="100%" title="图 2. Fusion Archtectures">
　　对于目标检测任务来说，two-stage 方法基本都是在 ROI 内作特征融合，经典的方法如图 2. 所示，这里不做展开。

## 4.&ensp;Datasets & Methodology
<img src="challenges.png" width="100%" height="100%" title="图 3. Challenges and Open Questions">
　　如图 3. 所示，目前基于多传感器融合的感知主要挑战有：

- Multi-modal data preparation  
公开数据量及数据的多样性还较少，数据中多传感器的标定，标注准确性存疑。

- Fusion Methodology  
"What to fuse" 中融合的传感器数据还较少，还可以融合超声波雷达，V2X 信息，物理模型，先验模型等；"How to fuse" 中目前都是简单的融合，或者说整合，缺少对信息源不确定性的估计(Uncertainty)，可以采用 BNN 对不确定性进行估计；"When to fuse" 中目前基本凭经验去寻找最优的网络融合结构，缺少理论指导。

- Others  
评估指标上，还需进一步体现模型的鲁棒性；网络结构上，目前缺少时序融合。

## 5.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Feng, Di, et al. "Deep multi-modal object detection and semantic segmentation for autonomous driving: Datasets, methods, and challenges." IEEE Transactions on Intelligent Transportation Systems (2020).

