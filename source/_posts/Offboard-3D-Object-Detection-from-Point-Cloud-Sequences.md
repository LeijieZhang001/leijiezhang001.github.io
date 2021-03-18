---
title: Offboard 3D Object Detection from Point Cloud Sequences
date: 2021-03-12 09:29:53
updated: 2021-03-18 09:34:12
tags: ["paper reading", "3D Detection", "Deep Learning", "Autonomous Driving", "Point Cloud"]
categories:
- 3D Detection
- Offline
mathjax: true
---

　　目前点云检测等任务基本集中在在线实时情况下的研究，然而在离线场景下，自动化/半自动化标注/高精地图语义信息提取/数据闭环中的教师模型等，这些任务也相当重要。相比在线模式，离线模式对点云的实时处理要求较低，而且能更容易提取时序信息。本文<a href="#1" id="1ref"><sup>[1]</sup></a>提出了一种基于时序的离线点云检测方法，能极大提高目标真值标注的自动化程度。其性能几乎达到了人工标注的水平。

## 1.&ensp;Problem Statement & Framework
　　对于一个 \\(N\\) 帧序列点云 \\(\\{\\mathcal{P} _ i\\in\\mathbf{R} ^ {n _ i\\times C}\\},i=1,2,...,N\\)，已知每帧点云传感器在世界坐标系下的位姿 \\(\\{\\mathcal{M} _ i=[R _ i|t _ i]\\in\\mathbf{R} ^ {3\\times 4}\\},i=1,2,...,N\\)，那么我们要得到每帧点云中的目标 3D 属性(包括中心点，尺寸，朝向)，类别，ID。  
<img src="framework.png" width="90%" height="90%" title="图 1. Framework">
　　方法框架如图 1. 所示，输入序列点云，首先用目标检测器检测每一帧中的 3D 目标，然后用跟踪器将目标进行数据关联作 ID 标记。最后提取每个 ID 目标跟踪轨迹上的所有点云及目标框信息，作进一步的 3D 目标框精修预测。

## 2.&ensp;3D Auto Labeling Pipeline

### 2.1.&ensp;Multi-frame 3D Object Detection
　　采用 {%post_link paperreading-End-to-End-Multi-View-Fusion-for-3D-Object-Detection-in-LiDAR-Point-Clouds End-to-End-Multi-View-Fusion-for-3D-Object-Detection-in-LiDAR-Point-Clouds%} 中的 MVF 检测器，输入用多帧经过运动补偿的点云代替，每个点增加相对时间的偏移量。在测试阶段，采用 Test-time Augmentation，将点云绕 Z 轴进行不同角度的增广，最终的检测框进行权重融合。这在离线计算中可用不同计算单元并行化实现。
<img src="mvf.png" width="90%" height="90%" title="图 2. MVF++">
　　MVF 基本思想可见 {%post_link paperreading-End-to-End-Multi-View-Fusion-for-3D-Object-Detection-in-LiDAR-Point-Clouds End-to-End-Multi-View-Fusion-for-3D-Object-Detection-in-LiDAR-Point-Clouds%}。这里对 MVF++ 作更详细的描述。结构如图 2. 所示，网络输入的点云尺寸为 \\(N\\times C\\)，经过 MLP 网络映射到高维特征，然后在 Birds Eye View 和 Perspective View 下体素化并作卷积提取特征的操作，最终融合得到点级别的三种特征类型。Loss 构成为：
$$L = L _ {cls} + w _ 1L _ {centerness} + w _ 2L _ {reg} + w _ 3L _ {seg}\tag{1}$$
其中 \\(L _ {seg}\\) 是区分前景，背景的辅助分支。

### 2.2.&ensp;Multi-object Tracking
　　多目标跟踪方法采用<a href="#2" id="1ref">[2]</a>，在三维空间下作前后帧率检测跟踪的的数据关联，然后用卡尔曼作状态估计。

### 2.3.&ensp;Object Track Data Extraction
　　经过多目标跟踪模块后，每个目标实例在时空内都作了 ID 标记。在世界坐标系下，可提取目标的 4D 时空信息，包括点云及 3D 属性框：第 \\(j\\) 个目标在其出现的帧 \\(S _ j\\) 下的点云 \\(\\{\\mathcal{P} _ {j,k}\\},k\\in S _ j\\)，以及对应的 3D 框 \\(\\{\\mathcal{B} _ {j,k}\\},k\\in S _ j\\)。

### 2.4.&ensp;Object-centric Auto  Labeling
　　有了目标检测，跟踪算法流程后。接下来将目标自动标注分为目标动静状态分析，静态目标自动标注以及动态目标自动标注。  
　　目标动静状态分析模块将每个目标实例提取 4D 特征，然后加入线性分类器来预测。特征包括目标框中心点的时空方差，以及目标从始至终的中心点偏移距离。真值标记时，将静态目标从始至终的距离阈值设定为 1.0m，最大速度不超过 1m/s。这种方式动静预测准确率高达 99%。此外，将行人都归为动态目标。

#### 2.4.1.&ensp;Static Object
<img src="static.png" width="70%" height="70%" title="图 3. Static Object">
　　如图 3. 所示，首先挑选 score 最高的目标框作为 initial box，将点云从世界坐标系转到该目标框坐标系。类似 Cascade-RCNN，作连续的前景分割-目标框回归的网络预测。Loss 项由中心点回归，朝向分类回归，尺寸分类回归三部分构成：
$$\begin{align}
L &= L _ {seg} + w\sum _ i ^ 2 L _ {box _ i} \\
&= L _ {seg} +  w\sum _ i ^ 2\left(L _ {center-reg _ i} + w _ 1L _ {size-cls _ i} + w _ 2L _ {size-reg _ i} + w _ 3L _ {heading-cls _ i} + w _ 4L _ {heading-reg _ i}\right)
\end{align}\tag{2}$$

#### 2.4.2.&ensp;Dynamic Object
<img src="dynamic.png" width="70%" height="70%" title="图 4. Dynamic Object">
　　如图 4. 所示，对于动态目标，也可以将目标点云累积到某一时刻的目标中心坐标系中，但是累积很难对齐(目测可以用迭代的思想，定位-累积来精修)。所以本文采用从轨迹中直接提取特征，从而预测目标尺寸的方法。对于 \\(T\\) 时刻前后的目标点云 \\(\\{\\mathcal{P} _ {j,k}\\} _ {k=T-r} ^ {T+r}\\) 以及目标框 \\(\\{\\mathcal{B} _ {j,k}\\} _ {k=T-s} ^ {T+s}\\)，特征由两部分构成：

- Point 分支。将点云增加时间信息后，转换到当前目标框 \\(\\mathcal{B} _ {j,T}\\) 的中心位置坐标系下，用 PointNet 网络作前后景的分割，得到前景目标的全局特征量。
- Box 分支。同样转换到当前目标框中心坐标系下。需要注意的是，取的 Box 帧数大多比 Point 长(Point 只取 5 帧)。目标框特征维度为 8(x,y,z,l,h,w,ry,time)，同样通过 PointNet 网络提取全局特征。

将这两个特征作融合，然后预测当前 \\(T\\) 时刻的目标尺寸及朝向。  
　　为了增强两个分支各自对预测目标的学习，分别对两个分支用真值目标作监督学习。最终的 Loss 为：
$$L = L _ {seg}+v _ 1L _ {box-traj} + v _ 2L _ {box-obj-pc}+v _ 3L _ {box-joint} \tag{3}$$

## 3.&ensp;Experiments

### 3.1.&ensp;Comparing with SOTA Detectors
<img src="sota.png" width="90%" height="90%" title="图 5. sota">
　　如图 5. 所示，毫无疑问，效果拔群。

### 3.2.&ensp;Comparing with Human Labels
<img src="human.png" width="65%" height="65%" title="图 6. Human labels">
　　如图 6. 所示，挑选 3 个经验丰富的标注员对数据进行标注，以此与本文方法进行比较，可见本方法只在高度估计上无法与人类标注员媲美，定位及其它尺寸上，几乎能达到类似水平。
<img src="human2.png" width="65%" height="65%" title="图 7. Consistency between human lablers">
　　进一步思考，人类不同的标注员，他们的标注结果一致性如何？如图 7. 所示，GT 由多个标注员交叉验证得到，以此为基准，发现人类标注员也很难达到与真值相同水平的程度，标注质量几乎与本方法差不多，当距离较远时，点数会较少，人类标注员反而标注质量会下降。

### 3.3.&ensp;Applications to Semi-supervised Learning
<img src="semi-supervised.png" width="65%" height="65%" title="图 8. Semi-supervised">
　　用本文的方法去标注未标注的数据，作为实时目标检测模型的训练数据，能极大提升其性能。也从另一方面论证了本文方法能比拟人工标注水平。

### 3.4.&ensp;Analysis of the Multi-frame Detector
<img src="mvf-exp.png" width="65%" height="65%" title="图 9. MVF Experiments">
　　如图 9. 所示，MVF 检测器性能实验，当输入帧数大于 5 帧时，帧数增多已经无法提升检测性能。

## 4.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Qi, Charles R., et al. "Offboard 3D Object Detection from Point Cloud Sequences." arXiv preprint arXiv:2103.05073 (2021).

