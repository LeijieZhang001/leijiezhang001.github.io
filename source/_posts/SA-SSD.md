---
title: '[paper_reading]-"SA-SSD: Structure Aware Single-stage 3D Object Detection from Point Cloud"'
date: 2020-05-22 11:27:38
tags: ["paper reading", "3D Detection", "Deep Learning", "Autonomous Driving"]
categories: 3D Detection
mathjax: true
---

　　Voxel-based 3D Detection 相比 {% post_link Point-based-3D-Det Point-based 3D Detection %} 的缺点是特征提取不仅在 Voxel 阶段损失了一定的点云信息，而且 Voxel 化后丢失了点云之间的拓扑关系。{% post_link Point-based-3D-Det Point-based 3D Detection %} 中详细描述了几种 Point-based 方法，这种方法目前比较棘手的地方是，即使作 Inference 时，也需要作 kd-tree 搜索与采样等运算量较大的操作。那么如何榨干 Voxel-based 的性能对工业界落地就显得比较重要了，本文<a href="#1" id="1ref"><sup>[1]</sup></a>提出了一种单阶段的 Voxel-based 3D 检测方法，并借助了 Point 级别特征提取的相关策略，使得检测性能有较大提升。

## 1.&ensp;Framework
<img src="framework.png" width="100%" height="100%" title="图 1. Framework of SA-SSD">
　　如图 1. 所示，SA-SSD 由三部分组成：Backbone，Detection Head，Auxiliary Network。  
　　Backbone 的输入是栅格化后的点云表示方式，文中栅格大小设定为 \\(0.05m,0.05m,0.1m\\)。Backbone 由一系列的 3D convolution 组成，因为需要保留空间三维位置信息，作 Voxel-to-Point 的映射。这里如果用 2D convolution 代替，那么 Auxiliary Network 估计也只能作 BridView 的分割了。  
　　Detection Head 主体就是传统 Anchor-Free 结构，一个分支用于预测每个特征层像素点的 Confidence，另一个分支用于预测基于每个特征层像素点的 BBox 属性，如，以该点为 "Anchor" 的四个顶点坐标。此外，为了消除 One-Stage 方法中目标框与置信度不对齐的问题，本文引入 Part-sensitive Warping 来实现与 PSRoiAlign 类似的作用，实现两者的对齐。  
　　Auxiliary Network 只在训练的阶段起作用，Inference 阶段不需要计算。该模块的作用是训练时通过 Voxel-to-Point 特征映射来反向传播监督 Backbone 中的 Voxel 特征学习 Point 级别的特征，包括点云的空间拓扑关系。**当然 Inference 时也可以保留该分割模块，那么还可以增加点级别的特征反映射到 Voxel 的模块(Point-to-Voxel)，进一步作特征增强。**

## 2.&ensp;Detachable Auxiliary Network
<img src="sa.png" width="60%" height="60%" title="图 2. Structured Aware Feature Learning">
　　如图 2. 所示，随着 Backbone 特征提取的感受野增大(特征分辨率下降)，背景点会接近目标的边缘，使得目标框大小不容易预测准确。本文提出的 Auxiliary Network，通过增加点级别分割及目标中心坐标预测任务，来监督 Backbone 特征层捕捉这种结构信息，从而达到更准确的目标检测的目的。  
　　Auxiliary Network 的输入来自 Backbone 各个分辨率的特征层。将特征层上不为零的特征点，通过 Voxel-to-Point 反栅格化映射到三维空间，设该特征点表示为 \\(\\{(f _ j,p _ j):j=1,...,M\\}\\)，其中 \\(f\\) 为特征向量，\\(p\\) 为坐标向量。有了栅格对应的伪三维坐标点下的特征表示后，即可插值出实际点云中每个点的特征向量。设点云中点的插值特征为：\\(\\{(\\hat{f} _ i,p _ i):i=1,...,N\\}\\)，采用 Inverse Distance Weighted 方法进行插值：
$$ \hat{f} _ i = \frac{\sum _ {j=1}^Mw _ j(p _ i)f _ j}{\sum _ {j=1}^Mw _ j(p _ i)} \tag{1}$$
其中：
$$w _ j(p _ i)=\left\{\begin{array}{l}
\frac{1}{\Vert p _ i-p _ j\Vert _ 2} & \mathrm{if} p _ j\in\mathcal{N}(p _ i)\\
0 & \mathrm{otherwise}
\end{array}\tag{2}\right.$$
\\(\\mathcal{N}(p _ i)\\) 为球状区域，本文在四个分辨率下分别设定为：0.05m，0.1m，0.2m，0.4m。然后通过 cross-stage link 对各个分辨率下的点特征进行 concatenate 融合。最后通过感知机进行点云分割及目标中心点预测任务的构建。  
　　对于点级别前景分割的任务，经过 sigmoid 函数后，应用二分类的 Focal Loss：
$$ \mathcal{L} _ {seg} = \frac{1}{N _ {pos}}\sum _ i^N -\alpha(1-\hat{s} _ i)^{\gamma}\mathrm{log}(\hat{s} _ i) \tag{3}$$
该分割任务使得目标检测的框更加准确，如图 2.c 所示。但是还得优化其尺度与形状。  
　　中心点的预测任务则能有效约束目标框的尺度与形状，具体的，预测的是每个属于目标的点云与中心点的相对位置(残差)。可用 Smooth-l1 来构建预测的中心点与实际中心点的 Loss。  

## 3.&ensp;Part-sensitive Warping
<img src="psw.png" width="60%" height="60%" title="图 3. Part-sensitive Warping">
　　One-Stage 方法都会有 Confidence 和 BBox 错位的现象，本文提出一种类似 PSROIAlign 但更有高效的 PSW 方法，具体步骤为：

1. 对于分类分支，修改为 \\(K\\) 个 Part-sensitive 的 cls maps，每个 map 包含目标的部分信息，比如当 \\(K=4\\) 时，可以理解为将目标切分为 \\(2\\times 2\\) 部分；
2. 对于回归分支，将每个目标框的 Feature map 划分为 \\(K\\) 个子区域，每个区域的中心点作为采样点；
3. 如图 3. 所示，通过采样得到最终 cls map 的平均值。

## 4.&ensp;Experiment
<img src="ablation.png" width="60%" height="60%" title="图 4. Ablation Study">
　　如图 4. 所示，Auxiliary Network 能有效提升网络的定位精度，PSWarp 也能有效消除 Confidence 与 BBox 的错位影响。

## 5.&ensp;Reference
<a id="1" href="#1ref">[1]</a> henhang, et al. "Structure Aware Single-stage 3D Object Detection from Point Cloud."  

