---
title: '[paper_reading]-"End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection"'
date: 2020-06-22 09:19:12
updated: 2020-06-25 09:19:12
tags: ["paper reading", "Deep Learning", "Autonomous Driving", "Point Cloud", "3D Detection"]
categories:
- 3D Detection
mathjax: true
---

　　基于视觉的 3D 目标检测方法因为成本较低，所以在 ADAS 领域应用非常广泛。其基本思路有以下几种：

- 单目\\(\\rightarrow\\)3D 框，代表文章有<a href="#1" id="1ref">[1]</a>。
- 单目\\(\\rightarrow\\)深度图\\(\\rightarrow\\)3D 框
- 双目\\(\\rightarrow\\)3D 框，代表文章有<a href="#2" id="2ref">[2]</a>。
- 双目\\(\\rightarrow\\)深度图\\(\\rightarrow\\)3D 框

由单目或双目直接回归 3D 目标框的属性，这种方法优势是 latency 小，缺点则是，没有显式的预测深度图，导致目标 3D 位置回归较为困难。而在深度图基础上回归 3D 目标位置则相对容易些，这种方法由两个模块构成：深度图预测，3D 目标预测。得到深度图后，可以在前视图下将深度图直接 concate 到 rgb 图上来做，另一种方法是将深度图转换为 pseudo-LiDAR 点云，然后用基于点云的 3D 目标检测方法来做，目前学术界基本有结论：pseudo-LiDAR 效果更好。  
　　本文<a href="#3" id="3ref"><sup>[3]</sup></a>即采用双目出深度图，然后基于 pseudo-LiDAR 来作 3D 目标检测的方案，并且解决了两个模块需要两个网络来优化的大 lantency 问题，实现了 End-to-End 联合优化的方式。

## 1.&ensp;Framework
<img src="framework.png" width="60%" height="60%" title="图 1. Framework">
　　基于点云作 3D 目标检测大致可分为 point-based 与 voxel-based 两大类，详见 {%post_link Point-based-3D-Det Point-based 3D Detection%}，传统的基于双目的 pseudo-LiDAR 方案无法 End-to-End 作俯视图下 voxel-based 3D 检测，因为点云信息需要作俯视图离散化，离散的过程是无法作反向传播训练的，本文提出了 Change of Representation(CoR) 模块有效解决了这个问题。如图 1. 所示，本方案中 Depth Estimation 可由任何深度估计网络实现，然后经过 CoR 模块，将深度图变换成点云形式用于 point-based 3D detection，或者是 Voxel 形式用于 voxel-based 3D detection。这里的关键是可求导的 CoR 模块设计。

## 2.&ensp;CoR
### 2.1.&ensp;Quantization
　　点云检测模块如果采用 voxel-based 方案，那么点云到俯视图栅格的离散化(quantization)是必不可少的。假设点云 \\(P = \\{p _ 1,...,p _ N\\}\\)，待生成的 3D 占据栅格(最简单的特征形式) \\(T\\) 包含 \\(M\\) 个 bins，即 \\(m\\in\\{1,...,M\\}\\)，每个 bin 的中心点设为 \\(\\hat{p} _ m\\)。那么生成的 \\(T\\) 可表示为：
$$ T(m) = \left\{\begin{array}{l}
1, & \mathrm{if}\;\exists p\in P \; \mathrm{s.t.}\; m = \mathop{\arg\min}\limits _ {m '}\Vert p - \hat{p} _ {m'}\Vert _ 2 \\
0, & \mathrm{otherwise}.
\end{array}\tag{1}\right.$$
即如果有点落在该 bin 里，那么该 bin 对应的值置为 1。这种离散方式是无法求导的。  
　　本文提出了一种可导的软量化模块(soft quantization module)，即用 RBF 作权重计数，另一种角度来看，**这其实类似于点的空间概率密度表示**。设 \\(P _ m\\) 为落在 bin \\(m\\) 的点集：
$$ P _ m=\left\{p\in |, \mathrm{s.t.}\; m=\mathop{\arg\min}\limits _ {m '}\Vert p - \hat{p} _ {m'}\Vert _ 2\right\} \tag{2}$$
那么，\\(m'\\) 作用于 \\(m\\) 的值为：
$$ T(m, m') = \left\{\begin{array}{l}
0 & \mathrm{if}\; \vert P _ {m'}\vert = 0;\\
\frac{1}{\vert P _ {m'}\vert} \sum _ {p\in P _ {m'}} e^{-\frac{\Vert p-\hat{p} _ m\Vert ^2}{\sigma ^ 2}} & \mathrm{if}\; \vert P _ {m'}\vert > 0.
\end{array}\tag{3}\right.$$
最终的 bin 值为：
$$ T(m) = T(m,m)+\frac{1}{\vert \mathcal{N} _ m\vert}\sum _ {m'\in\mathcal{N} _ m}T(m,m') \tag{4}$$
当 \\(\\sigma ^2\\gg 0\\) 以及 \\(\\mathcal{N} _ m=\\varnothing\\) 时，回退到式 (1) 的离散化方式。本文实验中采用 \\(\\sigma ^2 = 0.01\\)，\\(\\mathcal{N} _ m=3\\times 3\\times 3 -1 = 26\\)。传统的点云栅格概率密度计算方式为：将点云中的每个点高斯化，然后统计每个栅格中心坐标上覆盖到的值。与上述方法的高斯原点不一样，但是计算结果是一致的。
<img src="quantization.png" width="90%" height="90%" title="图 2. Quantization">
　　这种方法可将导数反向传播到 \\(m'\\) 中的每个点：\\(\\frac{\\partial\\mathcal{L} _ {det}}{\\partial T(m)}\\times\\frac{\\partial T(m)}{\\partial T(m,m')}\\times\\bigtriangledown _ pT(m,m')\\)。如图 2. 所示，蓝色 voxel 表示梯度为正，即 \\(\\frac{\\partial\\mathcal{L} _ {det}}{\\partial T(m)} > 0\\)，红色 voxel 表示梯度为负。那么蓝色 voxel 期望没有点，所以将点往外推，红色 voxel 则将点往里拉，最终使点云与 LiDAR 点云，即 GT 点云一致。

### 2.1.&ensp;Subsampling
　　点云检测模块如果采用 point-based 方案，那么就比较容易直接与深度图网络进行 End-to-End 整合。point-based 3D Detection 一般通过 sampling 来扩大感受野，提取局部信息，因为这种方法的计算量对点数比较敏感，所以 sampling 也是降低计算量的有效手段。一个 \\(640\\times 480\\) 的深度图所包含的点云超过 30 万，远远超过一个 64 线的激光雷达，所以对其进行采样就非常关键。  
　　本文对深度图点云进行模拟雷达式的采样，即定义球坐标系下栅格化参数：\\((r,\\theta,\\phi)\\)。其中 \\(\\theta\\) 为水平分辨率，\\(\\phi\\) 为垂直分辨率。对每个栅格内采样一个点，即可得到一个较为稀疏，且接近激光雷达扫描属性的点云。

## 3.&ensp;Loss
　　Loss 由 depth 估计与 3D Detection 两项构成：
$$\mathcal{L} = \lambda _ {det}\mathcal{L} _ {det} + \lambda _ {depth}\mathcal{L} _ {depth} \tag{5}$$

## 4.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Mousavian, Arsalan, et al. "3d bounding box estimation using deep learning and geometry." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.  
<a id="2" href="#2ref">[2]</a> Li, Peiliang, Xiaozhi Chen, and Shaojie Shen. "Stereo r-cnn based 3d object detection for autonomous driving." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.  
<a id="3" href="#3ref">[3]</a> Qian, Rui, et al. "End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.  
