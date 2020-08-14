---
title: '[paper_reading]-"Cylinder3D"'
date: 2020-08-12 11:43:33
updated: 2020-08-17 09:19:12
tags: ["paper reading", "Segmentation", "Deep Learning", "autonomous driving", "Point Cloud"]
categories:
- Semantic Segmentation
mathjax: true
---

　　Voxel-based 点云分割/检测等任务中，点云的投影表示方法有三种：

- Spherical
- Bird-eye View
- Cylinder

其中 Spherical 球坐标投影代表为 {%post_link paper-reading-RandLA-Net RandLA-Net %}；Bird-eye View 则是目前主流的方法。有关 Bird-eye View 点云处理的优劣已经说了很多了，这里不再赘述。本文<a href="#1" id="1ref"><sup>[1]</sup></a> 介绍一种 Cylinder 柱状投影的点云处理方式，类似 {%post_link paper-reading-Pillar-based-Object-Detection Pillar-based Object Detection%}，也可以认为是 {%post_link paper-reading-PolarNet PolarNet%} 的 3D 版本。
<img src="vs.png" width="70%" height="70%" title="图 1. Comparison">
　　{%post_link paper-reading-Pillar-based-Object-Detection Pillar-based Object Detection%} 中详细说明了 Cylinder 投影比 Spherical 投影的优势，这里不做赘述，如图 1. 所示，相比 Spherical 投影，Cylinder 投影效果提升很明显。

## 1.&ensp;Framework
<img src="framework.png" width="80%" height="80%" title="图 2. Framework">
　　如图 2. Cylinder3D 由 3D 柱坐标投影和 3D U-Net 特征提取两部分组成。框架比较简单，网络结构主要由 DownSample，UpSample，Asymmetry Residual Block，Dimension-Decomposition based Context Modeling 四种组件构成。

## 2.&ensp;Cylinder Partition
<img src="coord.png" width="80%" height="80%" title="图 3. Cylinder Partition">
　　如图 3. 所示，将笛卡尔坐标系下的点云 \\((x,y,z)\\) 转换到柱坐标系下 \\((\\rho,\\theta,z)\\)。对于每个扇形 Voxel，作 PointNet 特征提取，最终得到 3D Cylinder 点云特征表示 \\(\\mathbb{R}\\in C\\times H\\times W\\times L\\)。

## 3.&ensp;Network
<img src="block.png" width="70%" height="70%" title="图 4. A & DDCM">

### 3.1.&ensp;Asymmetry Residual Block
　　如图 4. 所示，Asymmetry Residual Block 将 \\(3\\times 3\\times 3\\) 卷积拆分成 \\(1\\times 3\\times 3\\) 和 \\(3\\times 1\\times 3\\) 两种，这样作有两个好处：

- 由于待检测的目标都接近于长方体，这种卷积形式更有利于提取长方体样式的特征；
- 减少 33% 的计算量，类似 Depth-wise Convolution；

该模块作为 3D 卷积的基本模块，嵌入在下采样前，以及上采样后。

### 3.2.&ensp;Dimension-Decomposition based Context Modeling
　　由于 3D 空间的特征表达是 high-rank 的，所以利用矩阵分解的思想，将其用 height，width，depth 三维的 low-rank 向量来权重化表达，由此设计如图 4. 中的 DDCM 模块。该模块将三个方向的特征计算各自的权重，然后与原始特征作权重化整合。输出的特征用于最终的预测，预测输出是 Voxel-based，维度为 \\(Class\\times H\\times W\\times L\\)。

## 4.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Zhou, Hui, et al. "Cylinder3D: An Effective 3D Framework for Driving-scene LiDAR Semantic Segmentation." arXiv preprint arXiv:2008.01550 (2020).  
