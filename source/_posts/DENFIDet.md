---
title: '[paper_reading]-"Boundary-Aware Dense Feature Indicator for Single-Stage 3D Object Detection from Point Clouds"'
date: 2020-05-22 11:27:38
tags: ["paper reading", "3D Detection", "Deep Learning", "Autonomous Driving"]
categories: 3D Detection
mathjax: true
---

　　俯视图下 Voxel-based 点云 3D 目标检测一般会使用 2D 检测网络及相关策略。但是不同于图像的 2D 目标检测，俯视图下目标的点云信息基本在边缘处，所以如何准确得捕捉目标的边缘信息对特征提取的有效性非常关键。本文<a href="#1" id="1ref"><sup>[1]</sup></a> 提出了一种能捕捉目标边缘信息的网络结构，从而作更准确的目标检测。  

## 1.&ensp;Framework
<img src="framework.png" width="80%" height="80%" title="图 1. Framework of DENFIDet">
　　如图 1. 所示，在传统的 One-Stage 2D/3D 检测框架下，嵌入了 DENFI 模块，该模块首先通过 DBPM 生成稠密的目标边缘 proposals，然后指导 DENFIConv 去提取更准确的目标特征，输出到检测头作 3D 目标框属性的分类与回归。

## 2.&ensp;DENFI(Dense Feature Indicator)
### 2.1.&ensp;DBPM(Dense Boundary Proposal Module)

<img src="DBPM.png" width="60%" height="60%" title="图 2. DBPM">
　　DBPM 的输入为 Backbone 输出的 \\(H\\times W\\times C\\) 特征图，其由分类和回归两个分支构成：

- 分类分支  
分类分支经过 \\(1\\times 1\\) 卷积输出 \\(H\\times W\\times K\\) 大小的 pixel 级别的 Score Map，其中 \\(K\\) 为类别数；
- 回归分支  
回归分支也经过 \\(1\\times 1\\) 卷积，输出 \\(H\\times W\\times (4+n\\times 2)\\) 大小的回归量。回归量包括 \\((l,t,r,b)\\) 以及角度 \\((\\theta ^{bin}, \\theta ^{res})\\)(角度回归分解成了 n 个 bin 分类与 bin 内残差回归两个问题)。最终解码为描述目标边缘的信息：\\((l,t,r,b,\\theta)\\)。

　　Loss 的计算首先得区分正负样本。正负样本的划分思想与传统的差不多，主要思想是正负样本过渡区域引入 Ignore。如图 4. 所示，设 3D 真值框属性表示为 \\((x,y,w,l,\\theta)\\)，正样本区域设计为 \\((x,y,\\sigma _ 1w,\\sigma _ 1l,\\theta)\\)，定义另一缩小框 \\((x,y,\\sigma _ 2w,\\sigma _ 2l,\\theta)\\)，其中 \\(\\sigma _ 1 < \\sigma _ 2\\)。由此可得，灰色为正样本区域，黄色为 Ignore 区域，其它为负样本区域。  
　　对于分类的 Loss，直接对正负样本进行 Focal Loss 计算。对于回归分支，则采用正样本的平均 Loss。回归 Loss 由目标框的 IoU Loss 以及角度 Loss 组成，角度 Loss 又由 bin 分类 Loss 加 bin 残差回归 Loss 组成。这里不做展开。  
　　需要注意的是，分类分支只在训练的时候计算，Inference 时候只作回归分支的计算，从而得到每个像素感知到的目标边缘的信息。

### 2.2.&ensp;DENFIConv
<img src="dconv.png" width="80%" height="80%" title="图 3. dconv">
　　如图 3. 所示，Deformable Convolution 的思想是自动去寻找感兴趣的卷积区域。DBPM 获得每个像素点的目标边缘信息以后，自然的，接下来对像素点的卷积运算，运用可变形卷积可以捕捉更准确的目标区域信息。
<img src="DSDC.png" width="60%" height="60%" title="图 4. DSDC">
　　如图 4. 所示，结合 Deformable Convolution 与 depth-wise separable Convolution，本文提出了 Depth-wise Separable Deformable Convolution。即将 \\(3\\times 3\\) 的可变形卷积拆解成 \\(3\\times 3\\) 的 Depth-wise 卷积以及 \\(1\\times 1\\) 的可变形卷积，极大减少参数量。

## 3.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Xu, Guodong, et al. "Boundary-Aware Dense Feature Indicator for Single-Stage 3D Object Detection from Point Clouds." arXiv (2020): arXiv-2004.  
