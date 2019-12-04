---
title: '[paper_reading]-"RandLA-Net"'
date: 2019-12-04 09:08:09
tags: ["paper reading", "Segmentation", "Deep Learning", "autonomous driving", "Point Cloud"]
categories: Semantic Segmentation
mathjax: true
---

　　不同与点云 3D 检测，可以 Voxel 化牺牲一定的分辨率，点云语义分割则要求点级别的分辨率，所以栅格化做点云分割信息会有一定的损失。但是直接对所有点进行特征提取，计算量又相当巨大，为了平衡效率与性能，一般也不得不对点云进行采样处理。这种点云级别的处理方式有 {% post_link PointNet-系列论文详读 PointNet++%}， {% post_link paperreading-FlowNet3D FlowNet3D%} 等。
<img src="arch2.png" width="90%" height="90%" title="图 1. RandLA-Net">
　　本文<a href="#1" id="1ref"><sup>[1]</sup></a>提出的方法主要为了解决大尺度点云集下，如何高效提取点云局部特征的问题。针对大尺度点云集，作者对比了不同采样算法，得出随机采样最简单高效的结论；针对随机采样丟失信息的问题，以及为了提高局部特征提取能力，本文提出了局部特征聚合(Local Feature Aggregation)模块，该模块包含 Local Spatial Encoding，Attentive Pooling，以及 Dilated Residual Block。  
　　如图 1. 所示，LFA 作为基本模块用于特征提取，下采样采用随机采用，上采样过程类似图像中的 dconv，包含向上插值以及 MLP 过程。

## 1.&ensp;Sampling
　　关于点云采样，在 {% post_link paperreading-FlowNet3D FlowNet3D%} 中有简单介绍。本文将采样算法分为两大类：

- Heuristic Sampling  
    1. Farthest Point Sampling(FPS)， {% post_link paperreading-FlowNet3D FlowNet3D%} 中有介绍，是一种均匀采样方法。其算法复杂度为 \\(\\mathcal{O}(N^2)\\)。
    2. Inverse Density Importance Sampling(IDIS)，计算每个点的密度属性，根据属性选取 K 个点，其复杂度为 \\(\\mathcal{O}(N)\\)。
    3. Random Sampling(RS)，随机采样，复杂度为 \\(\\mathcal{O}(1)\\)。

- Learning-based Sampling  
...

　　本文作者认为随机采样复杂度最低，其它采样复杂度太高。我认为也不能这么说，在一定策略及加速下，其它采样算法效率也可以很高。比如栅格化后在采样，可以高效的并行加速，并且使得稀疏区域保留更多信息。

## 2.&ensp;Local Feature Aggregation
<img src="arch1.png" width="90%" height="90%" title="图 2. RandLA-Net">
　　特征提取非常关键，尤其在本文采用随机采样后，稀疏区域信息丢失比较严重的情况下。如图 2. 所示，本文提出了局部特征聚合(Local Feature Aggregation)模块，包含 Local Spatial Encoding，Attentive Pooling，以及 Dilated Residual Block。

### 2.1.&ensp;Local Spatial Encoding
　　在原始点云中提取每个点的局部特征，类似 {% post_link paperreading-FlowNet3D FlowNet3D%}(PointNet++) 中的 set conv 层，这里多了手工特征信息，其步骤为：

1. 针对每个点 \\(p_i\\)，用 KNN 找到与其最近的 K 个点: \\(\\{p _ i^1,...p _ i^k,...p _ i^K\\}\\)；
2. 针对最近邻的每个点 \\(p_i^k\\)，设计其相对位置的特征：
$$ \mathrm{r}_i^k = \mathrm{MLP}\left(p_i\oplus p_i^k\oplus (p_i-p_i^k)\oplus ||p_i-p_i^k||\right) \tag{1}$$
3. 针对最近领的每个点 \\(p_i^k\\)，其本来的特征为 \\(\\mathrm{f}_i^k\\)，叠加相对位置特征 \\(\\mathrm{r}_i^k\\) 后得到每个点的特征为 \\(\\mathrm{\\hat{f}}_i^k\\)。由此最近领点集的特征为： \\(\\mathrm{\\hat{F}}_i=\\{\\hat{\\mathrm{f}}_i^1,...\\hat{\\mathrm{f}}_i^k,...\\hat{\\mathrm{f}}_i^K\\}\\)。

### 2.2.&ensp;Attentive Pooling
　　该模块的作用是聚合 \\(p_i\\) 的最近邻点集特征 \\(\\hat{\\mathrm{F}}_i\\)。PointNet 的 SA 层(FlowNet3D 中的 set conv 层)直接用 Max/Mean 这种对称函数聚合，本文采用一种更有效的基于注意力机制的 pooling 方式，其步骤为：

1. 计算注意力分数，对每个特征设计分数计算方式为：
$$ \mathrm{s}_i^k = \mathrm{g}\left(\hat{\mathrm{f}}_i^k, W\right) \tag{2}$$
其中 \\(\\mathrm{g}\\) 表示一个感知机 MLP(W 为其权重) 以及一个 softmax 函数。
2. 聚合，根据注意力分数，权重求和，得到 \\(p_i\\) 点的特征：
$$ \bar{\mathrm{f}}_i = \sum_{k=1}^K \left(\hat{\mathrm{f}}_i^k \cdot \mathrm{s}_i^k \right) \tag{3}$$

### 2.3.&ensp; Dilated Residual Block
<img src="LA.png" width="60%" height="60%" title="图 3. LA Module">
　　如图 2. 及 3. 所示，连续堆叠多个 LA 模块，能起到增加感受野的效果，然后引入 residual 思想，图 2. 下图就构成了一个 LFA 的基础模块。

<a id="1" href="#1ref">[1]</a> Hu, Qingyong, et al. "RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds." arXiv preprint arXiv:1911.11236 (2019).
