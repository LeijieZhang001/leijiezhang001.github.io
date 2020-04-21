---
title: '[paper_reading]-"Reconfigurable Voxels, A New Representation for LiDAR-Based Point Clouds"'
date: 2020-04-20 09:54:31
tags: ["paper reading", "Deep Learning", "autonomous driving", "Point Cloud"]
categories: Deep Learning
mathjax: true
---
　　Voxel-based 点云特征提取虽然损失了一定的信息，但是计算高效。Voxel-based 方法一个比较大的问题是，由于**点云分布的不均匀性**，作卷积时会导致可能计算的区域没有点，从而不能有效提取局部信息。为了解决栅格化后栅格中点云分布的不均匀问题，目前看到的有以下几种方法：

1. Deformable Convolution，采用可变形卷积方法，自动学习卷积核的连接范围，理论上应该能更有效得使卷积核连接到点密度较高的栅格；
2. {%post_link paper-reading-PolarNet PolarNet%} 提出了一种极坐标栅格化方式，因为点云获取的特性，这种方法获得的栅格中点数较为均匀;
3. 手动设计不同分辨率的栅格，作特征提取，然后融合。比如近处分辨率较高，远处较低的方式；
4. 本文<a href="#1" id="1ref"><sup>[1]</sup></a> 提出了一种自动选择栅格领域及分辨率，从而最大化卷积区域点数的方法；

<img src="reconfig.png" width="80%" height="80%" title="图 1. Reconfig Voxels">
　　如图 1. 所示，本文提出的 Reconfigurable Voxel 方法，能自动选择领域内点数较多的栅格特征提取，进而作卷积运算，避免点数较少，从而信息量较少的栅格作特征提取操作；此外还可根据点数自动调整分辨率以获得合适的栅格点数。通过这种方法，每个栅格输入到网络前都能有效提取周围点数较多区域的特征信息。

## 1.&ensp;Framework
<img src="pipeline.png" width="80%" height="80%" title="图 2. Framework">
　　如图 2. 所示，本文以检测任务为例，分三部分：Voxel/Pillar Feature Extraction，Backbone，RPN/Detection Head。后两个采用传统的方法，本文主要是改进 Voxel/Pillar Feature Extraction，这是输入到网络前的特征提取阶段。

## 2.&ensp;Voxel/Pillar Feature Extraction
　　传统的输入到 2D 卷积网络的特征要么是手工提取的，要么是用 {%post_link paperreading-PointPillars PointPillars%} 网络去学习每个 Voxel 的特征。由此输入到网络的特征不是最优的，因为点云的稀疏性会导致后面的 2D 卷积网络作特征提取时遇到很多“空”的 Voxel。本文提出的方法就能显式得搜索每个 Voxel 周围有点的区域作特征提取，使得之后 2D 卷积特征提取更加有效。其步骤为：

- 点云栅格化，并存储每个 Voxel 周围 Voxel 的索引；
- 每个 Voxel 周围 Voxel 作 Biased Random Walk，去搜索有更稠密点云的 Voxel；
- 将每个 Voxel 与新搜索到的周围 Voxel 作特征提取与融合，得到该 Voxel 特征；

### 2.1.&ensp;Biased Random Walking Neighbors
　　邻域 Voxel 搜索目标是：**在距离较近的情况下寻找较稠密的 Voxel**。由此设计几种策略：

- 点数越少的 Voxel，有更高概率作 Random Walk，以及更多 Step 去周围相邻的 Voxel；
- 点数越多的 Voxel，有更高概率被其它 Voxel Random Walk 到；

　　将以上策略数学化。设第 \\(j\\) 个 Voxel 有 \\(N(j)\\) 个点，最大点数为 \\(n\\)，其作 Random Walk 的概率为 \\(P _ w(j)\\)，步数 Step 为 \\(S(j)\\)，第 \\(i\\) 步到达的 Voxel 为 \\(w _ j(i)\\)，其四领域 Voxel 集合为 \\(V(w _ j(i))\\)，从该 Voxel 走到下一个 Voxel 的概率为 \\(P(w _ j(i+1)|w _ j(i))\\)。由此得到以上策略的数学描述：
$$P _ w(j)=\frac{1}{N(j)} \tag{1}$$
$$S(j)=n-N(j)\tag{2}$$
$$P\left(w _ j(i+1)|w _ j(i)\right) = \frac{N\left(w _ j(i+1)\right)}{\sum _ {v\in V(w _ j(i))}N(v)}\tag{3}$$
需要注意的是，\\(S(j)\\) 是在开始时计算的，此后每走一步就减1。
<img src="random_walk.png" width="90%" height="90%" title="图 3. Random walk">
　　如图 3. 所示，左边为单分辨率下 Voxel 搜索过程。

### 2.2.&ensp;Reconfigurable Voxels Encoder
　　每个 Voxel \\(v _ i\\) 搜索到最优的 4 领域 Voxel 集 \\(V(v _ i)\\) 后，需要融合得到该 Voxel 的特征：
$$\begin{align}
F(v _ i) &= \psi\left(f _ {v _ i}, f _ {V(v _ i)}\right)\\
&= \varphi _ 1\left[\varphi _ 2(f _ {v _ i}), \varphi _ 2\left(\sum _ {j=1}^4 W _ j(f _ {v _ i})f _ {V _ {j(v _ i)}}\right)\right] _ f
\tag{4}\end{align}$$
其中 \\(\\varphi _ 1\\) 为 low-level 操作，如 average pooling，\\(\\varphi _ 2\\) 为 high-level 操作，如 MLP。

### 2.3.&ensp;Multi-resolution Reconfigurable Voxels
　　图 3. 左边是单分辨率情况，Random Walking 可以拓展到多分辨率情形。当点云非常稀疏的时候，就很有必要降低栅格的分辨率。如图 3. 所示，\\(P _ w\\) 计算时除以 4，以维持与高分辨率的一致性；高分辨率到低分辨率搜索概率为 \\(0.25P _ w\\)，低分辨率到高分辨率搜索概率为 \\(0.5P _ w\\)。其余准则与单分辨率一致。实验结果表面多分辨率有一定提升，但是相比单分辨率提升不明显。

## 3.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Wang, Tai, Xinge Zhu, and Dahua Lin. "Reconfigurable Voxels: A New Representation for LiDAR-Based Point Clouds." arXiv preprint arXiv:2004.02724 (2020).
