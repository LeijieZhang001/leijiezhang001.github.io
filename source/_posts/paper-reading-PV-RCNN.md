---
title: '[paper_reading]-"PV-RCNN"'
date: 2020-08-19 09:24:10
updated: 2020-08-20 09:19:12
tags: ["paper reading", "3D Detection", "Deep Learning", "autonomous driving", "Point Cloud"]
categories:
- 3D Detection
mathjax: true
---
　　PV-RCNN<a href="#1" id="1ref"><sup>[1]</sup></a> 目前在 Waymo 数据集上排名第二，性能还是比较强悍的，顺便也看了下港中文多媒体实验室开源的 OpenPCDet<a href="#2" id="2ref"><sup>[2]</sup></a> 代码，收获还是蛮多，与图像点云通用的 mmdetection3d<a href="#3" id="3ref"><sup>[3]</sup></a> 各有优劣吧。虽然 PV-RCNN 对于实际应用还是略显复杂，以及超参数较麻烦，但是其相关思想还是非常值得借鉴。本文主要关注其 Point + Vexol 特征提取并融合的方式。

## 1.&ensp;Framework
<img src="framework.png" width="90%" height="90%" title="图 1. Framework">
　　如图 1. 所示，PV-RCNN 首先将原始点云体素化，然后用 3D Sparse Convolution({%post_link Rethinking-of-Sparse-3D-Convolution Rethinking of Sparse 3D Convolution%}) 作 Voxel-level 的特征提取，并预测俯视图下的目标 ROI Proposal；另一方面，在原始点云中用 FPS 采样出特定数量的 key-point，然后通过 Voxel Set Abstraction 模块，将提取到的多尺度的 Voxel-level 特征融合到 key-point 特征；最后用 ROI-Grid 模块将 key-point 特征融合到 ROI Grid-Point 中，作进一步的 3D 目标框属性精细化预测。  
　　由此不仅利用了 Voxel-level 3D Sparse Convolution 的高效性，还利用了 Point-based 模型对局部信息提取更加精细有效的特性。总体上，PV-RCNN 框架中特征提取操作由两大块组成：1. Voxel-to-Keypoint Scene Encoding；2. Point-to-Grid RoI Feature Abstraction。

## 2.&ensp;Voxel-to-Keypoint Scene Encoding
　　该模块的作用是将提取的特征用特定数量的 Keypoint 来表示。所以有 Keypoints Sampling，Voxel Set Abstraction Module，Extended VSA Module，Predicted Keypoint Weighting 等组成。

### 2.1.&ensp;Keypoints Sampling
　　因为期望采样的 Keypoints 能完全覆盖整个场景，所以采用 Furthest-Point-Sampling(FPS) 来采样 \\(n\\) 个点 \\(\\mathcal{K}=\\{p _ 1,...,p _ n\\}\\)。对于 KITTI 数据集，取 \\(n=2048\\)，对于 Waymo 数据集，取 \\(n=4096\\)。

### 2.2.&ensp;Voxel Set Abstraction Module
　　VSA 模块将经过 3D Sparse Convolution 得到的多尺度的 Voxel 特征编码为 Keypoints 表达形式，与 {%post_link PointNet-系列论文详读 PointNet-系列论文详读%} 类似，只不过这里点周围不是点，而是 Voxel。  
　　具体的，设尺度 \\(k\\) 的 3D voxel 特征集合为 \\(\\mathcal{F}^ {(l _ k)}=\\{f _ 1 ^ {(l _ k)},...,f _ {N _ k}^{(l _ k)}\\}\\)，对应的 Voxel 3D 坐标为 \\(\\mathcal{V}^ {(l _ k)}=\\{v _ 1 ^ {(l _ k)},...,v _ {N _ k}^{(l _ k)}\\}\\)，其中 \\(N _ k\\) 为非零 Voxel 数量。对于 keypoint \\(p _ i\\)，在半径 \\(r _ k\\) 内找到所有 voxel 的特征向量：
$$S _ i ^ {(l _ k)} = \left\{\left[f _ j^{(l _ k)};v _ j^{(l _ k)}-p _ i\right] ^ T 
\left\vert\begin{array}{l}
\Vert v _ j^{(l _ k)}-p _ i\Vert ^ 2 < r _ k,\\
\forall v _ j^{(l _ k)} \in \mathcal{V} ^ {(l _ k)},\\
\forall f _ j ^ {(l _ k)} \in \mathcal{F} ^ {(l _ k)}
\end{array}\right.
\right\}\tag{1}$$
其中 \\(v _ j^{(l _ k)}-p _ i\\) 为对应的 Voxel 与该点 \\(p _ i\\) 的相对位置，与特征向量串联得到该 Voxel 在该点的投影特征。由此用 PointNet 方式可得到该 Keypoint 融合领域内 Voxel 特征集 \\(S _ i ^ {(l _ k)}\\) 后的特征：
$$f _ i ^ {(pv _ k)}=\mathrm{max}\left\{G\left(\mathcal{M}(S _ i^{(l _ k)})\right)\right\}\tag{2}$$
其中 \\(\\mathcal{M}\\) 为随机采样 Voxel 的操作，目的是为了减少计算量；\\(G\\) 为 MLP 网络。  
　　本文采用了 4 个尺度的 Voxel 特征，每个尺度的领域半径 \\(r _ k\\) 根据感受野而变化。最终得到的多尺度的语义的 Keypoint 特征为：
$$f _ i^{(pv)} = \left[f _ i^{(pv _ 1)},f _ i^{(pv _ 2)},f _ i^{(pv _ 3)},f _ i^{(pv _ 4)}\right],\;\mathrm{for}\; i = 1,...,n\tag{3}$$

### 2.3.&ensp;Extended VSA Module
　　此外，Keypoint 还利用了原始点云的特征以及经过 3D Sparse Convolution 和 ToBEV 后的 2D BEV 特征。Keypoint 通过双线性插值的方式从 BEV 特征层中计算对应空间位置的特征。综上，Keypoint 特征为：
$$f _ i^{(p)} = \left[f _ i^{(pv)},f _ i^{(raw)},f _ i^{(bev)}\right],\;\mathrm{for}\; i = 1,...,n\tag{4}$$

### 2.4.&ensp;Predicted Keypoint Weighting
<img src="weight.png" width="50%" height="50%" title="图 2. PKW">
　　通过 FPS 采样得到的 Keypoint 也包含了大量的背景点，所以需要弱化背景点的特征，强化前景点特征，以便之后前景目标框属性的精细化估计。如图 2. 所示，训练阶段，对 Keypoint 进行点级别前景背景分类，标签可由目标框内点自动生成；测试阶段，直接预测点的类别，然后作点特征的权重化整合：
$$\tilde{f _ i} ^ {(p)} = \mathcal{A}(f _ i^{(p)})\cdot f _ i^{(p)}\tag{5}$$
其中 \\(\\mathcal{A}(\\cdot)\\) 为 MLP 网络。

## 3.&ensp;Point-to-Grid RoI Feature Abstraction
<img src="roi.png" width="50%" height="50%" title="图 3. RoI-grid Pooling">
　　得到了 Keypoint 特征 \\(\\mathcal{\\tilde{F}}=\\{\\tilde{f} _ 1^{(p)},...,\\tilde{f} _ n^{(p)}\\}\\) 以及俯视图下的 3D proposal ROI 后，可进一步提取 ROI 特征，以作 3D 目标框属性的精细化估计。如图 3. 所示，类似 Voxel-to-Point 的 Voxel Set Abstraction 模块，本文提出了 Point-to-Grid 的 Set Abstraction，称之为 ROI-Grid Pooling。具体的，对每个 ROI Proposal，采样 \\(6\\times 6\\times 6\\) 个栅格点：\\(\\mathcal{G}=\\{g _ 1,...,g _ {216}\\}\\)。Set Abstraction 操作将 Keypoint 的特征映射到栅格点处。类似 VSA，首先在 \\(\\tilde{r}\\) 半径内查找栅格点的周围 Keypoint：
$$\tilde{\psi} = \left\{\left[\tilde{f} _ j^{(p)};p _ j-g _ i\right] ^ T 
\left\vert\begin{array}{l}
\Vert p _ j-g _ i\Vert ^ 2 < \tilde{r},\\
\forall p _ j\in \mathcal{K},\\
\forall \tilde{f} _ j ^ {(p)} \in \mathcal{\tilde{F}}
\end{array}\right.
\right\}\tag{6}$$
类似的，再通过 PointNet 得到栅格点 \\(g _ i\\) 的特征：
$$\tilde{f} _ i ^ {(g)}=\mathrm{max}\left\{G\left(\mathcal{M}(\tilde{\psi})\right)\right\}\tag{7}$$
　　由此得到所有 ROI 固定长度的特征向量，进而可在 ROI Proposal 的基础上，作最后的尺寸，角度，位置等属性的精细化估计，这里不做展开。

## 4.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Shi, Shaoshuai, et al. "Pv-rcnn: Point-voxel feature set abstraction for 3d object detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.  
<a id="2" href="#2ref">[2]</a> https://github.com/open-mmlab/OpenPCDet  
<a id="3" href="#3ref">[3]</a> https://github.com/open-mmlab/mmdetection3d
