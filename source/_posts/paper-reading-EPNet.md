---
title: '[paper_reading]-"EPNet"'
date: 2020-08-24 09:38:40
updated: 2020-08-28 09:19:12
tags: ["paper reading", "3D Detection", "Deep Learning", "autonomous driving", "Point Cloud"]
categories:
- 3D Detection
mathjax: true
---

　　在大多数场景下，融合激光雷达与图像数据能有效提升各种深度学习任务性能。本文<a href="#1" id="1ref"><sup>[1]</sup></a>提出了一种图像数据与激光雷达的前融合框架，并且考虑到分类分数与定位置信度的不一致性，提出了一种约束两者一致性的 Loss。

## 1.&ensp;Framework
<img src="framework.png" width="90%" height="90%" title="图 1. Framework">
　　如图 1. 所示，EPNet 主要由 Image Stream，Geometric Stream，LI-Fusion Module，及 Detect Head 组成。  
　　Image Stream 中，提取不同尺度的图像特征 \\(F _ i(i = 1,2,3,4)\\)，然后经过 2D Transposed Convolution 将不同尺度的特征变换到图像分辨率，得到特征 \\(F _ U\\)。  
　　Geometric Stream 用 PointNet++({%post_link PointNet-系列论文详读 PointNet-系列论文详读%}) 作特征提取，对应的 SA，FP 特征为 \\(S _ i,P _ i(i =1,2,3,4)\\)。\\(S _ i,F _ i\\) 通过 LI-Fusion 模块进行深度融合，此外 \\(F _ U\\) 与 \\( P _ 4\\) 也通过 LI-Fusion 进行融合。  

## 2.&ensp;LI-Fusion Module
<img src="LI.png" width="70%" height="70%" title="图 2. LI-Fusion">
　　LiDAR-guided Image Fusion Module 是图像点云两个数据流融合的核心模块。如图 2. 所示，LI-Fusion 由 Point-wise Correspondence Generation 和 LiDAR-guided fusion 两部分组成。Point-wise Correspondence Generation 又由 Grid Generator 和 Image Sampler 实现，对于点云中的点 \\(p(x,y,z)\\)，可得到不同尺度图像上的像素点 \\(p'(x',y')\\)：
$$p'=M\times p\tag{1}$$
其中 \\(M\\in\\mathbb{R}^ {3\\times 4}\\)。\\(p'\\) 可能不是正好位于图像坐标像素点上，所以用双线性插值的方法取邻近像素点的特征值：
$$V^{(p)}=\mathcal{K}\left(F^{\mathcal{N}(p')}\right)\tag{2}$$
其中 \\(V^{(p)}\\) 表示点 \\(p\\) 对应的图像点特征，\\(\\mathcal{K}\\) 表示双线性插值，\\(\\left(F^{\\mathcal{N}(p')}\\right)\\) 表示图像 \\(p'\\) 邻近点的特征。  
　　LiDAR-guided Fusion 考虑到不能直接将点的图像特征与点特征进行串联融合，因为图像特征容易受光照，遮挡等因素影响，所以通过点云对图像点特征进行重要性权重融合。如图 2. 所示，重要性权重设计为：
$$\mathbf{w}=\sigma\left(\mathcal{W}\;\mathrm{tanh}(\mathcal{U} F _ P+\mathcal{V}F _ I)\right)\tag{3}$$
其中 \\(\\mathcal{W,U,V}\\) 为 MLP 网络，\\(\\sigma\\) 为 sigmoid 归一化函数。  
　　最终的融合特征为：
$$F _ {LI}=\mathrm{Concate}(F _ P,\mathbf{w}F _ I)\tag{4}$$

## 2.&ensp;Consistency Enforcing Loss
　　NMS 操作时，一般用分类的分数，但是分类分数与定位置信度是不一致的。本文提出 Consistency Enforcing Loss，将定位与分类的分数监督成一致：
$$L _ {ce}=-log\left(c\times\frac{Area(D\cap G)}{Area(D\cup G)}\right)\tag{5}$$
其中 \\(D,G\\) 分别为预测框与真值框，\\(c\\) 为分类分数，该 Loss 鼓励定位准的框分类分数越高。  
　　这与 IoU Loss 作用相似！

## 3.&ensp;Ablation Study
<img src="ablation.png" width="90%" height="90%" title="图 3. Ablation Study">
　　如图 3. 所示，LI-Fusion 和 CE Loss 对检测性能提升还是比较明显的。此外，本文还对比了三种 Fusion 方式，另外两种为：SC(simple concatenation)，将原始图像像素值串联到对应的点云原始数据中，没有 Image Stream；SS(single scale)，只用最后一层的图像点云特征作融合。  
　　实验表明，SC 性能反而下降，SS 有所提升，但是 Multi-scale 的性能最好。结论就是，在一个尺度下，相对靠后的前融合可能比相对靠前的前融合效果更好(类比 {%post_link PointPainting PointPainting%}，其虽然是前融合，但是直接提取的是图像的语义分割结果，所以相对靠后，效果也好)，当然多尺度的效果会是最好的。

## 4.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Huang, Tengteng, et al. "EPNet: Enhancing Point Features with Image Semantics for 3D Object Detection." arXiv preprint arXiv:2007.08856 (2020).
