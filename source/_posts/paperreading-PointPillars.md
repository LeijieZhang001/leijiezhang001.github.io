---
title: '[paper_reading]-PointPillars'
date: 2019-09-03 22:08:12
tags: ["paper reading", "3D Detection", "Deep Learning", "autonomous driving", "Point Cloud"]
categories: paper reading
mathjax: true
---

## 1.&ensp;VoxelNet->SECOND->PointPillars
　　相比于图像，激光点云数据是 3D 的，且有稀疏性，所以对点云的前期编码预处理尤其重要，目前大多数算法都是在鸟瞰图下进行点云物体检测，由此对点云的编码预处理主要有两大类方法：
1. 以一定的分辨率将点云体素化，每个垂直列中的体素集合被编码成一个固定长度，手工制作的特征，最终形成一个三维的伪图像，以此为代表的方法有 MV3D，AVOD，PIXOR，Complex YOLO；
2. PointNet 无序点云处理方式，以此为代表的方法 Frustum PointNet<a href="#1" id="1ref"><sup>[1]</sup></a>, VoxelNet<a href="#2" id="2ref"><sup>[2]</sup></a>，SECOND<a href="#3" id="3ref"><sup>[3]</sup></a>，后两者是在鸟瞰图下进行编码的，需要 3D 卷积运算；

　　本文提出的 PointPillar<a href="#4" id="4ref"><sup>[4]</sup></a> 是延续 VoxelNet，SECOND 的工作，VoxelNet 将 PointNet({% post_link PointNet-系列论文详读 PointNet-系列论文详读 %}) 思想引入体素化后的体素特征编码中，然后采用 3D 卷积做特征提取，再用传统的 2D 卷积进行目标检测；SECOND 则考虑到点云特征的稀疏性，用 2D 稀疏卷积代替传统卷积，速度得到了很大的提示。而 PointPillar 则在体素的垂直列上不做分割，从而移除了 3D 卷积的操作，其优点有：
- 无手工编码的过程，利用了点云的所有信息，且无需要调节的参数；
- 运算均为 2D 卷积，高效；
- 可迁移至其它点云数据；

　　这三篇工作框架结构基本一致，由三部分组成：
1. 特征编码网络(Encoder，作特征编码)，在鸟瞰图下，将点云编码为稀疏的伪图像；
2. 卷积中间网络(Middle，作特征提取)，将伪图像用 backbone 网络进行特征提取；
3. 区域生成网络(RPN)，也可以是 SSD FPN 等检测头的改进，用于分类和回归 3D 框，与图像检测不一样的地方是，点云鸟瞰图下的最后一层特征层不能很小；

<img src="PointPillar.png" width="100%" height="100%" title="图 3. PointPillar 网络框架">
　　如图 1. 所示，本文 Pointpillar 主要的工作集中在特征编码网络，所以以下主要介绍其特征编码网络方式，以及实现细节。

## 2.&ensp;特征编码
　　Pointpillar 只对 \\(x-y\\) 平面作 \\(H\\times W\\) 栅格化，栅格化后形成 \\(H\\times W=P\\) 个柱子(Pillar)，每个柱子采样出 \\(N\\) 个点，每个点编码为 \\(D=9\\) 维的向量：\\(\\{x,y,z,r,x_c,y_c,z_c,x_p,y_p \\}\\)，其中 \\(\\{x_c,y_c,z_c\\}\\) 为该点与柱子内所有点的均值点的距离，\\(x_p,y_p \\) 为该点与柱子中心的距离。综上最后形成\\(\(D,P,N \)\\) 维的张量，然后用 PointNet 网络输出 \\(\(C,P,N \)\\) 维的张量，最后用 \\(MAX\\) 操作输出 \\(\(C,P\) = \(C,H,W\)\\) 的伪图像。

## 3.&ensp;实现细节


<a id="1" href="#1ref">[1]</a> Qi, Charles R., et al. "Frustum pointnets for 3d object detection from rgb-d data." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.  
<a id="2" href="#2ref">[2]</a> Zhou, Yin, and Oncel Tuzel. "Voxelnet: End-to-end learning for point cloud based 3d object detection." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.  
<a id="3" href="#3ref">[3]</a> Yan, Yan, Yuxing Mao, and Bo Li. "Second: Sparsely embedded convolutional detection." Sensors 18.10 (2018): 3337.  
<a id="4" href="#4ref">[4]</a> Lang, Alex H., et al. "PointPillars: Fast encoders for object detection from point clouds." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.  

