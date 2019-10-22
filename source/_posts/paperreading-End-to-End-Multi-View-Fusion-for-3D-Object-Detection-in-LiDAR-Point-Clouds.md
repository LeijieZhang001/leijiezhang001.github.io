---
title: '[paper_reading]-"End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds"'
date: 2019-10-21 11:30:27
tags: ["paper reading", "3D Detection", "Deep Learning", "autonomous driving", "Point Cloud"]
categories: paper reading
mathjax: true
---
　　在多视角融合 3D 检测上，研究比较多的是俯视图下的激光点云以及前视图下的图像做多传感器融合，而融合点云俯视图(Bird's Eye View)与前视图(Perspective View)的特征则比较少，新鲜出炉的本文<a href="#1" id="1ref"><sup>[1]</sup></a>提供了一种较好的点云前视图与俯视图特征前融合(early fusion)方法。

## 1.&ensp;为什么要融合点云前视图特征
　　目前主流的点云检测算法，都是将点云在俯视图下以一定分辨率体素化(Voxelization)，然后用网络提取特征做 3D 检测。单纯在俯视图下提取特征虽然比单纯在前视图下做有优势，但还是存在几个问题：

1. 激光点云在远处，会变得很稀疏，从而空像素会比较多；
2. 行人等狭长型小目标特征所占像素会很小；

将点云投影到前视图，这两个问题则能有效减弱，所以本文提出融合点云前视图特征。

## 2.&ensp;贡献点
　　本文是在 {% post_link paperreading-PointPillars PointPillars %} 基础上做的工作，PointPillars 主要由三个模块构成：

- Voxelization；
- Point Feature Encoding；
- CNN Backbone；

本文改进了前两个模块，但是本质思想还是 PointNet 形式。其余包括 Loss 形式等与 PointPillars 一致。  
　　针对这两个模块，本文有两个贡献点，Dynamic Voxelization 以及 Point-level Feature Fusion，接下来作详细介绍。

### 2.1.&ensp;动态体素化(Dynamic Voxelization)
<img src="voxelization.png" width="90%" height="90%" title="图 1. 体素化过程对比">
　　如图 1. 所示，PointPillars (包括之前的 VoxelNet 等工作)体素化的过程都是 Hard Voxelization，即 Voxel 数目要采样，每个 Voxel 里面的点数也会采样，比如 PointPillars 将每个 Voxel 的点数定义为 100 个，少于 100 个点，则作补零处理。这样会存在问题：

- 内存消耗大，很多稀疏的区域导致体素中要补零的内存很多；
- 采样导致信息丢失；
- 采样导致检测输出有一定的不一致性；
- 不能作点级别的特征融合；

　　由此提出动态体素化(Dynamic Voxelization)，取消所有的采样过程，为什么可以这么做呢？其实这么做也比较自然，PointPillars 中 PointNet 网络将 \\((P, N, D)\\) 特征映射为 \\((P, N, C)\\)，这里就是多层感知机将输入的 channel 维度从 \\(D\\) 变换到 \\(C\\)，与其它两个维度没有关系，而接下来做的 max-pooling 操作则将 \\(N\\) 维(N 个点)压缩到 1，PointPillars 中每个柱子的 N 是采样成一样的。但是可以不一样！这就是本文的动态体素化思想了。

### 2.2.&ensp;点级别特征融合(Point-level Feature Fusion)
　　{% post_link paperreading-MT-MS-Fusion-for-3D-Object-Detection MMF %} 以 Voxel-level 将前视图的图像特征融合到俯视图的点云特征中，并以 ROI-level 融合图像前视图特征及点云俯视图特征做检测分类，本文则提出了更加前序的特征融合-Point-level 融合。
<img src="MVF.png" width="90%" height="90%" title="图 2. 点级别特征融合框架">
　　如图 2. 所示，首先将每个点的特征(x,y,z,intensity...)映射到高维度，然后经过 FC+Maxpool(PointNet 形式) 得到标准卷积网络需要的输入数据形式，再经过 Convolution Tower 模块进行环境上下文特征提取，最终每个体素的特征作为体素内每个点的特征，由此拼接成总的点特征。  
<img src="encoding.png" width="40%" height="40%" title="图 3. Convolution Tower">
　　其中 Convolution Tower 网络结构如图 3. 所示，输入输出的尺寸保持不变，类似于 FPN 结构。  
　　最终每个点的特征由三部分构成：

- 自身特征维度映射；
- 俯视图下抽取的 Voxel 级别特征，有一定的感受野；
- 前视图下抽取的 Voxel 级别特征，有一定的感受野；

　　俯视图下点云特征提取过程我们比较熟悉了，这里再详细介绍下点云在前视图下提取特征的过程(还没看懂，论文中好像没有详细信息，看懂了再补充)。

## 3.&ensp;实验结果
<img src="eval.png" width="90%" height="90%" title="图 4. 实验结果">
　　网络参数配置可详见论文，图 4. 是在 Waymo 公开数据集上的实验结果。可知：

1. 动态体素化在全距离范围内对检测都有一定的提升；
2. 融合前视图特征能有效提升提升检测性能，尤其是远距离情况，距离越远，提升越明显；
3. 融合前视图特征对小目标提升更加明显，如行人；

## 4.&ensp;参考文献
<a id="1" href="#1ref">[1]</a> Zhou, Yin, et al. "End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds." arXiv preprint arXiv:1910.06528 (2019).
