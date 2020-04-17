---
title: '[paper_reading]-"PolarNet"'
date: 2020-04-16 09:19:12
tags: ["paper reading", "Segmentation", "Deep Learning", "autonomous driving", "Point Cloud"]
categories: Semantic Segmentation
mathjax: true
---

　　Point-wise 特征提取在 {%post_link PointCloud-Feature-Extraction PointCloud-Feature-Extraction%} 中已经有较为详细的描述，虽然 Point-wise 提取的特征更加精细，但是一般都有 KNN 构建及索引操作，计算量较大，而且实践中发现学习收敛较慢。Voxel-based 虽然理论上损失了一定的信息，但是能直接应用 2D 卷积网络，网络学习效率很高。本文<a href="#1" id="1ref"><sup>[1]</sup></a>提出了一种在极坐标下栅格化后进行点云 Semantic Segmentation 的方法，相比传统的笛卡尔坐标系下栅格化有一定的优势。

## 1.&ensp;Voxelization
<img src="pts.png" width="98%" height="98%" title="图 1. Cartesian VS. Polar">
　　如图 1. 所示，传统的笛卡尔坐标系下栅格化的栅格是矩形，而极坐标系下栅格是饼状的。激光雷达是在极坐标方式下获取点云的，所以由图可知，**极坐标栅格化下，每个栅格拥有的点数更加均匀**，有利于网络学习并减少计算量。此外，本文统计后显示，相比笛卡尔坐标栅格，极坐标的栅格内点属于同一目标的概率更大。

## 2.&ensp;PolarNet Framework
<img src="framework.png" width="98%" height="98%" title="图 2. PolarNet">
　　如图 2. 所示，点云经过 Polar 栅格化后，对每个栅格首先进行 PointNet 特征提取，然后对所有栅格作 ring-convolution 操作。  
　　ring-convolution 是指卷积在环形方向进行，没有边缘截断效应。实现上，将栅格从某处展开，然后边缘处用另一边对应的栅格进行 padding，即可用普通的卷积进行运算。  
　　网络是作 Voxel-wise 的分割，然后直接将预测的类别应用到栅格内的点云中。统计上，同一栅格内的点云属于不同类别的概率很低，所以本文并没进一步作 Point-wise 的分割。

## 3.&ensp;Rethinking
　　PolarNet 作 Semantic Segmentation 比其它方法提升很多。但是实际应用时，PolarNet 不能指定各个方向的范围，所以计算效率较低。比如，自动驾驶中，我们可以设定前 100m，后 60m，左右各 30m 的检测范围，笛卡尔坐标系下很容易进行栅格化，而极坐标下则没法搞。所以为了解决点云的分布不均匀问题，另一种思路是在笛卡尔坐标系下，近处打高分辨率的栅格，远处打低分辨率的栅格。具体实现，可以先用低分辨率过一遍网络，然后再对感兴趣的特定区域作高分辨率检测。

## 4.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Zhang, Yang, et al. "PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation." arXiv preprint arXiv:2003.14032 (2020).
