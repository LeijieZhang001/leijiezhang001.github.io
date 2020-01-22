---
title: Ground Segmentation with Gaussian Process
date: 2020-01-21 17:00:34
tags: ["Segmentation", "autonomous driving", "Point Cloud"]
categories: Semantic Segmentation
mathjax: true
---

　　地面分割可作为自动驾驶系统的一个重要模块，本文介绍一种基于高斯过程的地面分割方法。

## 1.&ensp;算法概要
<img src="ground_seg.png" width="80%" height="80%" title="图 1. ground segmentation">
　　为了加速，本方法<a href="#1" id="1ref"><sup>[1]</sup></a>将三维地面分割问题分解为多个一维高斯过程来求解，如图 1. 所示，其步骤为：

1. Polar Grid Map  
将点云用极坐标栅格地图表示，二维地面估计分解成射线方向的多个一维地面估计；
2. Line Fitting  
在每个一维方向，根据梯度大小，作可变数量的线段拟合；
3. Seed Estimation  
在半径 \\(B\\) 范围内，如果某个 Grid 绝对高度(Grid 高度定义为该 Grid 内所有点的最小高度，其绝对高度则是与本车传感器所在地面的比较)大于 \\(T_s\\)，那么就将其作为 Seed；
4. Ground Model Estimation with Gaussian Process  
采用高斯过程生成每个一维方向 Grid 的地面估计量，这里为了进一步加速，可以删除冗余的 Seed；根据地面估计模型，将满足模型的 Grid 加入 Seed，更新模型，迭代直至收敛，满足模型的 Seed 条件为：
$$\begin{align}
V[z]&\leq  t_{model}\\
\frac{|z_*-\bar{z}|}{\sqrt{\sigma^2_n+V[z]}} &\leq t_{data}
\end{align}$$
5. Point-wise Segmentation
得到地面估计模型后，每个 Grid 都获得了模型的地面高度，对于 Grid 内的点，与估计高度的相对高度小于 \\(T_r\\)，则认为该点属于地面。

## 2.&ensp;高斯过程


## 3.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Chen, Tongtong, et al. "Gaussian-process-based real-time ground segmentation for autonomous land vehicles." Journal of Intelligent & Robotic Systems 76.3-4 (2014): 563-582.  
