---
title: '[paper_reading]-"CenterFusion"'
date: 2020-11-30 09:20:39
updated: 2020-12-02 09:34:12
tags: ["3D Detection", "Deep Learning", "Radar", "Fusion"]
categories:
- 3D Detection
- Fusion
mathjax: true
---

　　自动驾驶领域多传感器融合对感知及定位都非常重要，对于感知而言，融合可分为数据前融合，特征级融合，目标状态后融合等类型。前融合对外参标定要求较高，后融合没有深度融合各传感器特征，特征级融合是比较折中的方法。{%post_link MOT-Fusion MOT Multimodal Fusion%} 中介绍了基于 BCM 来实现目标状态后融合的方法，本文<a href="#1" id="1ref"><sup>[1]</sup></a> 提出一种特征级融合相机以及毫米波雷达数据的目标检测方法。  
　　{%post_link paper-reading-RadarNet RadarNet%} 中详细介绍了毫米波雷达与激光雷达的优劣势，并提出了一种特征级融合方法，能更准确的测量目标的速度。此外，相机与激光雷达一样对恶劣天气环境比较敏感，所以毫米波与相机的结合，能有效利用毫米波能测量目标径向速度，应对恶劣环境以及测量范围较远等优势，并且保留相机高分辨率捕捉环境视觉信息的特性。  

## 1.&ensp;Framework
<img src="framework.png" width="100%" height="100%" title="图 1. Framework">
　　如图 1. 所示，CenterFusion 网络首先用 CenterNet 作图像的 3D 目标检测，然后通过 Frustum Association Module 提取并融合对应的图像特征以及毫米波雷达特征，最后通过网络进一步准确估计目标的 3D 属性。  
　　{%post_link Anchor-Free-Detection Anchor Free Detection%} 以及 {%post_link CenterTrack CenTrack%} 已经较为详细的介绍了 CenterNet 的网络结构和 Loss 形式，其真值 Heatmap 生成方式与 {%post_link paper-reading-AFDet AFDet%} 也类似，这里不做展开。  
<img src="vel.png" width="60%" height="60%" title="图 2. Radial Velocity">
　　目前广泛使用的 3D 毫米波雷达测量的量有 \\(x,y,v\\)，如图 2. 所示，其中 \\(v\\) 是径向速度，为目标实际速度在径向的投影。为了准确估计目标的实际速度，需要估计目标的运动方向。

## 2.&ensp;Association and Feature Fusion
　　图像经过 CenterNet 得到目标的 2D size, Center Offset 等 2D 属性，以及 dimensions，depth, rotation 等 3D 属性。接下来要将 CenterNet 得到的目标与毫米波雷达的测量量进行数据关联，以便作进一步的特征融合与属性估计。  

### 2.1.&ensp;Frustum Association Mechanism
　　基于图像的目标检测结果与毫米波雷达关联最简单的方法是将毫米波的测量点投影到图像中，看其是否处于图像 2D 框内。但是毫米波雷达测测量量没有 \\(z\\) 信息，所以这种方式不准确。  
<img src="asso.png" width="90%" height="90%" title="图 3. Frustum Association">
　　本文提出一种锥形关联方法，如图 3. 所示，在俯视三维坐标下的锥形中进行关联，其中 \\(\\sigma\\) 用来控制感兴趣锥形的尺寸，因为基于图像的目标 depth 估计准确度较差，所以 \\(\\sigma\\) 可用来调节 depth 范围。图像目标只关联距离坐标原点最近的毫米波测量量。

### 2.2.&ensp;Radar Feature Fusion
　　 当图像 2D 目标框与毫米波雷达测量量关联上后，将毫米波雷达的测量信息融合到 2D 框中。具体的，在 2D 目标框提取出的图像特征上，concate 毫米波雷达测量的 heatmap。heatmap 尺寸与 2D 目标框尺寸相关，heatmaps 定义为：
$$ F ^ j _ {x,y,i} = \frac{1}{M _ i}
\left\{\begin{array}{l}
f _ i \;\;\vert x - c _ x ^ j\vert \leq \alpha w ^j \;\mathrm{and}\;\vert y- c _ y ^ i\vert\leq\alpha h ^ j\\
0 \;\;\mathrm{otherwise}
\end{array}\tag{1}\right.$$
其中 \\(\\alpha\\) 是 heatmap 尺寸比例；\\(i\\in 1,2,3\\) 是 heatmaps 特征维度；\\(M _ i\\) 是归一化系数；\\(f _ i\\in d, v _ x, v _ y\\) 为毫米波雷达的测量量。  
　　有了目标框内融合的图像及毫米波雷达特征后，可进一步精确估计目标的位置，朝向，速度，尺寸等 3D 属性。

## 3.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Nabati, Ramin, and Hairong Qi. "CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection." arXiv preprint arXiv:2011.04841 (2020).  

