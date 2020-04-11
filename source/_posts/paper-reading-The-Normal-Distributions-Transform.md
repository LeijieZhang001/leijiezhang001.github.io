---
title: The Normal Distributions Transform for Laser Scan Matching
date: 2020-04-10 09:39:54
tags: ["SLAM", "Localization", "Point Cloud"]
categories: SLAM
mathjax: true
---

　　机器人系统中，定位是非常重要的模块。基于 SLAM/VO/VIO 技术的算法能实时作机器人的自定位，但是这种开环下的里程计方案很容易累积绝对误差，使得定位漂移。而离线建立的地图因为有闭环检测，精度很高，所以基于地图的定位方法有很高的绝对定位精度。  
　　{% post_link LOAM LOAM %} 是一种基于点云的实时建图与定位方法，其中当前帧点云与前序建立的地图点云配准的方法，采用了提取线、面特征并建立点-线，点-面特征匹配误差函数，从而最小二乘非线性优化求解位姿。这种方案如果特征点噪声较大无匹配对，那么就会有较大的误差。本文<a href="#1" id="1ref"><sup>[1]</sup></a> 将地图点云栅格化，每个栅格又统计点云的高斯分布，匹配的时候计算该帧点云在每个栅格的概率，从而迭代至最优匹配位姿。  
　　**有闭环检测**的 SLAM 建立的地图即可作为离线定位地图，定位的过程就是当前时刻点云与地图配准的过程，当然后续可以融合其它传感器(GPS，IMU)输出最终的绝对位姿。**点云与地图配准的过程与建图时点云与局部地图或上一时刻点云配准的过程非常相似**。本文介绍一种区别于 {% post_link LOAM LOAM %} 特征匹配的基于概率统计优化的 NDT 配准方法。

## 1.&ensp;点云配准算法过程
　　考虑二维情况，本文点云配准算法过程为：

1. 建立 \\(t-1\\) 帧点云的 NDT；
2. 初始化待优化的相对位姿参数 \\(T\\);
3. 用 \\(T\\) 将 \\(t\\) 帧点云变换到 \\(t-1\\) 坐标系；
4. 找到变换每个变换点对应的 \\(t-1\\) 帧栅格的高斯分布；
5. 该变换 \\(T\\) 的度量分数为变换点在高斯分布下的概率和；
6. 用 Newton 法迭代优化 \\(T\\);
7. 重复 3. 直到收敛；

　　这里主要涉及 NDT，目标函数构建(即 \\(T\\) 的度量分数)，Newton 法优化三个内容。

### 1.1.&ensp;NDT
　　NDT 是点云栅格化后一系列高斯分布的表示，其过程为：

1. 将点云进行栅格化；
2. 统计每个栅格的点 \\(\\mathbf{x} _ {i=1..n}\\)；
3. 计算每个栅格高斯分布的 Mean: \\(\\mathbf{q} = \\frac{1}{n}\\sum _ i\\mathbf{x} _ i\\);
4. 计算 Covariance Matrix: \\(\\Sigma = \\frac{1}{n}\\sum _ i(\\mathbf{x} _ i -\\mathbf{q})(\\mathbf{x} _ i-\\mathbf{q})^t\\)；

　　由此，**NDT 描述了栅格内每个位置出现点的概率**，即 \\(\\mathbf{x}\\) 有点的概率为：
$$ p(\mathbf{x}) \sim \mathrm{exp}\left(-\frac{(\mathbf{x-q})^t\sum ^ {-1}(\mathbf{x-q})}{2}\right) \tag{1}$$
需要注意的是 {% post_link Grid-Mapping Grid-Mapping %} 描述的是每个栅格有点的概率，NDT 描述的是每个栅格点云的概率分布。为了更准确的建模，采用重叠栅格化的设计以消除离散化的影响，以及限定 Covariance 矩阵的最小奇异值。

### 1.2.&ensp;目标函数构建
　　考虑二维情况，需要优化的位姿参数为 \\(\\mathbf{p}=(t _ x, t _ y, \\varphi)^t\\)，第2个点云(待配准点云)中的点为 \\(\\mathbf{x} _ i\\)，其变换到第1个点云坐标系后的表示为 \\(\\mathbf{x}' _ i\\)，对应的第1个点云栅格的 NDT 表示为 \\(\\mathbf{\\Sigma} _ i, \\mathbf{q} _ i\\)。由此可计算该变换位姿下，其度量分数为：
$$\mathrm{score}(\mathbf{p})=\sum _ i\mathrm{exp}\left(-\frac{(\mathbf{x}' _ i-\mathbf{q} _ i)^t\sum _ i ^ {-1}(\mathbf{x}' _ i-\mathbf{q} _ i)}{2}\right) \tag{2}$$
最大化度量函数即可求解最优的位姿，优化过程一般都是最小化目标函数，所以设定目标函数为 \\(-\\mathrm{score}\\)。

### 1.3.&ensp;Newton 法优化迭代
　　设 \\(\\mathbf{q}=\\mathbf{x}' _ i-\\mathbf{q} _ i\\)，那么目标函数为：
$$ s = -\mathrm{exp}\frac{-\mathbf{q^t\sum ^ {-1}q}}{2} \tag{3}$$
每次迭代过程为：
$$\mathbf{p\gets p+\Delta p} \tag{4}$$
而 \\(\\mathbf{\\Delta p}\\) 来自：
$$\mathbf{H\Delta p} = \mathbf{-g} \tag{5}$$
其中 \\(\\mathbf{g}\\) 是目标函数对优化参数的导数，\\(\\mathbf{H}\\) 为目标函数的 Hessian 矩阵：
$$\left\{\begin{array}{l}
g _ i=\frac{\partial s}{\partial p _ i}\\
H _ {ij} = \frac{\partial s}{\partial p _ i\partial p _ j}
\end{array}\tag{6}\right.$$

## 2.&ensp;建图与定位
　　本文的建图是通过**关键帧集合与关键帧之间的位姿变化实现的**，定位的时候去找重合度最高的关键帧作点云配准。此外，当找不到重合度较高的关键帧时，可以实时更新当前帧作为关键帧添加到地图中，还可以对地图作进一步的全局，半全局优化。

## 3.&ensp;一些思考
　　本文建图是关键帧的形式，更鲁棒的做法是将点云配准到一起，在世界坐标系下获得场景的稠密点云，然后再 NDT 化，这样能更准确的建模点云分布。  
　　{% post_link LOAM LOAM %} 维护的是栅格化的地图，每个栅格限制特征点的数量，所以本质上存储的是原始点云图(被选出是特征点的点云)。为了更好的描述栅格内的特征分布，可以对其作类似 NDT 近似，同时加入能描述该分布的特征，比如对于面特征，加入法向量。  

## 4.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Biber, Peter & Straßer, Wolfgang. (2003). The Normal Distributions Transform: A New Approach to Laser Scan Matching. IEEE International Conference on Intelligent Robots and Systems. 3. 2743 - 2748 vol.3. 10.1109/IROS.2003.1249285. 
