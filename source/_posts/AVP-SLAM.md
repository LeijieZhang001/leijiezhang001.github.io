---
title: AVP-SLAM
date: 2020-07-15 09:17:56
updated: 2020-07-08 09:19:12
tags: ["Deep Learning", "Autonomous Driving", "Tracking", "MOT"]
categories:
- SLAM
mathjax: true
---

　　Visual-SLAM 一般采用特征点或像素直接法来建图定位，这种方式对光照较为敏感。本文<a href="#1" id="1ref"><sup>[1]</sup></a> 提出了一种基于语义特征的 Visual Semantic SLAM，应用于光照条件较为复杂的室内停车场环境，相比于采用特征点的 ORB-SLAM，性能较为鲁棒。

## 1.&ensp;Framework
<img src="framework.png" width="95%" height="95%" title="图 1. AVP-SLAM Framework">
　　如图 1. 所示，AVP-SLAM 由 Mapping，Localization 两部分组成。Mapping 阶段，将车周围的四张图通过 IPM 拼接并变换到俯视图，然后作 Guide Signs，Parking Lines，Speed Bumps 等语义信息的提取，接着通过 Odometry 将每帧的特征累积成局部地图，最后通过回环检测，全局优化出全局地图。Localization 阶段，提取出每帧的语义信息后，用 Odometry 初始化位姿，然后用 ICP 匹配求解当前帧在全局地图中的位姿，得到基于地图的位姿观测量，最后用 EKF 融合该观测量与 Odometry 信息，得到本车的最终位姿。  
　　有了本车在全局地图下的位姿后，然后通过语义信息识别停车位，即可达到本车自动泊车的目的。

## 2.&ensp;Mapping

### 2.1.&ensp;IPM
　　传感器为车身四周四个鱼眼相机，相机内外参已知。IPM(Inverse Perspective Mapping) 是将图像中的像素点投影到车身物理坐标系下的俯视图中，具体的：
$$\frac{1}{\lambda}\;
\begin{bmatrix}
x ^ v \\
y ^ v \\
1
\end{bmatrix} =
[\mathbf{R} _ c \;\mathbf{t} _ c] ^ {-1} _ {col:1,2,4} \;\pi _ c ^ {-1}
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}
\tag{1}$$
其中 \\(\\pi _ c(\\cdot)\\) 为鱼眼相机的内参矩阵，\\([\\mathbf{R} _ c\\;\\mathbf{t} _ c]\\) 为每个相机到车身坐标系的外参矩阵，\\([x ^ v\\; y ^ v]\\) 为车身坐标系下语义特征的位置。关于 IPM 更多细节可参考 {%post_link lane-det-from-BEV Apply IPM in Lane Detection from BEV%}。  
　　进一步将 IPM 图拼接成一张全景图：
$$\begin{bmatrix}
u _ {ipm}\\
v _ {ipm}\\
1
\end{bmatrix}=\mathbf{K} _ {ipm}
\begin{bmatrix}
x ^ v \\
y ^ v \\
1
\end{bmatrix}
\tag{2}$$
其中 \\(\\mathbf{K} _ {ipm}\\) 是全景图的内参。

### 2.2.&ensp;Feature Detection
<img src="segment.png" width="65%" height="65%" title="图 2. Segmentation in IPM Image">
　　将每张 IPM 图拼接成一张大全景图，然后用基于深度学习的语义分割方法，对全景图作像素级别作 lane，parking line，guide sign，speed bump，free space，obstacle，wall 等类别的语义分割。如图 4. 所示，parking line，guide sign，speed bump 是稳定的特征，用于定位；parking line 用于车位的识别；free space 与 obstacle 用于路径规划。

### 2.3.&ensp;Local Mapping
　　全景图语义分割得到的用于定位的特征(parking line，guide sign，speed bump)需要反投影回车身物理坐标系：
$$\begin{bmatrix}
x ^ v \\
y ^ v \\
1
\end{bmatrix}=\mathbf{K} _ {ipm} ^ {-1}
\begin{bmatrix}
u _ {ipm}\\
v _ {ipm}\\
1
\end{bmatrix}
\tag{3}$$
然后基于 Odometry 的相对位姿，将当前的语义特征点转换到世界坐标系下：
$$\begin{bmatrix}
x ^ w \\
y ^ w \\
z ^ w
\end{bmatrix}=\mathbf{R _ o}
\begin{bmatrix}
x ^ v \\
y ^ v \\
0
\end{bmatrix} + \mathbf{t _ o}
\tag{4}$$
由此得到局部地图，本文保持车身周边 30m 内的局部地图。

### 2.4.&ensp;Loop Detection
<img src="loop_det.png" width="65%" height="65%" title="图 3. Loop Detection">
　　因为 Odometry 有累计误差，所以这里对局部地图作一个闭环检测。如图 3. 所示，通过 ICP 对两个局部地图作匹配，一旦匹配成功，就说明检测到了闭环，ICP 匹配的相对位姿用于之后的全局位子图优化，以消除里程计累计误差。

### 2.5.&ensp;Global Optimization
　　检测到闭环后，需进行全局位姿图优化。位姿图中，节点(node)为每个局部地图的位姿：\\((\\mathbf{r, t})\\)；边(edge)有两种：odometry 相对位姿以及闭环检测中 ICP 匹配位姿。由此位姿图优化的损失函数为：
$$\chi ^ * = \mathop{\arg\min}\limits _ \chi \sum _ t\Vert f(\mathbf{r} _ {t+1},\mathbf{t} _ {t+1}, \mathbf{r} _ t, \mathbf{t} _ t) - \mathbf{z} ^ o _ {t,t+1}\Vert ^ 2 + \sum _ {i,j\in\mathcal{L}}\Vert f(\mathbf{r} _ i,\mathbf{t} _ i,\mathbf{r} _ j, \mathbf{t} _ j)-\mathbf{z} ^ l _ {i,j}\Vert ^ 2 \tag{5}$$
其中 \\(\\chi = [\\mathbf{r} _ 0,\\mathbf{t} _ 0,...,\\mathbf{r} _ t,\\mathbf{t} _ t] ^ T\\) 是所有局部地图的位姿，也是待优化的参数。\\(\\mathbf{z} ^ 0 _ {t,t+1}\\) 为 Odometry 得到的位姿。\\(\\mathbf{z} ^ l _ {i,j}\\) 为闭环检测 ICP 得到的位姿。\\(f(\\cdot)\\) 为计算两个局部地图相对位姿的方程。该优化问题可通过 Gauss-Newton 法求解。  
　　用优化后的位姿将局部地图叠加起来，就获得了整个场景的全局地图。

## 3.&ensp;Localization
<img src="loc.png" width="65%" height="65%" title="图 4. Localization">
　　有了全局地图后，基于全局地图的定位观测量可通过当前帧与全局地图的匹配得到。如图 4. 所示，绿色为当前帧检测到的语义特征，与全局地图匹配后即可得到当前的绝对位置。匹配通过 ICP 算法实现：
$$ \mathbf{r ^ * ,t ^ * } =  \mathop{\arg\min}\limits _ {\mathbf{r,t}}\sum _ {k\in\mathcal{S}}\Vert\mathbf{R(r)}
\begin{bmatrix}
x ^ v  _ k\\
y ^ v  _ k\\
0
\end{bmatrix} + \mathbf{t} - 
\begin{bmatrix}
x ^ w _ k \\
y ^ w _ k\\
z ^ w _ k
\end{bmatrix}
\Vert ^ 2 \tag{6}$$
其中 \\(\\mathcal{S}\\) 为当前帧语义特征点集，\\([x _ k ^ w\\; y _ k ^ w\\; z _ k ^ w]\\) 分别为对应的全局地图中最近的特征点集。  
　　ICP 的初始化非常重要，本文提出了两种初始化方法：1. 直接在地图上标记车库入口作为全局坐标点；2. 室外 GPS 信号初始化，然后用 Odometry 累积到车库。

## 4.&ensp;Extended Kalman Filter
　　Visual Localization 在语义特征较少的情况下，比如车辆停满了，定位会不稳定，所以这里采用 EKF 对 Visual Localization 与 Odometry 中的轮速计和 IMU 作扩展卡尔曼融合，这里不做展开。

## 5.&ensp;Thinkings
　　Semantic SLAM 相比基于几何特征点的 SLAM 更加鲁棒。但是在车库场景下，一旦车子停满后，停车线等语义信息会急剧减少，所以实际商业应用中，AVP-SLAM 能满足室内自动泊车的需求吗？  
　　对此我持怀疑态度。我认为，对于车库自动泊车的商业落地，可能最有效且低成本的方法还是得基于室内 UWB 定位技术。至少 UWB 可作为辅助。当然要将 UWB 应用于车载装置，目前好像还没有，不过随着车载软硬件系统的完善，手机上能做的事，车载平台问题也不大。

## 6.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Qin, Tong, et al. "AVP-SLAM: Semantic Visual Mapping and Localization for Autonomous Vehicles in the Parking Lot." arXiv preprint arXiv:2007.01813 (2020).

