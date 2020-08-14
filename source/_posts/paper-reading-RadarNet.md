---
title: '[paper_reading]-"RadarNet"'
date: 2020-08-07 09:25:41
updated: 2020-08-12 09:19:12
tags: ["paper reading", "3D Detection", "Deep Learning", "autonomous driving", "Point Cloud"]
categories:
- 3D Detection
mathjax: true
---

　　Radar 相比 LiDAR/Camera，对天气影响鲁棒性较强，而且能直接测量目标速度，所以是多传感器融合感知里比较重要的一个输入。其在 ADAS 领域应用已经较为广泛。但是 Radar 的测量噪声较大，这给有效的多传感器融合带来了难度。本文提出了 RadarNet<a href="#1" id="1ref"><sup>[1]</sup></a>，利用 Radar 的几何数据以及动态数据，同时做特征级别的前融合以及注意力机制下的后融合，达到了较好的效果。

## 1.&ensp;Comparison between LiDAR and Radar
<img src="compare.png" width="90%" height="90%" title="图 1. Comparison">
　　LiDAR 可分为三种类型：Spinning，Solid State，Flash。目前主要采用的是 Spinning 旋转式的激光雷达。激光雷达的缺点有：

- 对雨雾雪，车尾气等比较敏感；
- 对玻璃等物体没有反射;
- 点云密度随着距离增加而下降，远距离探测能力较弱。

毫米波雷达则能克服以上缺点，并且能直接测量速度；但是缺点也比较明显：

- 分辨率低，对小目标探测能力较弱；
- 误检较多；
- 测量的速度只是径向速度。

毫米波雷达可输出三种形式的数据：1. 原始点运数据；2. 经过 DBSCAN 等聚类算法获得的聚类点集数据；3. 对点集作跟踪的数据。三种数据越来越高层，但是噪音越来越大。本文考虑输出点集的数据，设探测到的点集目标为 \\(Q = (q, v _ {||},m,t)\\)，其中 \\(q = (x,y)\\) 是俯视图下的位置，\\(v _ {||}\\) 是径向速度，\\(m\\) 代表目标是否运动，\\(t\\) 则为时间戳。我们需要进一步估计出目标的 2D 速度，由此在毫米波雷达的辅助下，能获得更长的检测距离，以及更准确的速度。
<img src="fusion.png" width="90%" height="90%" title="图 2. Fusion">
　　激光雷达数据与毫米波雷达数据融合示意图如图 2. 所示。

## 2.&ensp;RadarNet
<img src="framework.png" width="90%" height="90%" title="图 3. Framework">
　　如图 3. 所示，RadarNet 主要由 Voxel-Based Early Fusion，Detection Network 以及 Attention-Based Late Fusion 来做融合检测。前融合将各传感器数据通过俯视图形式进行表示并融合；后融合则通过基于注意力的数据关联及整合机制来对目标速度进行精细估计。检测网络是在传统的分类＋回归基础上，多了速度预测的分支。具体的，所有回归的预测量为 \\((x-p _ x,y - p _ y,w,l,\\mathrm{cos}(\\theta),\\mathrm{sin}(\\theta),m,v _ x, v _ y)\\)，其中 \\(p _ x,p _ y\\) 为体素/栅格的中心点，\\(m\\) 为目标是运动的概率，如果 \\(m < 0.5\\)，则将速度置为 0。

### 2.1.&ensp;Early Fusion
　　对于激光雷达数据，类似 {%post_link paperreading-Fast-and-Furious FAF%}，将时序多帧(0.5s)的点云数据在本车坐标系下打成俯视图体素表示，然后在通道维度进行串联。如果体素内没有点，那么该体素值为 0；如果体素内有点 \\(\\{(x _ i,y _ i,z _ i),i=1,...,N\\}\\)，那么体素值为 \\(\\sum _ i\\left( 1- \\frac{|x _ i-a|}{dx / 2}\\right)\\left( 1- \\frac{|y _ i-b|}{dy / 2}\\right)\\left( 1- \\frac{|z _ i-c|}{dz / 2}\\right)\\)，其中 \\((a,b,c)\\) 为体素中心坐标，\\(dx,dy,dz\\) 为体素尺寸。  
　　对于毫米波雷达数据，将其转到激光雷达坐标系后，也进行 BEV 时序串联，并将每一帧中不同线束的数据在 BEV 下体素化，然后串联。具体的，如果体素(本文丢掉了高度信息，所以退化为栅格)中没有毫米波探测到的目标，那么置为 0，如果探测到动态目标，则置为 1，如果探测到静态目标，则置为 -1。  
　　分别得到激光雷达与毫米波雷达的 BEV 表示后，将二者在通道维度串联起来，就完成了前融合。

### 2.2.&ensp;Late Fusion
　　**前融合关注毫米波雷达探测到的目标的位置和密度，后融合则使用毫米波雷达探测到的目标径向速度信息**。检测网络输出的目标状态为 \\(D = (c,x,y,w,l,\\theta,\\mathbf{v})\\)，毫米波雷达输出的目标状态为 \\(Q=(q,v _ {||},m,t)\\)。两者的目标级别的后融合通过 Association 和 Aggregation 两步骤组成。
<img src="late-fusion.png" width="90%" height="90%" title="图 4. Late-Fusion">
　　本文将 Association 和 Aggregation 用 End-to-End 的网络来处理。如图 4. 所示，首先进行检测目标与毫米波目标的成对特征提取，然后经过 MLP/Softmax 作匹配分数估计，最后根据分数作速度的加权优化。

#### 2.2.1.&ensp;Pairwise Detection-Radar Association
　　定义 Pairwise Feature 为：
$$\begin{align}
f(D,Q) &= \left(f ^ {det}(D),f ^ {det-radar}(D,Q)\right) \tag{1}\\
f ^ {det}(D) &= \left(w,l,||\mathbf{v}||,\frac{v _ x}{||\mathbf{v}||},\frac{v _ y}{||\mathbf{v}||},\mathrm{cos}(\gamma)\right) \tag{2}\\
f ^ {det-radar}(D,Q) &= \left(dx,dy,dt,v ^ {bp}\right) \tag{3}\\
v ^ {bp} &= \mathrm{min}\left(50,\frac{v _ {||}}{\mathrm{cos}(\phi)}\right) \tag{4}
\end{align}$$
其中 \\((\\cdot,\\cdot)\\) 是 Concatenation 操作；\\(\\gamma\\) 是网络检测 \\(D\\) 的运动方向与径向方向的夹角；\\(\\phi\\) 是 \\(D\\) 运动方向与雷达探测目标 \\(Q\\) 的径向方向的夹角；\\(v ^ {bp}\\) 是径向速度反投影到运动方向(朝向)的速度值；\\((dx,dy,dt)\\) 为俯视图下 \\(D,Q\\) 的相对位置和时间。由此，通过 MLP 学习匹配分数：
$$s _ {i,j}=\mathrm{MLP} _ {match}\left(f(D _ i,Q _ j)\right)\tag{5}$$

#### 2.2.2.&ensp;Velocity Aggregation
　　根据 \\(D,Q\\) 间的匹配分数，优化**目标的绝对速度值**。为了解决没有匹配的情况，分数 Concate 1，然后计算归一化的匹配分数：
$$s _ i ^ {norm}=\mathrm{softmax}((1,s _ {i,:})) \tag{6}$$
然后优化每个检测目标 \\(i\\) 的绝对速度值：
$$v _ i' = s _ i ^ {norm}\cdot
\begin{bmatrix}
||\mathbf{v} _ i|| \\
v _ {i,:} ^ {bp}
\end{bmatrix}
\tag{7}$$
最终可得到目标的 2D 速度：
$$\mathbf{v}' _ i = v' _ i\cdot\left(\frac{v _ x}{||\mathbf{v}||},\frac{v _ y}{||\mathbf{v}||}\right) \tag{8}$$
　　**由此可知，该后融合优化的只是目标的绝对速度(毫米波雷达也没办法探测目标的朝向或运动方向)，目标的朝向准确度还是由检测网络决定。**

## 3.&ensp;Experiments
　　根据式(2,3)提取的特征，作者设计了基于 Heuristic 的关联方法，毫米波雷达探测的目标与网络检测的目标关联的条件为：
$$\left\{\begin{array}{rl}
\sqrt{(dx) ^ 2+(dy) ^ 2} &< 3m \\
\gamma &< 40°\\
||\mathbf{v}|| &> 1m/s\\
v ^ {bp} &< 30m/s \\
\end{array}\tag{9}\right.$$
一旦关联上后，去毫米波雷达速度的中位数作目标速度的进一步优化。这种传统的 Heuristic 与本文的 Attention 方法对比如下，优势明显。
<img src="exp.png" width="80%" height="80%" title="图 5. Heuristic VS. Attention">

## 4.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Yang, Bin, et al. "RadarNet: Exploiting Radar for Robust Perception of Dynamic Objects." arXiv preprint arXiv:2007.14366 (2020).  
