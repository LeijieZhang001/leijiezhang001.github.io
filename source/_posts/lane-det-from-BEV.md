---
title: Lane Detection from BEV
date: 2019-10-17 17:53:12
tags: ["Deep Learning", "Lane Detection"]
categories: Lane Detection
mathjax: true
---
　　车道线检测(Lane Detection)是 ADAS 系统中重要的功能模块，而对于 L4 自动驾驶系统，在不完全依赖高精度地图的情况下，车道线检测结果也是车辆运动规划的重要输入信息。由于俯视图(BEV, Bird's Eye View)下做车道线检测相比于前视图，有天然的优势，所以本文根据几篇论文(就看了两三篇)及项目经验，探讨总结俯视图下做车道线检测的流程方案，<a href="#0" id="0ref">[0]</a>为车道线检测资源集。

## 1.&ensp;流程框架
　　由于激光点云的稀疏性，所以目前车道线检测主要还是依靠图像，激光点云数据当然可作为辅助输入。由此归纳一种可能的粒度较粗的车道线检测的流程：

1. 仿射变换，将图像前视图变换为俯视图；
2. 网络，提取特征，进行像素级别的分类或回归；
3. 后处理，根据网络像素级别的预测，提取车道线；

网络相对比较成熟，后处理则在不同网络方法下复杂度差异很大，这里不做讨论。接下来主要讨论如何进行仿射变换。

## 2.&ensp;仿射变换
　　设变换前后图像坐标为 \\((u,v)\\), \\((u',v')\\), 仿射变换(Affine transformation)矩阵为 A，则：
$$\begin{bmatrix}
u' \\
v' \\
\end{bmatrix} = A
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix} =
\begin{bmatrix}
a_{11} &a_{12} &a_{13} \\
a_{21} &a_{22} &a_{23}
\end{bmatrix}
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix} \tag{1}$$

根据传感器的配置及传感器参数标定情况，从图像前视图到俯视图的仿射变换有许多方法，这里介绍两种代表性的方法。

### 2.1.&ensp;IPM(Inverse Perspective Mapping)
　　<a href="#1" id="1ref">[1]</a>可能是首次用 IPM 进行车道线检测的文章，其检测流程基于传统方法，包括图像二值化，轮廓提取，IPM，聚类，拟合车道线等流程。  
　　逆透视 IPM 是透视变换的逆变换，透视变换过程为：
$$\begin{align}
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix} &= 
\frac{1}{Z_{cam}}
\begin{bmatrix}
f_x &0 &u_0 \\
0 &f_y &v_0\\
0 &0 &1
\end{bmatrix}
\begin{bmatrix}
R &t\\
0 &1
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z 
\end{bmatrix}_{world}\\
&= \frac{1}{Z_{cam}}
\begin{bmatrix}
m_{11} &m_{12} &m_{13}\\
m_{21} &m_{22} &m_{23}\\
m_{31} &m_{32} &m_{33}\\
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z 
\end{bmatrix}_{world} = \frac{1}{Z_{cam}} M 
\begin{bmatrix}
X \\
Y \\
Z 
\end{bmatrix}_{world} \tag{2}
\end{align}$$

其中 \\(M\\) 为相机内参及外参，所以 **IPM 需要预先标定相机的内外参**，尤其是外参 \\(R\\)，表示与地面平行的世界坐标系与相机成像平面的相机坐标系之间的旋转关系。  
　　基于(2)，可以求出世界坐标系下两条平行 \\(z\\) 轴的平行直线在图像坐标系下的交点，即**消失点(Vanishing Point)**。假设世界坐标系下平行 \\(z\\) 轴的直线表示为，点 \\((x_a,x_b,x_c)\\) 及方向向量 \\(k(0,0,1)\\)，那么可得该直线上任意一点投影到图像坐标系下表示，当 \\(k\\) 趋向于无穷大时，即可得到消失点坐标：
$$\begin{bmatrix}
u \\
v \\
1
\end{bmatrix} = \frac{1}{Z_{cam}} M 
\begin{bmatrix}
x_a \\
x_b \\
x_c + k
\end{bmatrix}_{world} = 
\begin{bmatrix}
\frac{m_{11}x_a+m_{12}x_b+m_{13}(x_c+k)}{m_{31}x_a+m_{32}x_b+m_{33}(x_c+k)} \\
\frac{m_{21}x_a+m_{22}x_b+m_{23}(x_c+k)}{m_{31}x_a+m_{32}x_b+m_{33}(x_c+k)} \\
1
\end{bmatrix} \overset{k\to\infty}{\simeq}
\begin{bmatrix}
\frac{m_{13}}{m_{33}} \\
\frac{m_{23}}{m_{33}} \\
1
\end{bmatrix}
\tag{3}
$$
有了图像坐标系下的消失点坐标以后，我们就可以选定需要作仿射变换的 ROI 梯形区域(仿射到俯视图后，梯形变矩形)。选定梯形四个角点后，根据像素距离关系，定义俯视图下其对应的矩形框四个像素坐标点，这样能得到四组(1)方程组，由最小二乘即可得到仿射变换的矩阵 A。更详细的代码原理可见<a href="#2" id="2ref">[2]</a>。  
　　直接 IPM 需要标定相机内外参，并且有一个较强的假设：路面是平坦的。所以时间一长标定参数，尤其是外参会失效，而且距离越远，路面的不平坦导致的逆透视变换误差也会增大。但对于 ADAS 系统来说，车道偏离预警(LDW，Lane Departure Warnings) 中车道线的检测距离在 50m 已经能满足要求。  
　　按照之前的项目经验，LDW 系统完成度可以很高，基本思路就是 IPM，parsing(segmentation)，clustering，hough，optimization 等几个步骤(这里就不能说得太细了)，更多的精力可能在指标设计及 cornercase 优化上。唯一对用户不太友好的地方就是安装时要进行相机外参(尤其是 pitch 角)的标定，当然标定方法比较简单，我们假设相机坐标系与路面平行，所以仿射函数是固定的，用户只要看路面经过仿射后，两条 \\(z\\) 方向的直线是否平行即可。相对于 Mobileye 这种标定巨麻烦的产品，这种标定方式算是非常友好了。此外还可以用自动外参标定方法，脑洞也可以开出很多，效果嘛看具体环境了，需要作谨慎的收敛判断。

### 2.2.&ensp;学习映射
　　对于 L4 自动驾驶，以上方法检测的车道线不管是精度还是可靠性，都远远不够。如果有高精度地图，那么这些问题都有方法来消除。当然，如果有高精度地图，且自定位准确，也就不需要车道线检测了，所以这里讨论，在无高精度地图下，但是有激光点云数据，我们如何通过学习的方法解决上述问题。  
<img src="lane_det.png" width="90%" height="90%" title="图 1. Multi-Sensor Lane Detection">
　　这里主要介绍<a href="#3" id="3ref">[3]</a>的思路。如图 1. 所示，整个算法有两个网络组成，Lane Prediction 网络是做车道线检测；另一个网络是作地面估计(Ground Height Estimation)。地面估计网络的输入是历史 N 帧的栅格点云，点云经过 ego-motion 补偿到当前本车位置，点云只对运动物体会存在变形，而网络正好需要忽视运动物体。  
　　得到了俯视图下稠密的地面估计后，就可以将前视图的图像投影到俯视图下了。具体的过程为：取地面估计的三维点，投影到图像上，然后双线性插值取得图像像素值，填充至俯视图上。这种仿射变换是借助 3D 点信息完成的，俯视图上获得的地面信息与实际物理尺寸是一致的(IPM 法并不一致)。  
　　其实这里估计出来的地面高度就是个简陋的高精度地图，所以这种方案理论上就能消除上述问题。并且，投影的过程采用了可求导的映射方程(differentiable warping function)，所以整个算法可以端到端的训练。
<img src="STN.png" width="90%" height="90%" title="图 2. Spatial Transformer Networks">
　　关于可求导的映射方程，这里借鉴了 DeepMind 的 Spatial Transformer Networks<a href="#4" id="4ref"><sup>[4]</sup></a> 的思想。传统卷积网络只对较小的位移有位移不变性，而 STN 引入 2D/3D 仿射变换，显示得将特征层变换到有利于分类的形态，这样整个网络就具有了仿射(位移，旋转，裁剪，尺度，歪斜)不变性。如图 2. 所示，STN 有三部分构成：

1. Localisation Net，对于 2D 仿射，回归预测出仿射变换矩阵 \\(\\theta \\in \\mathbb{R}_{2\\times 3}\\);
2. Grid Generator，根据仿射变换矩阵及仿射变换前后特征图的大小，建立仿射前后坐标映射关系；
3. Sampler，根据坐标映射关系设计可求导的插值采样方法(如双线性)，从输入特征中采样出特征值填入仿射后的特征图中；

　　这里不需要回归仿射变换矩阵 \\(\\theta\\)，预测的地面高度即可作为 Grid Generator，然后采用可求导的 Sampler，这个模块就可以嵌入到网络中，进行端到端的训练。

## 3.&ensp;其它思考
　　既然 STN 专门是用来作仿射变换的，那么是否可以在不借助激光点云的情况下，用前视图图像直接回归出仿射变换到俯视图的仿射矩阵 \\(\\theta\\) ？理论上是可行的，但是训练过程不一定能收敛，需要精心设计训练过程，以及针对斜坡会有一定的距离误差。

## 4.&ensp;参考文献
<a id="0" href="#0ref">[0]</a> [awesome-lane-detection](https://github.com/amusi/awesome-lane-detection)  
<a id="1" href="#1ref">[1]</a> Wang, Jun, et al. "An approach of lane detection based on inverse perspective mapping." 17th International IEEE Conference on Intelligent Transportation Systems (ITSC). IEEE, 2014.  
<a id="2" href="#2ref">[2]</a> [LDW 原理及代码](https://blog.csdn.net/qq_32864683/article/details/85471800)  
<a id="3" href="#3ref">[3]</a> Bai, Min, et al. "Deep Multi-Sensor Lane Detection." 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018.  
<a id="4" href="#4ref">[4]</a> Jaderberg, Max, Karen Simonyan, and Andrew Zisserman. "Spatial transformer networks." Advances in neural information processing systems. 2015.
