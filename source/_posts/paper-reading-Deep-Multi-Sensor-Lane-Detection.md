---
title: '[paper_reading] "Deep Multi-Sensor Lane Detection"'
date: 2019-11-09 11:54:34
tags: ["Deep Learning", "Lane Detection"]
categories: Lane Detection
mathjax: true
---

　　前文 {% post_link lane-det-from-BEV  Apply IPM in Lane Detection from BEV%} 已经较详细得阐述了俯视图下作车道线检测的逆透视原理，提到传统 IPM 有个较强的假设：地面是平坦的。对于 L4 自动驾驶，在这个假设下车道线检测不管是精度还是可靠性，都远远不够。如果有高精度地图，那么这些问题都有方法来消除。当然，如果有高精度地图，且自定位准确，也就不需要车道线检测了，所以这里讨论，在无高精度地图下，本文<a href="#1" id="1ref"><sup>[1]</a>如何通过激光点云数据学习的方法解决上述问题。  

## 1.&ensp;网络结构

<img src="lane_det.png" width="90%" height="90%" title="图 1. Multi-Sensor Lane Detection">
　　如图 1. 所示，整个算法有两个网络组成：

- **地面估计(Ground Height Estimation)网络**  
输入是俯视图下历史 N 帧的栅格点云，输出的是俯视图下地面高度；
- **车道线检测(Lane Prediction)网络**  
输入是俯视图下历史 N 帧的栅格点云，并且叠加前视图图像逆透视变换到俯视图后的图像，输出为像素级别的车道线检测结果；

历史 N 帧点云需要经过 ego-motion 补偿到当前本车位置，补偿后的点云只对运动物体会存在变形，而网络正好需要忽视运动物体。通过地面估计得到了俯视图下稠密的地面估计后，就可以将前视图的图像投影到俯视图下了。具体的过程为：取地面估计的三维点(高度+像素坐标经过分辨率变换后的物理坐标)，投影到图像上，然后双线性插值取得图像像素值，填充至俯视图上。这种透视变换是借助 3D 点信息完成的，原理可详见 {% post_link lane-det-from-BEV  Apply IPM in Lane Detection from BEV%}。  

## 2.&ensp;Differentiable Warping Function

　　其实这里估计出来的地面高度就是个简陋的高精度地图，所以这种方案理论上就能消除上述问题。并且，投影的过程采用了可求导的映射方程(differentiable warping function)，所以整个算法可以端到端的训练。
<img src="STN.png" width="90%" height="90%" title="图 2. Spatial Transformer Networks">
　　关于可求导的映射方程，这里借鉴了 DeepMind 的 Spatial Transformer Networks<a href="#2" id="2ref"><sup>[2]</sup></a> 的思想。传统卷积网络只对较小的位移有位移不变性，而 STN 引入 2D/3D 仿射/透视变换，显示得将特征层变换到有利于分类的形态，这样整个网络就具有了仿射甚至透视(位移，旋转，裁剪，尺度，歪斜)不变性。如图 2. 所示，STN 有三部分构成：

1. **Localisation Net**，对于 2D 仿射，回归预测出仿射变换矩阵 \\(\\theta \\in \\mathbb{R}_{2\\times 3}\\);
2. **Grid Generator**，根据仿射变换矩阵及仿射变换前后特征图的大小，建立仿射前后坐标映射关系；
3. **Sampler**，根据坐标映射关系设计可求导的插值采样方法(如双线性)，从输入特征中采样出特征值填入仿射后的特征图中；

　　本文则是一个透视变换矩阵 \\(P\\)，但是 \\(P\\) 不需要网络预测，其完全由激光雷达与相机的内外参决定，这个需要提前标定好。预测的地面高度通过 {% post_link lane-det-from-BEV  Apply IPM in Lane Detection from BEV%} 中的式 (3) 即可与图像坐标系建立联系，作为 Grid Generator。最后采用可求导的 Sampler，这个模块就可以嵌入到网络中，进行端到端的训练。

## 3.&ensp;Loss
　　Loss 采用 SmoothL1 Loss，其有两种构成：

- 地面估计项  
$$ L_{gnd} = \sum_{p\in Output Image} \Vert z_{p,gt}-z_{p,pred}\Vert \tag{1}$$
- 车道线检测项  
$$ L_{lane} = \sum_{p\in Output Image} \left\Vert \left(\tau-\mathrm{min}\{d_{p,gt}, \tau\}\right)-d_{p,pred}\right\Vert \tag{2}$$
其中 \\(\\tau\\) 是车道线真值标签的衰减像素区域，高速场景设为 30，城市道路设为 20。

## 4.&ensp;其它思考
　　既然 STN 专门是用来作仿射/透视变换的，那么是否可以在不借助激光点云的情况下，用前视图图像直接回归出透视变换到俯视图的透视矩阵 \\(P\\) ？理论上是可行的，但是训练过程不一定能收敛，需要精心设计训练过程，以及针对斜坡还会有一定的距离误差。

## 5.&ensp;参考文献
<a id="1" href="#1ref">[1]</a> Bai, Min, et al. "Deep Multi-Sensor Lane Detection." 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018.  
<a id="2" href="#2ref">[2]</a> Jaderberg, Max, Karen Simonyan, and Andrew Zisserman. "Spatial transformer networks." Advances in neural information processing systems. 2015.
