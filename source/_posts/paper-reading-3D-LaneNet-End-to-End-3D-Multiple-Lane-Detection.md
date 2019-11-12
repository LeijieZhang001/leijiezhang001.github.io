---
title: '[paper_reading]-"3D-LaneNet End-to-End 3D Multiple Lane Detection"'
date: 2019-11-09 17:48:38
tags: ["Deep Learning", "Lane Detection"]
categories: Lane Detection
mathjax: true
---

　　在上一篇 paper reading {% post_link paper-reading-Deep-Multi-Sensor-Lane-Detection Deep Multi-Sensor Lane Detection%} 中，最后我提到一个思考点：借鉴 STN 的思路，用前视图直接去回归 IPM 变换需要的矩阵参数。本文<a href="#1" id="1ref"><sup>[1]</a>就是采用了这种思路！
<img src="res.png" width="60%" height="60%" title="图 1. 方法概图">
如图 1. 所示，车道线检测还是在俯视图下来做的，车道线输出是三维曲线，一定程度上估计出了地面高度。

## 1.&ensp;网络结构
<img src="arch.png" width="90%" height="90%" title="图 2. 网络结构">
　　如图 2. 所示，网络有两部分组成：

- Image-view 通路  
输入为前视图图像，输出相机 pitch 角度 \\(\\theta\\) 以及相机高度 \\(H\\)，这里假设相机坐标系相对地面坐标系没有 roll，yaw 偏转，由此可得到相机外参矩阵，用于 IPM 变换；
- Top-view 通路  
输入为前视图某个特征层经过 Projective Transformation Layer 变换后的特征，之后的特征层叠加来自经过变换的前视图特征层，最后输出车道线检测；

### 1.1.&ensp;Projective Transformation Layer
　　{% post_link lane-det-from-BEV  Apply IPM in Lane Detection from BEV%} 中较详细得阐述了 IPM 原理，{% post_link paper-reading-Deep-Multi-Sensor-Lane-Detection Deep Multi-Sensor Lane Detection%} 则阐述了 STN 的原理。Projective Transformation Layer 类似 STN 的结构，输入相机内外参后，沿用 STN 中的 Grid Generator 以及 Sampler 模块，Grid Generator 就是 IPM 的过程。此外，Projective Transformation Layer 还增加一个卷积层，将前视图的 C 维特征卷积为 C/2 维特征与俯视图的特征层进行叠加。  
　　该层不仅从前视图特征层上产生了俯视图特征，还融合了前视图与俯视图特征层，融合前视图特征有两大好处：

- 瘦高型物体，如栅栏，行人，在俯视图下信息量很小，而前视图能有效提取丰富特征；
- 远距离时，俯视图下的信息会比较稀疏(类似点云)，而前视图信息会比较密集，能有效提取远距离下的信息特征；

### 1.2.&ensp;Anchor-Based Lane Prediction
<img src="anchor.png" width="60%" height="60%" title="图 3. Anchor-Based Lane Prediction">
　　如图 3. 所示，作者提出了一种 Anchor-Based 车道线检测方法，其实这和目标检测中的 Anchor-Based 还是不太一样，这里的 Anchor 指的是几条线。设定 \\(y\\) 方向的 anchor 线段：\\(\\{X_A^i\\} _ {i=1}^N\\)，\\(y\\) 坐标上的预定义位置：\\(\\{y_j\\} _ {j=1}^K\\)。对于每个 anchor 线段，分类上以 \\(Y\_{ref}\\) 为基准，输出三种类别(距离 \\(Y\_{ref}\\) 最近的线的类型)，两种车道中心线，一种车道线，即 \\(\\{c_1,c_2,d\\}\\)；回归上每种类别都输出 2K 个 Offsets：\\(\\{(x_j ^ i,z_j ^ i)\\} _ {j=1}^K\\)，对应的第 \\(i\\) 个 anchor，在第 \\(j\\) 位置上的 3D 点表示为 \\((x_j ^ i+X_A ^ i,y_j,z_j ^ i)\\in\\mathbb{R}^3\\)。综上网络输出 \\(N\\times(3(2K+1))\\) 维的向量，最后经过 1D NMS 处理后，每个 anchor 上的 3D 点通过样条插值出 3D 线条。

## 2.&ensp;Loss
　　训练阶段，真值如何匹配 anchor 很重要，过程如下：

1. 将所有车道线以及车道中心线通过 IPM 投影到俯视图下；
2. 在 \\(Y_{ref}\\) 位置上将每条线匹配给 \\(x\\) 方向距离最近的 anchor 线段；
3. 对于每个 anchor 上匹配到的线，将最左边的车道线与中心线赋为 \\(d,c_1\\)，如果还有其它中心线，则赋为 \\(c_2\\)；

对于没有穿过 \\(Y_{ref}\\) 的车道线，则予以忽略，中心线理论上都会穿过 \\(Y_{ref}\\)。所以理论上，本文预测的中心线是全的，而车道线会不全，前方的岔路口，一部分车道线不会被预测出来。  
　　Loss 项有四部分组成，分别为车道线分类，车道线锚点 Offsets 回归，相机外参 pitch 角 \\(\\theta\\) 以及高度 \\(h_{cam}\\) 的回归，如下：
$$\begin{align}
\mathcal{L} =& - \sum_{t\in\{c_1,c_2,d\}} \sum_{i=1}^N\left(\hat{p}_t^i\mathrm{log}p_t^i + \left(1-\hat{p}_t^i\right)\mathrm{log}\left(1-p_t^i\right)\right) \\
&+ \sum _ {t\in\{c_1,c_2,d\}}\sum_{i=1}^N \hat{p}_t^i\left(\left\Vert x_t^i-\hat{x}_t^i\right\Vert+\left\Vert z_t^i-\hat{z}_t^i\right\Vert\right) \\
&+ \left|\theta-\hat{\theta}\right| + \left|h_{cam}-\hat{h}_{cam}\right| \tag{1}
\end{align}$$

## 3.&ensp;参考文献
<a id="1" href="#1ref">[1]</a> Garnett, Noa, et al. "3D-LaneNet: end-to-end 3D multiple lane detection." Proceedings of the IEEE International Conference on Computer Vision. 2019.  
<a id="2" href="#2ref">[2]</a> {% post_link lane-det-from-BEV  Apply IPM in Lane Detection from BEV%}  
<a id="3" href="#3ref">[3]</a> {% post_link paper-reading-Deep-Multi-Sensor-Lane-Detection Deep Multi-Sensor Lane Detection%}
