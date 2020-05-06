---
title: '[paper_reading]-"Probabilistic 3D Multi-Object Tracking for Autonomous Driving"'
date: 2020-04-07 17:28:57
tags: ["MOT", "autonomous driving", "Uncertainty", "paper reading"]
categories: MOT
mathjax: true
---

　　{% post_link 卡尔曼滤波器在三维目标状态估计中的应用 卡尔曼滤波器在三维目标状态估计中的应用%}中已经较详细得阐述了 3D MOT 状态估计过程，文章末提到观测过程的协方差矩阵初始化问题可以用观测的不确定性解决，{% post_link Heteroscedastic-Aleatoric-Uncertainty Heteroscedastic Aleatoric Uncertainty%} 就是通过贝叶斯深度神经网络来建模该不确定性。本文<a href="#1" id="1ref"><sup>[1]</sup></a> 提供了另一种简化的观测不确定性计算方法，同时估计运动模型与观测模型的不确定性，即过程噪声与测量噪声。

## 1.&ensp;Kalman Filter
<img src="framework.png" width="80%" height="80%" title="图 1. MOT Framework">
　　如图 1. 所示，本文采用的卡尔曼滤波框架与传统的一样，分为预测与更新。预测阶段，根据上一时刻结果通过 Motion Model(Process Model) 预测当前时刻的状态(先验)；数据关联阶段，将预测的状态与观测的状态作目标数据关联，出 ID；更新阶段，融合预测与观测的状态，得到状态的后验估计。  

### 1.1.&ensp;Predict Step
　　本文采用 CTRV(Constant Turn Rate and Velocity) 运动模型。不同与{% post_link 卡尔曼滤波器在三维目标状态估计中的应用 卡尔曼滤波器在三维目标状态估计中的应用%}中描述的 CTRV，本文作了**线性简化**，其运动方程为：
$$\begin{align}
&\begin{bmatrix}
\hat{x}\\
\hat{y}\\
\hat{z}\\
\hat{a}\\
\hat{l}\\
\hat{w}\\
\hat{h}\\
\hat{d} _ x\\
\hat{d} _ y\\
\hat{d} _ z\\
\hat{d} _ a\\
\end{bmatrix} _ {t+1}=
\begin{bmatrix}
1 &0 &0 &0 &0 &0 &0 &1 &0 &0 &0\\
0 &1 &0 &0 &0 &0 &0 &0 &1 &0 &0\\
0 &0 &1 &0 &0 &0 &0 &0 &0 &1 &0\\
0 &0 &0 &1 &0 &0 &0 &0 &0 &0 &1\\
0 &0 &0 &0 &1 &0 &0 &0 &0 &0 &0\\
0 &0 &0 &0 &0 &1 &0 &0 &0 &0 &0\\
0 &0 &0 &0 &0 &0 &1 &0 &0 &0 &0\\
0 &0 &0 &0 &0 &0 &0 &1 &0 &0 &0\\
0 &0 &0 &0 &0 &0 &0 &0 &1 &0 &0\\
0 &0 &0 &0 &0 &0 &0 &0 &0 &1 &0\\
0 &0 &0 &0 &0 &0 &0 &0 &0 &0 &1\\
\end{bmatrix}
\begin{bmatrix}
x\\
y\\
z\\
a\\
l\\
w\\
h\\
d _ x\\
d _ y\\
d _ z\\
d _ a\\
\end{bmatrix} _ {t}  +
\begin{bmatrix}
q _ x\\
q _ y\\
q _ z\\
q _ a\\
0\\
0\\
0\\
q _ {d _ x}\\
q _ {d _ y}\\
q _ {d _ z}\\
q _ {d _ a}\\
\end{bmatrix} _ {t}\\
\Longleftrightarrow & \\
&\hat{\mu} _ {t+1} = \mathbf{A}\mu _ t \\
\end{align}\tag{1}
$$
其中未知的线加速度与角加速度 \\((q _ x, q _ y, q _ z, q _ a)\\)，\\((q _ {d _ x},q _ {d _ y},q _ {d _ z},q _ {d _ a})\\) 符合\\((0,\\mathbf{Q})\\)高斯分布。  
　　根据 Motion Model，卡尔曼的预测过程计算状态量的先验：
$$\begin{align}
\hat{\mu} _ {t+1} &= \mathbf{A}\mu _ t \\
\hat{\Sigma} _ {t+1} &= \mathbf{A}\Sigma _ t\mathbf{A}^T + \mathbf{Q}\\
\end{align}\tag{2}$$
　　观测模型为每一时刻检测的结果，包括位置，朝向，目标框尺寸，即观测矩阵 \\(\\mathbf{H} _ {7\\times 11} = [\\mathbf{I}, \\mathbf{0}]\\)。观测噪声也符合高斯分布，由此得到预测的观测量：
$$\begin{align}
\hat{o} _ {t+1} &= \mathbf{H}\hat{\mu} _ {t+1} \\
\mathbf{S} _ {t+1} &= \mathbf{H}\hat{\Sigma} _ {t+1}\mathbf{H}^T + \mathbf{R}\\
\end{align}\tag{3}$$

### 1.2.&ensp;Update Step
　　首先将预测的观测量与实际的观测量作数据关联。基本思想是将预测目标与观测目标作 Cost Matrix，然后用匈牙利/贪心算法求解最优匹配对。本文采用 Mahalanobis distance：
$$ m = \sqrt{(o _ {t+1}- \mathbf{H}\hat{\mu} _ {t+1})^T\mathbf{S} _ {t+1} ^{-1}(o _ {t+1}-\mathbf{H}\hat{\mu} _ {t+1})} \tag{4}$$
需要注意的是，计算距离前先做角度矫正，如果两个目标框角度相差大于 90 度，那么作 180 度旋转。  
　　得到预测与观测的匹配对后，计算后验概率更新该目标的状态：
$$\begin{align}
\mathbf{K} _ {t+1} &= \hat{\Sigma} _ {t+1}\mathbf{H} ^T\mathbf{S} _ {t+1}^{-1}\\
\mu _ {t+1} &= \hat{\mu} _ {t+1} + \mathbf{K} _ {t+1}(o _ {t+1}-\mathbf{H}\hat{\mu} _ {t+1})\\
\Sigma _ {t+1} &=(\mathbf{I}-\mathbf{K} _ {t+1}\mathbf{H})\hat{\Sigma} _ {t+1}\\
\end{align}\tag{5}$$
　　以上卡尔曼过程与{% post_link 卡尔曼滤波器在三维目标状态估计中的应用 卡尔曼滤波器在三维目标状态估计中的应用%}，以及{% post_link 卡尔曼滤波详解 卡尔曼滤波详解%}完全一致。

## 2.&ensp;Covariance Matrices Estimation
　　如何确定卡尔曼滤波过程中的 \\(\\Sigma _ 0, \\mathbf{Q, R}\\)？传统方法是直接用一个确定的经验矩阵赋值；理想的是用{% post_link Heteroscedastic-Aleatoric-Uncertainty Heteroscedastic Aleatoric Uncertainty%} 建模处理，但是会相对较复杂；本文用更简单的基于统计方法来确定协方差矩阵。  
　　**观测量的方差(不确定性)与目标的属性有关**，如距离，遮挡，类别等。本文没有区分这些属性，只统计了一种观测量的方差，**更好的处理方式是按照不同属性，统计不同的方差**。而 {% post_link Heteroscedastic-Aleatoric-Uncertainty Heteroscedastic Aleatoric Uncertainty%} 是 Instance 级别的方差预测。**这种统计出来的方差虽然细粒度差一点，但是非常合理，因为只要模型训练好后，模型预测的分布是与训练集分布相似的(理想情况)**，所以用训练集的方差来直接代替模型预测的方差也较为合理。  
　　<span style="color:red">**更准确的来说，不确定性与物体的属性以及标注误差有关，这里只统计了标注误差(标注误差在大多数情况下都是同分布的)，而实际上遮挡大的目标，是更难学习的(目标学习有难易之分，即预测分布与训练集分布会有偏差)，即预测结果会有额外量的不确定性，所以这种离线统计方法也有很大的局限性**。</span>  
　　设训练集的真值标签：\\(\\left\\{\\left\\{x _ t^m, y _ t^m, z _ t^m, a _ t^m\\right\\} _ {m=1}^M\\right\\} _ {t = 1}^T\\)。  

## 2.1.&ensp;Motion/Process Noise Model
　　假设各状态量的噪声独立同分布，那么对于位置与朝向噪声，有：
$$\begin{align}
Q _ {xx} &= \mathbf{Var}\left(\left(x _ {t+1}^m-x _ t^m\right)-\left(x _ t^m-x _ {t-1}^m\right)\right)\\
Q _ {yy} &= \mathbf{Var}\left(\left(y _ {t+1}^m-y _ t^m\right)-\left(y _ t^m-y _ {t-1}^m\right)\right)\\
Q _ {zz} &= \mathbf{Var}\left(\left(z _ {t+1}^m-z _ t^m\right)-\left(z _ t^m-z _ {t-1}^m\right)\right)\\
Q _ {aa} &= \mathbf{Var}\left(\left(a _ {t+1}^m-a _ t^m\right)-\left(a _ t^m-a _ {t-1}^m\right)\right)\\
\end{align}\tag{6}$$
　　对于线速度与角速度，因为：
$$\begin{align}
q _ {x _ t} &\approx x _ {x+1} - x _ t - d _ {x _ t}\\
& \approx (x _ {t+1}-x _ t) - (x _ t-x _ {t-1})\\
q _ {d _ {x _ t}} &\approx d _ {x _ {t+1}} - d _ {x _ t}\\
& \approx (x _ {t+1}-x _ t) - (x _ t-x _ {t-1})\\
\end{align}\tag{7}$$
所以：
$$ (Q _ {d _ xd _ x}, Q _ {d _ yd _ y}, Q _ {d _ zd _ z}, Q _ {d _ ad _ a}) = (Q _ {xx}, Q _ {yy}, Q _ {zz}, Q _ {aa})\tag{8}$$

## 2.2.&ensp;Observation Noise Model
　　在训练集上，找到检测与真值的匹配对 \\(\\left\\{\\left\\{(D _ t^k, G _ t^k)\\right\\} _ {k=1}^K\\right\\} _ {t=1}^T\\)，从而计算观测噪声：
$$\begin{align}
&R _ {xx} = \mathbf{Var}\left(D _ {x _ t}^k-G _ {x _ t}^k\right)\\
&R _ {yy} = \mathbf{Var}\left(D _ {y _ t}^k-G _ {y _ t}^k\right)\\
&R _ {zz} = \mathbf{Var}\left(D _ {z _ t}^k-G _ {z _ t}^k\right)\\
&R _ {aa} = \mathbf{Var}\left(D _ {a _ t}^k-G _ {a _ t}^k\right)\\
&R _ {ll} = \mathbf{Var}\left(D _ {l _ t}^k-G _ {l _ t}^k\right)\\
&R _ {ww} = \mathbf{Var}\left(D _ {w _ t}^k-G _ {w _ t}^k\right)\\
&R _ {hh} = \mathbf{Var}\left(D _ {h _ t}^k-G _ {h _ t}^k\right)\\
\end{align}\tag{8}$$
初始的状态协方差 \\(\\Sigma _ 0 = \\mathbf{R}\\)。

## 3.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Chiu, Hsu-kuang, et al. "Probabilistic 3D Multi-Object Tracking for Autonomous Driving." arXiv preprint arXiv:2001.05673 (2020).

