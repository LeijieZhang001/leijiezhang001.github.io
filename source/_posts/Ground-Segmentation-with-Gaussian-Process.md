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

1. **Polar Grid Map**  
将点云用极坐标栅格地图表示，二维地面估计分解成射线方向的多个一维地面估计；
2. **Line Fitting**  
在每个一维方向，根据梯度大小，作可变数量的线段拟合；
3. **Seed Estimation**  
在半径 \\(B\\) 范围内，如果某个 Grid 绝对高度(Grid 高度定义为该 Grid 内所有点的最小高度，其绝对高度则是与本车传感器所在地面的比较)大于 \\(T_s\\)，那么就将其作为 Seed；
4. **Ground Model Estimation with Gaussian Process**  
采用高斯过程生成每个一维方向 Grid 的地面估计量，这里为了进一步加速，可以删除冗余的 Seed；根据地面估计模型，将满足模型的 Grid 加入 Seed，更新模型，迭代直至收敛，满足模型的 Seed 条件为：
$$\begin{align}
V[z]&\leq  t_{model}\\
\frac{|z_*-\bar{z}|}{\sqrt{\sigma^2_n+V[z]}} &\leq t_{data}
\end{align} \tag{0}$$
5. **Point-wise Segmentation**  
得到地面估计模型后，就得到了每个 Grid 是否为地面的标签量，对于属于地面标签量的 Grid 内的点，与 Grid 高度的相对高度小于 \\(T_r\\)，则认为该点属于地面。

## 2.&ensp;高斯过程
　　步骤四中用高斯过程来估计地面模型，对于每个极射线方向的 Grids，假设有 \\(n\\) 个已经确定是地面的训练集：\\(D=\\{(r _ i,z _ i)\\} _ {i=1}^n\\)。根据高斯过程定义，这些样本的联合概率分布为：
$$p(Z|R)\sim N(\mu,K) \tag{1}$$
其中 \\(R=[r_1,...,r_n]^T\\) 为每个 Grid 的距离量，\\(Z=[z_1,...,z_n]^T\\) 为该 Grid 是否为地面的标签量。\\(\\mu\\) 设计为零，协方差矩阵 \\(K\\) 表示变量之间的关系，由协方差方程与噪音项构成：
$$K(r_i,r_j)=k(r_i,r_j)+\sigma^2_n\delta_{ij}\tag{2}$$
其中当且仅当 \\(i==j\\) 时 \\(\\delta _ {ij} =1\\)。  
　　一般的协方差方程是静态，同向的(stationary, isotropic):
$$k(r_i,r_j)=\sigma_f^2\mathrm{exp}\left(-\frac{(r_i-r_j)^2}{2l^2}\right) \tag{3}$$
其中 \\(\\sigma_f^2\\) 是信号协方差，\\(l\\) 是 length-scale。该方程假设了全空间内 length-scale 的一致性，然而实际上，**越平坦的地面区域，我们需要越大的 length-scale，因为此时该区域对周围区域的概率输出能更大**，所以可进一步设计协方差方程为:
$$k(r_i,r_j)=\sigma_f^2\left(l_i^2\right)^{\frac{1}{4}}\left(l_j^2\right)^{\frac{1}{4}}\left(\frac{l_i^2+l_j^2}{2}\right)^{-\frac{1}{2}}  \mathrm{exp}\left(-\frac{2(r_i-r_j)^2}{l_i^2+l_j^2}\right) \tag{4}$$
其中 \\(l_i\\) 为位置 \\(r_i\\) 的 length-scale。\\(l_i\\) 由该位置距离最近的线段梯度决定(步骤二):
$$l_i=\left\{\begin{array}{l}
a\cdot \mathrm{log}\left(\frac{1}{|g(r_i)|}\right) \,\, if\, |g(r_i)|>g_{def}\\
a\cdot \mathrm{log}\left(\frac{1}{|g_{def}|}\right) \,\, otherwise
\end{array}\tag{5}\right.$$
　　高斯回归预测的过程为，对于测试集 \\(T=(r_\\ast,z_\\ast)\\)，其与训练集的联合概率分布为：
$$\begin{bmatrix}
Z\\
z_\ast\\
\end{bmatrix}\sim
N\left(0,
\begin{bmatrix}
K(R,R) & K(R,r_\ast)\\
K(r_\ast,R) & K(r_\ast,r_\ast)\\
\end{bmatrix}\right)
\tag{6}$$
那么，高斯过程回归预测为：
$$\begin{align}
\bar{z}_\ast &=K(r_\ast,R)K^{-1}Z\\
V[z_\ast] &= K(r_\ast,r_\ast)-K(r_\ast,R)K^{-1}K(R,r_\ast)
\end{align} \tag{7}$$
由此得到测试集的预测量，由式(0)可决定该测试量是否标记为地面，进一步迭代估计地面模型，直至收敛。  
　　需要注意的是，以上我们假设高斯过程的超参数 \\(\\theta=\\{\\sigma_f,a,\\sigma_n\\}\\) 是已知的，实际应用中，可以将超参数设定为经验量，也可以基于训练集用 SGD 学习出一个最优量，这里不做展开。

## 3.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Chen, Tongtong, et al. "Gaussian-process-based real-time ground segmentation for autonomous land vehicles." Journal of Intelligent & Robotic Systems 76.3-4 (2014): 563-582.  
