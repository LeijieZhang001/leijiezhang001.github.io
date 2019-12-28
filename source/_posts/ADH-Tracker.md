---
title: ADH(Annealed Dynamic Histograms) Tracker
date: 2019-12-24 08:57:49
tags: ["MOT", "Point Cloud", "tracking"]
categories: MOT
mathjax: true
---

　　{% post_link 卡尔曼滤波详解 卡尔曼滤波详解%}中详细推导了卡尔曼滤波及其扩展卡尔曼滤波基于贝叶斯的推导过程。由贝叶斯法则式(7)，**状态估计问题可定义为：已知似然及先验概率，最大化后验概率的过程**。其中先验即为“运动学模型(motion model)”，似然即为“观测”，后验概率即为待估计的状态量。对于卡尔曼滤波，对应了式(1)的运动方程及测量方程。  
　　用扩展卡尔曼滤波来估计目标状态的原理可见{% post_link 卡尔曼滤波器在三维目标状态估计中的应用 卡尔曼滤波器在三维目标状态估计中的应用%}。该文重点讨论基于质点的一系列运动学模型，以及基于刚体的前转向车模型；测量模型则没做深入研究，默认是目标重心级别的测量量。比如，观测量如果是三维框，那么自然可得到目标的位置，相减就得到速度的观测量。  
　　但是基于点云的目标检测中，目标的观测量更准确的应该是点集(cluster)。**如何在贝叶斯框架下，定义点集的运动学模型及观测模型**，对提高目标状态的估计显得尤其重要。ADH Tracker<a href="#1" id="1ref"><sup>[1]</sup></a> 就是一种点集状态估计方法，其描述了一种可跟踪目标表面形状特性的概率模型，本文主要阐述 ADH Tracker 的原理及实现细节。

## 1.&ensp;点集状态估计的概率模型

### 1.1.&ensp;贝叶斯框架
<img src="bayesian.png" width="50%" height="50%" title="图 1. 点集状态估计的贝叶斯概率模型">
　　如图 1. 所示，状态量为 \\(x_t\\)，点集状态为 \\(s_t\\)，测量/观测量为 \\(z_t\\)，\\(s_t\\) 表示为从目标点集中采样的点集。
<img src="gaussian.png" width="50%" height="50%" title="图 2. 传感器噪声">
　　如图 2. 所示，由于传感器的噪声 \\(\\Sigma_e\\)，实际的目标上的点集 \\(s_t\\) 需要加上传感器噪声，以及目标的当前位置，才是最终的观测量点集 \\(z_t\\):
$$z_{t,j} \sim \mathcal{N}(s_{t,j},\Sigma_e) + x_{t,p}  \tag{1}$$
注意坐标系是在前一时刻目标的中心，状态量中的位置是相对位置，所以前一时刻目标点服从分布：
$$z_{t-1,i} \sim \mathcal{N}(s_{t-1,i},\Sigma_e)  \tag{2}$$
图 1. 的贝叶斯模型下：
$$p(z_{t-1}|x_t,s_{t-1}) = p(z_{t-1}|s_{t-1}) \tag{3}$$
由于目标的遮挡等位置变换，目标上的点集 \\(s_t\\) 又是随时间变化的，假设 \\(p(V)\\) 表示当前时刻点集从前一时刻点集采样的先验概率，那么当前时刻每个点从前一时刻采样的概率为：
$$p(s_{t,j}|s_{t-1}) = p(V)p(s_{t,j}|s_{t-1},V) + p(\neg V)p(s_{t,j}|s_{t-1},\neg V) \tag{4}$$
假设当前点在前一时刻不可见的均为被遮挡的情况，那么：
$$p(s_{t,j}|s_{t-1},\neg V) = k_1(k_2-(s_{t,j}|s_{t-1},V))$$
合并可得：
$$p(s_{t,j}|s_{t-1}) = \eta(p(s_{t,j}|s_{t-1},V) +k) \tag{5}$$

### 1.2.&ensp;状态估计问题
　　式(1)~(5)描述了该贝叶斯网络下各变量之间的关系，状态估计求解的目标是：在所有观测量的基础上估计当前状态，即\\(p(x_t|z_1...z_t)\\)。根据贝叶斯法则：
$$p(x_t|z_1...z_t)=\eta\; p(z_t|x_t,z_1...z_{t-1}) p(x_t|z_1...z_{t-1}) \tag{6}$$
其中 \\(\\eta\\) 为归一化常数，**第一项是观测模型，第二项是运动模型**。如果依据条件独立，观测模型则可简化为：
$$p(z_t|x_t,z_1...z_{t-1}) = p(z_t|x_t)$$
但是这里考虑到 \\(s_t\\) 均是从同一目标采样的，所以条件独立性不成立，将观测模型简化近似为：
$$p(z_t|x_t,z_1...z_{t-1}) \approx p(z_t|x_t,z_{t-1}) \tag{7}$$
直观上理解为，当前观测不仅依赖当前状态，还依赖上一时刻的观测量。

## 2.&ensp;ADH Tracker 观测模型
　　观测模型式(7)可重写为：
$$\begin{align}
p(z_t|x_t,z_{t-1}) &= \int p(z_t,s_t|x_t,z_{t-1})ds_t \\
&= \int p(z_t|s_t,x_t)p(s_t|x_t,z_{t-1})ds_t \\
&= \int p(z_t|s_t,x_t)\left(\int p(s_t,s_{t-1}|x_t,z_{t-1})ds_{t-1}\right)ds_t \\
&= \int p(z_t|s_t,x_t)\left(\int p(s_t|s_{t-1})p(s_{t-1}|x_t,z_{t-1})ds_{t-1}\right)ds_t \\
&= \int p(z_t|s_t,x_t)\left(\int \eta\;p(s_t|s_{t-1})p(z_{t-1}|x_t,s_{t-1})p(s_{t-1})ds_{t-1}\right)ds_t \\
&= \int p(z_t|s_t,x_t)\left(\int \eta\;p(s_t|s_{t-1})p(z_{t-1}|s_{t-1})p(s_{t-1})ds_{t-1}\right)ds_t
\tag{8}
\end{align}$$
式(1)(2)(5)可得高斯模型:
$$\left\{\begin{array}{l}
p(z_t|s_t,x_t) = \mathcal{N}(z_t;s_t+x_{t,p},\Sigma_e) \\
p(z_{t-1}|s_{t-1}) = \mathcal{N}(z_{t-1};s_{t-1},\Sigma_e) \\ 
p(s_t|s_{t-1}) = \eta\left(\mathcal{N}(s_{t};s_{t-1},\Sigma_r)+k \right) \\ 
\end{array}\tag{9}\right.$$
其中 \\(\\Sigma_e \\) 为传感器噪声方差，\\(\\Sigma_r\\) 为传感器不同距离的分辨率。因为两个高斯分布相乘还是高斯分布，所以由式(8)(9-2)(9-3)，可得：
$$ p(s_t|x_t,z_{t-1}) = \eta (\mathcal{N}(s_t;z_{t-1},\Sigma_r+\Sigma_e)+k) \tag{10}$$
进一步由式(8)(9-1)(10)可得：
$$p(z_t|x_t,z_{t-1}) = \eta \left(\mathcal{N}(z_t;z_{t-1}+x_{t,p},\Sigma_r+2\Sigma_e)+k \right) \tag{11}$$
　　观测模型实际计算中，令 \\(\\bar{z} _ {t-1}\\) 为点集 \\(z_{t-1}\\) 经过状态量变换后的点集，即 \\(\\bar{z} _ {t-1}=z _ {t-1}+x _ {t,p}\\)；对于 \\(z _ j\\in z _ t\\)，令 \\(\\bar{z} _ i \\) 为 \\(z _ j\\) 在点集 \\(\\bar{z}_ { t-1}\\) 中的最近点。那么:
$$ p(z_t|x_t,z_{t-1}) = \eta \left(\prod_{z_j\in z_t} \mathrm{exp}\left(-\frac{1}{2}(z_j-\bar{z_i})^T\Sigma^{-1}(z_j-\bar{z}_i)\right)+k\right) \tag{12}$$
其中 \\(\\Sigma=2\\Sigma_e+\\Sigma_r\\)。

## 3.&ensp;ADH Tracker 运动模型
　　这里使用的是质点匀速模型，因为在 \\((R,t)\\) 搜索空间中得到了一组不同概率的解，所以可用多变量高斯分布去拟合这组解：
$$\left\{\begin{array}{l}
\mu_t=\sum_i p(x_{t,i}|z_i...z_t)x_{t,i}\\
\Sigma_t = \sum_i p(x_{t,i}|z_1...z_t)(x_{t,i}-\mu_t)(x_{t,i}-\mu_t)^T
\end{array}\tag{13}\right.$$
其中 \\(x_{t,i}\\) 为第 \\(i\\) 组解对应的状态量。得到该状态量的高斯分布后，就可以用匀速运动模型预测下一时刻的状态。  
　　同时针对每一组解空间中的候选解，还可计算其匀速模型下的速度概率项，叠加到观测概率中。

## 4.&ensp;ADH 算法
<img src="adh.png" width="60%" height="60%" title="图 3. ADH 原理">
　　对 \\((R,t)\\) 解空间进行有效搜索直接决定求解速度，如图 3. 所示，将解空间(state space)分割成一系列搜索区域，每个区域基于后验概率 \\(p(x_t|z_1...z_t)\\) 计算区域离散概率：
$$\begin{align}
p(c_i) &= p(c_i\cap R) \\
&= p(c_i|R)p(R) \\
&= \frac{p(x_i|z_1...z_t)\vert c_i\vert}{\sum_{j\in R}p(x_j|z_1...z_t)\vert c_i\vert} p(R) \\
&= \eta p(x_i|z_1...z_t)p(R)
\tag{14}
\end{align}$$
其中 \\(R\\) 为待细分的区域集合(cells)，其被划分为子区域 \\(c_i\\in R\\)，所以区域概率满足 \\(\\sum_{i\\in R}p(c_i) = p(R)\\)。对拥有较大离散概率的区域，进一步细分搜索区域，进行迭代搜索。初始化时，\\(p(R)=1\\)。  
　　这里需要制定区域细分的策略，考虑最大化划分前后区域概率分布的 KL-divergence，即 KL-divergence 能描述划分后，后验概率与真实分布的相似性，越接近真实分布，前后区域离散概率分布的 KL-divergence 会越小。而为了提高搜索效率，要求前后离散概率分布的 KL-divergence 要最大，最终收敛到真实分布。  
　　假设 \\(R\\) 区域的离散概率分布为 \\(P_i\\)，需要划分 \\(k\\) 个区域。那么划分前，可以认为其概率分布为每个 cell 概率为 \\(P_i/k\\)；划分后，其概率分布为：\\(\\sum_{j=1}^kp_j=P_i\\)。这两个分布的 KL-divergence 为：
$$ D_{KL}(A\Vert B)=\sum_{j=1}^k p_j \mathrm{In}\left(\frac{p_j}{P_i/k}\right) \tag{15} $$
当某个细分区域 \\(p_{j'} = P_i\\) 时：
$$ D_{KL}(A\Vert B)=P_i \mathrm{In}k  \tag{16}$$
如果每个 cell 后验概率计算需要时间 \\(t\\) 秒，那么每秒能获得最大的 DL-divergence 为 \\(P_i\\mathrm{In}k/(kt)\\)，由此可以选择策略：

- 对 \\(P_i\\) 大于一定阈值的区域进行划分；
- 每个搜索维度划分的区域个数设定为 \\(k=3\\)。因为该函数在 \\(k=e\\) 时取得最大值。

<img src="adh_alg2.png" width="80%" height="80%" title="图 4. ADH Tracker">
　　图 4. 为 ADH Tracker 算法的伪代码。

## 5.&ensp;ADH Tracker 实现细节<a href="#2" id="2ref"><sup>[2]</sup></a>
### 5.1.&ensp;Kalman 部分
　　ADH 代码中 centroid-based kalman 的运动模型为质点匀速模型，较为简单。 其设置为：状态量 \\(x=[v_x,v_y,v_z]\\)，测量量 \\(z=\\frac{1}{\\delta t}[p_x,p_y,p_z]\\)。状态转移矩阵 \\(A\\) 以及观测矩阵 \\(C\\) 均为单位阵。过程噪声为高斯分布，其协方差矩阵为 \\(Q_k = diag(\\sigma_a,\\sigma_a,\\sigma_{a_z})\\cdot \\delta ^2 t\\)，测量噪声的协方差矩阵为 \\(R_k = diag(\\sigma_m,\\sigma_m,\\sigma_m)\\)。由此可方便的计算 kalman 预测及更新两个过程。

### 5.2.&ensp;ADH 部分
　　ADH 算法中，每个采样分辨率下需要多次计算解空间中各 \\((R,t)\\) 下的观测模型，而观测模型计算中，每次需要通过 KD-Tree 寻找两个点集的匹配点，再通过式(12)计算观测概率模型。这样会非常耗时，因为观测模型本质上就是求解两个点集相似度，所以代码实现中，作者采用的策略为：首先将被匹配的点集进行栅格化，然后将点集中每个点以稠密度(density)高斯概率分布的形式拓展一定栅格范围，每个栅格取拓展到该栅格的点的高斯概率值的最大值。之后任何一个点集需要与之计算观测模型(相似度)，只要直接统计索引这个点集在该栅格下的概率值即可。


## 6.&ensp;参考文献
<a id="1" href="#1ref">[1]</a> Held, David, et al. "Robust real-time tracking combining 3D shape, color, and motion." The International Journal of Robotics Research 35.1-3 (2016): 30-49.  
<a id="2" href="#2ref">[2]</a> https://github.com/davheld/precision-tracking
