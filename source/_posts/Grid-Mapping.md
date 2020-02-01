---
title: Grid Mapping
date: 2020-01-20 10:19:15
tags: ["SLAM", "Point Cloud", "Mapping"]
categories: SLAM
mathjax: true
---

　　占据栅格地图(Occupied Grid Map)是机器人领域一种地图表示方式。可以作为 SLAM 的一个模块，但是这里讨论：**在本体位姿已知的情况下，如何构建 2D Grid Map**。本文介绍两种方法，贝叶斯概率模型以及高斯过程。

## 1.&ensp;贝叶斯概率模型<a href="#1" id="1ref"><sup>[1]</sup></a>
　　设机器人位姿序列为 \\(x_{1:t}\\)，观测序列为 \\(z_{1:t}\\)，那么 Grid Map 的构建就是求解地图的后验概率：\\(p(m|x_{1:t},z_{1:t})\\)，其中地图由栅格构成：\\(m=\\{m_1,m_2,...,m_n\\}\\)。**假设每个栅格独立同分布**，那么：
$$p(m|x_{1:t},z_{1:t})=p(m_1,m_2,...,m_n|x_{1:n},z_{1:t}) = \prod_{i=1}^n p(m_i|x_{1:t},z_{1:t}) \tag{1}$$
　　每个栅格有三种状态：被占有，空，未被观测。设被占有的概率为 \\(occ(m_i) = p(m_i|x_{1:t},z_{1:t})\\)，那么空的概率为 \\(free(m_i)=1-occ(m_i)\\)，对于未被观测的区域认为 \\(occ(m_i) = free(m_i) =0.5\\)。下面通过贝叶斯法则及马尔科夫性推理后验概率计算过程：
$$\begin{align}
occ_t(m_i) &= p(m_i|x_{1:t},z_{1:t}) \\
&= \frac{p(z_t|m_i,x_{1:t},z_{1:t-1})\,p(m_i|x_{1:t},z_{1:t-1})}{p(z_t|x_{1:t},z_{1:t-1})} \\
&= \frac{p(z_t|m_i,x_{t})\,p(m_i|x_{1:t-1},z_{1:t-1})}{p(z_t|x_{1:t},z_{1:t-1})} \\
&= \frac{p(m_i|z_t,x_{t})\,p(z_t|x_t)\,p(m_i|x_{1:t-1},z_{1:t-1})}{p(m_i|x_t)\,p(z_t|x_{1:t},z_{1:t-1})} \\
&= \frac{p(m_i|z_t,x_{t})\,p(z_t|x_t)\,occ_{t-1}(m_{i})}{p(m_i)\,p(z_t|x_{1:t},z_{1:t-1})} \tag{2}
\end{align}$$
对应的栅格为空的概率为：
$$\begin{align}
free_t(\hat{m}_i) &=\frac{p(\hat{m}_i|z_t,x_{t})\,p(z_t|x_t)\,free_{t-1}(\hat{m}_{i})}{p(\hat{m}_i)\,p(z_t|x_{1:t},z_{1:t-1})} \\
&= \frac{(1-p(m_i|z_t,x_{t}))\,p(z_t|x_t)\,(1-occ_{t-1}(m_{i}))}{(1-p(m_i))\,p(z_t|x_{1:t},z_{1:t-1})} \tag{3}
\end{align}$$
由(2),(3)可得：
$$\frac{occ_t(m_i)}{1-occ_t(m_i)} = \frac{1-p(m_i)}{p(m_i)}\cdot\frac{occ_{t-1}(m_i)}{1-occ_{t-1}(m_i)}\cdot\frac{p(m_i|z_t,x_t)}{1-p(m_i|z_t,x_t)}   \tag{4}$$
将上式进行对数化：
$$lm_i^{t} = lm_i^{t-1} + \mathrm{log}\left(\frac{p(m_i|z_t,x_t)}{1-p(m_i|z_t,x_t)}\right) - \mathrm{log}\left(\frac{p(m_i)}{1-p(m_i)}\right) \tag{5}$$
其中 \\(p(m_i)\\) 表示未观测下其被占有的概率，\\(p(m_i|z_t,x_t)\\) 表示当前观测下其被占有的概率。比如，考虑到激光点云的测量噪声，我们可以假设如果该栅格有点云，那么 \\(p(m_i|z_t,x_t) = 0.9\\)；对于激光点光路经过的栅格区域 \\(p(m_i|z_t,x_t) = 0.02\\)，即 \\(p(\\hat{m}_i|z_t,x_t) = 0.98\\)。  
　　该模型下，每个栅格被占有的概率可以转换为前后相加测量量的过程，实际每个栅格被占有的概率为：
$$occ_t(m_i) = \frac{\mathrm{exp}(lm_i^t)}{1+\mathrm{exp}(lm_i^t)} \tag{6}$$

## 2.&ensp;高斯过程<a href="#2" id="2ref"><sup>[2]</sup></a>
　　以上概率模型有个缺陷，其假设栅格独立。实际上栅格并不是独立的，相邻的栅格有很强的相关性。高斯过程则可以处理时域及空域的概率估计与融合问题。  
　　高斯过程基本理论在 {% post_link Ground-Segmentation-with-Gaussian-Process Ground Segmentation with Gaussian Process%} 中已经有较详细阐述，这里作简要概述。假设有训练集 \\(\\{X_n,y_n\\}_{n=1}^N\\)，那么高斯过程下其符合分布：
$$y_n=f(X_n)+\epsilon, \epsilon\sim \mathcal{N}(0,\sigma^2) \tag{7}$$
对于测试集，则有：
$$f(X^\ast) = \mathcal{N}(\mu,\sigma) \tag{8}$$
高斯过程对测试集的预测结果为：
$$\begin{align}
\mu^\ast &=K(X^\ast,X)(K(X,X)+\sigma_n^2I)^{-1}y\\
\sigma^\ast &=K(X^\ast,X^\ast) - K(X^\ast,X)(K(X,X)+\sigma_n^2I)^{-1}K(X,X^\ast)
\end{align} \tag{9}$$
<img src="GPOM.png" width="60%" height="60%" title="图 1. GPOM">
　　高斯过程占据栅格地图(Gaussian Process Occupancy Maps, GPOM)算法过程如图 1. 所示。\\(\\mathrm{p,r}\\) 分别为机器人位姿以及观测量。基本思想就是根据当前时刻的观测数据，提取出正负样本训练集，然后构建高斯模型，对于未观测到的区域，用高斯模型进行预测；每个栅格的信息通过 BCM<a href="#3" id="3ref"><sup>[3]</sup></a> 进行时序的融合，最终采用 logistic 回归得到每个栅格被占据的概率(贝叶斯概率模型中，代替 BCM 及 logistic 的是 log 函数累加融合并求取概率，这里应该也可以用这种方式实现)。  
　　可见，高斯过程来求解占据栅格地图，**能融合时序及空间信息**，但是效率会比较低，不过除了高斯过程中的矩阵求逆操作，其它操作基本可以并行化处理。代码可参考<a href="#4" id="4ref">[4]</a>。

## 3.&ensp;reference
<a id="1" href="#1ref">[1]</a> Thrun, Sebastian. "Probabilistic robotics." Communications of the ACM 45.3 (2002): 52-57.  
<a id="2" href="#2ref">[2]</a> Yuan, Yijun, Haofei Kuang, and Sören Schwertfeger. "Fast Gaussian Process Occupancy Maps." 2018 15th International Conference on Control, Automation, Robotics and Vision (ICARCV). IEEE, 2018.  
<a id="3" href="#3ref">[3]</a> Tresp, Volker. "A Bayesian committee machine." Neural computation 12.11 (2000): 2719-2741.  
<a id="4" href="#4ref">[4]</a> https://github.com/STAR-Center/fastGPOM

