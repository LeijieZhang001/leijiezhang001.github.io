---
title: Perception Uncertainty in Deep Learning
date: 2020-11-12 10:32:13
updated: 2020-11-17 09:19:12
tags: ["Deep Learning", "Uncertainty"]
categories:
- Uncertainty
mathjax: true
---

　　基于深度学习感知技术的发展极大推动了自动驾驶行业的落地，然而基于深度学习的感知技术还存在很多问题，比如针对经典的目标检测任务，算法无法保证在所有场景下做到 100% 的检测准确率。由此，针对 L3/L4 自动驾驶，产品落地只能寄期望于 ODD(operational design domain) 的精心设计。要实现真正意义上的 L4/L5 自动驾驶，精准的环境感知很重要，但更重要的是，算法能否给出当前传感器数据及模型感知的不确定性(uncertainty)。换句话说，我们期望算法模型知道什么，但更期望算法模型不知道什么。  
　　另一方面，多传感器融合是 L4/L5 自动驾驶的基础，而传感器及模型的不确定性估计，则又是多传感器后融合的基础。以下以多目标状态估计任务为例，来描述不确定的作用及估计方式。

## 1.&ensp;基于多传感器的目标状态估计
　　这里考虑基于毫米波雷达，相机，激光雷达等三种传感器的目标状态估计后融合方案。假设 3D 毫米波雷达测量的目标属性为 \\(x, y, v\\)；基于深度学习的相机与激光雷达测量的目标属性均为 \\(x, y, z, l, w, h, \\theta\\)。经过前后多目标的数据关联后，这些量通过 KF 或 UKF 进行融合，最终得到目标状态的鲁邦估计。以最简单的 KF 为例，融合的过程需要测量量以及测量的方差。这里的方差就是我们要讨论的不确定性，即假设测量满足高斯分布下对应的方差。  
　　传统的做法是将测量方差设为经验固定值，但是在多传感器融合框架里，这样无法区分 1) 单传感器在不同场景下算法模型的性能；2) 多传感器在同一场景下算法模型的性能。由此，基于深度学习的感知测量输出需要同时估计测量值的不确定性，以此作更鲁棒的状态估计。  

## 2.&ensp;不确定性概述
　　不确定性由传感器及模型产生。传感器方面，比如激光雷达的点云测量存在厘米级别的误差。模型方面，在贝叶斯框架下，可以建模两大类不确定性[2]

- 认知不确定性(Epistemic Uncertainty)，描述模型因为缺少训练数据而存在的未知，可通过增加训练数据解决；
- 偶然不确定性(Aleatoric Uncertainty)，描述数据不能解释的信息，可通过提高数据的精度来消除，与传感器相关；
    - 数据依赖/异方差不确定性(Data-dependent/Heteroscedastic Uncertainty)，与模型输入数据有关，可作为模型预测输出；
    - 任务依赖/同方差不确定性(Task-dependent/Homoscedastic Uncertainty)，与模型输入数据无关，且不是模型的预测输出，不同任务有不同的值；

　　认知不确定性多应用于 Active Learning，数据可标注性等领域，能有效挖掘提升模型感知能力的数据。同方差不确定性多应用于神经网络的多任务学习，能根据各个任务对应的数据方差学习其损失函数权重，提升所有任务的整体性能。异方差不确定性是反映在线网络预测量是否可信的主要不确定性。

## 3.&ensp;贝叶斯深度学习中不确定性的数学描述
　　针对一批训练数据集\\(\\{\\mathbf{X,Y}\\}\\)，训练模型 \\(\\mathbf{y=f^W(x)}\\)，在贝叶斯框架下，预测量的后验分布为<a href="#3" id="3ref"><sup>[3]</sup></a>：
$$p\left(\mathbf{y\vert x,X,Y}\right) = \int p\left(\mathbf{y\,|\,f^W(x)}\right) p\left(\mathbf{W\,|\,X,Y}\right)d\mathbf{W} \tag{1}$$
其中 \\(p(\\mathbf{W\\,|\\,X,Y})\\) 为模型参数的后验分布，描述了模型的不确定性，即认知不确定性；\\(p\\left(\\mathbf{y\\,|\\,f^W(x)}\\right)\\) 为观测似然，描述了观测不确定性，即偶然不确定性。  

### 3.1.&ensp;认知不确定性
　　认知不确定性表征的是模型的不确定性，对于训练集 \\(\\mathrm{D}=\\{\\mathbf{X},\\mathbf{Y}\\}\\)，认知不确定性即为权重参数的分布 \\(p(\\mathbf{\\omega} | \\mathbf{X},\\mathbf{Y})\\)。可采用 Monte-Carlo 采样方法来近似估计模型权重分布:
$$p(\omega|\mathbf{X},\mathbf{Y})\approx q(\mathbf{\omega};\mathbf{\Phi})=Bern(\mathbf{\omega};\mathbf{\Phi}) \tag{9}$$
其中 \\(\\mathbf{\\Phi}\\) 是 Bernolli Rates，具体的采样通过 Dropout 实现。在训练阶段，Dropout 等价于优化网络权重 \\(W\\) 的 Bernoulli 分布；在测试阶段，使用 Dropout 对样本进行多次测试，能得到模型权重的后验分布，由此模型的不确定性即为 T 次采样的方差：
$$\mathbf{Var} _ {p(\mathbf{y}|\mathbf{x})} ^ {model}(\mathbf{y})=\sigma _ {model} = \frac{1}{T}\sum _ {t=1} ^ T(\mathbf{y} _ t-\bar{\mathbf{y}}) ^ 2\tag{10}$$
其中 \\(\\{\\mathbf{y} _ t\\} _ {t=1} ^ T\\) 是不同权重 \\(\\omega ^ t\\sim q(\\omega;\\mathbf{\\Phi})\\) 采样下的输出。  

　　这种模型不确定性的计算方式，直观的理解为：当模型对某些数据预测比较好，误差比较小的时候，那么模型对这些数据的冗余度肯定是较高的，所以去掉模型的一部分网络，模型对这些数据的预测与原模型应该会有较高的一致性，即不确定性会较小。

### 3.2.&ensp;偶然不确定性

　　偶然不确定性估计是最大化高斯似然过程。对于回归任务，定义模型输出为高斯分布：
$$p\left(\mathbf{y}\vert\mathbf{f^W(x)}\right) = \mathcal{N}\left(\mathbf{f^W(x)}, \sigma ^2\right) \tag{1}$$
其中 \\(\\sigma\\) 为观测噪声方差，描述了模型输出中含有多大的噪声。对于分类任务，玻尔兹曼分布下的模型输出概率分布为：
$$p\left(\mathbf{y}\vert\mathbf{f^W(x)},\sigma\right) = \mathrm{Softmax}\left(\frac{1}{\sigma ^2}\mathbf{f^W(x)}\right) \tag{2}$$
由此对于多任务，模型输出的联合概率分布为：
$$p\left(\mathbf{y}_1,\dots,\mathbf{y}_K\vert\mathbf{f^W(x)}\right) = p\left(\mathbf{y}_1\vert\mathbf{f^W(x)}\right) \dots p\left(\mathbf{y}_K\vert\mathbf{f^W(x)}\right) \tag{3}$$

　　对于回归任务，\\(log\\)似然函数：
$$\mathrm{log}p\left(\mathbf{y}\vert\mathbf{f^W(x)}\right) \propto -\frac{1}{2\sigma ^2} \Vert \mathbf{y-f^W(x)} \Vert ^2 - \mathrm{log}\sigma \tag{4}$$
对于分类任务，\\(log\\)似然函数：
$$\mathrm{log}p\left(\mathbf{y}=c\vert\mathbf{f^W(x)}, \sigma\right) = \frac{1}{2\sigma ^2}f_c^{\mathbf{W}}(\mathbf{x})- \mathrm{log}\sum_{c'} \mathrm{exp}\left(\frac{1}{\sigma^2}f^{\mathbf{W}}_{c'}(\mathbf{x}) \right) \tag{5}$$

　　最大化高斯似然，等价于最小化其负对数似然函数。现同时考虑回归与分类任务，则多任务的联合 \\(Loss\\)：
$$\begin{align}
\mathcal{L}(\mathbf{W}, \sigma _1, \sigma _2) &= -\mathrm{log}p\left(\mathrm{y_1,y_2}=c\vert\mathbf{f^W(x)} \right) \\
&= -\mathrm{log}\mathcal{N}\left(\mathbf{y_1};\mathbf{f^W(x)}, \sigma_1^2\right) \cdot \mathrm{Softmax}\left(\mathbf{y_2}=c;\mathbf{f^W(x)},\sigma_2\right) \\
&= \frac{1}{2\sigma_1^2}\Vert \mathbf{y}_1-\mathbf{f^W(x)}\Vert ^2 + \mathrm{log}\sigma_1 - \mathrm{log}p\left(\mathbf{y}_2=c\vert\mathbf{f^W(x)},\sigma_2\right) \\
&= \frac{1}{2\sigma_1^2}\mathcal{L}_1(\mathbf{W}) +\frac{1}{\sigma_2^2}\mathcal{L}_2(\mathbf{W}) + \mathrm{log}\sigma_1 + \mathrm{log}\frac{\sum_{c'}\mathrm{exp}\left(\frac{1}{\sigma_2^2}f_{c'}^{\mathbf{W}}(x)\right)}{\left(\sum_{c'}\mathrm{exp}\left(f_{c'}^{\mathbf{W}}(x) \right) \right)^{\frac{1}{\sigma_2^2}}} \\
&\approx \frac{1}{2\sigma_1^2}\mathcal{L}_1(\mathbf{W}) +\frac{1}{\sigma_2^2}\mathcal{L}_2(\mathbf{W}) + \mathrm{log}\sigma_1 + \mathrm{log}\sigma_2 \tag{6}
\end{align}$$

由此可见，分类及回归的偶然不确定性估计，可通过额外预测对应的方差，并将方差通过上式作用于损失函数实现。实际应用中，为了数值稳定，令 \\(s:=\\mathrm{log}\\sigma^2\\):
$$\mathcal{L}(\mathbf{W}, s_1, s_2) = \frac{1}{2}\mathrm{exp}(-s_1)\mathcal{L}_1(\mathbf{W}) + \mathrm{exp}(-s_2)\mathcal{L}_2(\mathbf{W}) + \mathrm{exp}(\frac{1}{2}s_1) + \mathrm{exp}(\frac{1}{2}s_2) \tag{7}$$



## 4.&ensp;不确定性估计方法

### 4.1.&ensp;统计法
　　观测量的方差(不确定性)与目标的属性有关，如距离，遮挡，类别等。可以按照不同属性，统计不同的方差。这种统计出来的方差实际上就是在特定传感器精度下，标注的不确定性，比如随着距离越远点云越稀少，标注误差也会越大。这样统计出来的方差与实际网络输出的不确定性不是等价的，但是只要模型训练好后，模型预测的分布是与训练集分布相似的，所以用训练集的方差来直接代替模型预测的方差也合理。  
　　但是更准确的来说，不确定性对每个目标都应该是不同的，这里只统计了特定属性以及标注误差所产生的不确定性，而实际上遮挡大的目标，是更难学习的(目标学习有难易之分，即预测分布与训练集分布会有偏差)，即预测结果会有额外的不确定性，所以这种离线统计方法也有很大的局限性。

### 4.2.&ensp;网络分支预测法

　　假设网络输出符合多变量混合高斯分布，将偶然不确定性设计为网络的输出，
$$\left\{\begin{array}{l}
p\left(\mathbf{y}\vert\mathbf{f^W(x)}\right) = \sum_k \alpha_k \mathcal{N}\left(\mathbf{f^W(x)}_{(k)}, \Sigma(\mathbf{x})_{(k)} \right)\\
\sum_k \alpha_k = 1
\end{array}\tag{5}\right.$$
　　对于 3D Detection 问题，网络输出的 3D 框参数为 \\(\\mathbf{y}=\(x,y,z,l,h,w,\\theta\)\\)，当输出满足 \\(K\\) 个混合高斯分布时，网络的输出量有：

- \\(K\\) 组目标框参数预测量 \\(\\{\\mathbf{y}_k\\}\\)；
- \\(K\\) 个对数方差 \\(\\{s_k\\}\\)；
- \\(K\\) 个混合高斯模型权重参数 \\(\\{\\alpha_k\\}\\)；

　　训练时，找出与真值分布最近的一组预测量，混合高斯模型权重用 softmax 分类，找到最相似的分布后，将该分布的方差用式(3)作用于回归的 Loss 项；测试时，找到混合高斯模型最大的权重项，对应的高斯分布，即作为最终的输出分布。这里只考虑了输出 3D 框的一个整体的方差，也可以输出定位方差+尺寸方差+角度方差，只要将该方差作用于对应的 Loss 项即可。当 \\(K=1\\) 时，就是多变量单高斯模型。


### 4.3.&ensp;Assumed Density Filtering(ADF) 估计法 
　　假设传感器得到的数据符合噪音水平 \\(\\mathbf{v}\\) 的高斯分布，那么输入网络的数据 \\(\\mathbf{z}\\) 与其真实数据 \\(\\mathbf{x}\\) 的关系为：
$$q(\mathbf{z}|\mathbf{x})\sim \mathcal{N}(\mathbf{z};\mathbf{x},\mathbf{v})\tag{1}$$
为了计算网络预测量的不确定性，通过 Assumed Density Filtering(ADF) 来传递输入数据的噪音，从而计算模型的偶然不确定性。网络的联合概率分布为：
$$p(\mathbf{z}^{(0:l)})=p(\mathbf{z}^{(0)})\prod _ {i=1} ^ l p(\mathbf{z}^{(i)}|\mathbf{z} ^ {(i-1)}) \tag{2}$$
其中：
$$p(\mathbf{z}^{(i)}|\mathbf{z}^{(i-1)})=\sigma[\mathbf{z} ^ {(i)}-\mathbf{f} ^ {(i)}(\mathbf{z}^{(i-1)})]\tag{3}$$
ADF 将其近似为：
$$p(\mathbf{z}^{(0:l)})\approx q(\mathbf{z}^{(0:l)})=q(\mathbf{z}^{(0)})\prod _ {i=1} ^ l q(\mathbf{z}^{(i)}) \tag{4}$$
其中 \\(q(\\mathbf{z})\\) 符合独立高斯分布：
$$q(\mathbf{z}^{(i)})\sim \mathcal{N}\left(\mathbf{z}^{(i)};\mathbf{\mu}^{(i)},\mathbf{v}^{(i)}\right)=\prod _ j\left(\mathbf{z} _ j ^ {(i)};\mathbf{\mu} _ j ^ {(i)}, \mathbf{v} _ j ^ {(i)}\right)\tag{5}$$
特征 \\(\\mathbf{z}^{(i-1)}\\) 通过第 \\(i\\) 层映射方程 \\(\\mathbf{f} ^{(i)}\\)，得到：
$$\hat{p}(\mathbf{z}^{(0:i)})=p(\mathbf{z}^{(i)}|\mathbf{z}^{(i-1)})q(\mathbf{z}^{(0:i-1)}) \tag{6}$$
ADF 的目标是找到 \\(\\hat{p}(\\mathbf{z}^{(0:i)})\\) 的近似分布 \\(q(\\mathbf{z}^{(0:i)})\\)，比如 KL divergence：
$$q(\mathbf{z}^{(0:i)})=\mathop{\arg\min}\limits _ {\hat{q}(\mathbf{z}^{(0:i)})}\mathbf{KL}\left(\hat{q}(\mathbf{z}^{(0:i)})\;\Vert\;\hat{p}(\mathbf{z}^{(0:i)})\right)\tag{7}$$
基于高斯分布的假设，误差通过每层的映射方程 \\(\\mathbf{f} ^{(i)}\\) 进行独立传播，故以上解为：
$$\begin{align}
\mathbf{\mu}^{(i)}=\mathbb{E} _ {q(\mathbf{z}^{(i-1)})}[\mathbf{f}^{(i)}(z^{(i-1)})]\\
\mathbf{v}^{(i)}=\mathbb{V} _ {q(\mathbf{z}^{(i-1)})}[\mathbf{f}^{(i)}(z^{(i-1)})]\\
\end{align}\tag{8}$$
以映射方程 \\(\\mathbf{f} ^{(i)}\\) 为卷积层为例，均值传递就是正常的卷积操作，方差传递则需要将卷积权重平方，然后作卷积操作。其它神经网络层都可推导出对应的方差传递方程。  
　　结合蒙特卡洛采样计算认知不确定性与 ADF 方法计算偶然不确定性，网络预测的结果与对应的总的不确定性可计算为：
$$\left\{\begin{array}{l}
\mu = \frac{1}{T}\sum _ {t=1} ^ T \mathbf{\mu} _ t ^ {(l)}\\
\sigma _ {tot} = \frac{1}{T}\sum _ {t=1} ^ {T} \mathbf{v} _ t ^ {(l)} + \frac{1}{T}\sum _ {t=1} ^ T\left(\mathbf{\mu} _ t ^ {(l)}-\bar{\mathbf{\mu}}\right) ^ 2
\end{array}\tag{11}\right.$$
其中 \\(\\{\\mathbf{\\mu} _ t ^ {(l)},\\mathbf{v} _ t ^ {(l)}\\} _ {t=1} ^ T\\) 是 ADF 网络 \\(T\\) 次蒙特卡洛采样结果。**由此可见，不同于以往将认知不确定性和偶然不确定性完全作独立假设的方式，本文方法是将二者联合来估计的。这也比较好理解，如果数据噪音很大，那么模型就很难训，其模型不确定性也会很大，所以二者不可能是完全独立的**。该方法可归纳为：

0. 以正常方式训练网络；
1. 将现有的网络转换为 ADF 网络形式，增加每层的方差传递函数；
2. 计算 \\(T\\) 次蒙特卡洛采样的网络输出；
3. 计算网络预测的均值和方差。

## 5.&ensp;总结
　　本文以多传感器目标检测后融合任务为出发，介绍了基于深度学习的感知不确定性估计方法。需要注意的是，一般情况下，同一传感器同一模型的不同预测量之间不确定性的相对大小才有意义。所以进行多传感器融合时，需要对不同模型估计的不确定性进行幅值标定，这里不作展开。

## 6.&ensp;参考文献
<a id="1" href="#1ref">[1]</a> Kendall, Alex, and Yarin Gal. "What uncertainties do we need in bayesian deep learning for computer vision?." Advances in neural information processing systems. 2017.  
<a id="2" href="#2ref">[2]</a> Gal, Yarin. Uncertainty in deep learning. Diss. PhD thesis, University of Cambridge, 2016.  
<a id="3" href="#1ref">[3]</a> Loquercio, Antonio , Segù, Mattia, and D. Scaramuzza . "A General Framework for Uncertainty Estimation in Deep Learning." (2019).  
<a id="4" href="#1ref">[4]</a> Kendall, Alex, Yarin Gal, and Roberto Cipolla. "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.  
