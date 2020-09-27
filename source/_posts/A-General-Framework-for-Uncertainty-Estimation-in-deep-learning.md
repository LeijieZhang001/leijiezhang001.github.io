---
title: A General Framework for Uncertainty Estimation in Deep Learning
date: 2020-09-04 09:16:44
updated: 2020-09-27 09:19:12
tags: ["Deep Learning", "Uncertainty"]
categories:
- Uncertainty
mathjax: true
---

　　Uncertainty 估计在深度学习网络预测中同样非常重要，因为我们不仅需要知道预测结果，还想知道该结果的不确定性。Uncertainty 可分为偶然不确定性(Aleatoric Uncertainty，详见 {%post_link Heteroscedastic-Aleatoric-Uncertainty Heteroscedastic Aleatoric Uncertainty%}) 以及认知不确定性(Epistemic Uncertainty，详见 {%post_link Epistemic-Uncertainty-for-Active-Learning%})。  
　　本文<a href="#1" id="1ref"><sup>[1]</sup></a>提出了一种同时估计偶然不确定性与认知不确定性的方法。对于网络输入 \\(\\mathbf{x}\\)，输出的后验概率为 \\(p(\\mathbf{y}|\\mathbf{x})\\)，那么由 Aleatoric Uncertainty 和 Epistemic Uncertainty 构成的总的不确定性为 \\(\\sigma _ {tot} = \\mathbf{Var} _ {p(\\mathbf{y}|\\mathbf{x})}(\\mathbf{y})\\)。

## 1.&ensp;Aleatoric Uncertainty(Data Uncertainty)
　　假设传感器得到的数据符合噪音水平 \\(\\mathbf{v}\\) 的高斯分布，那么输入网络的数据 \\(\\mathbf{z}\\) 与其真实数据 \\(\\mathbf{x}\\) 的关系为：
$$q(\mathbf{z}|\mathbf{x})\sim \mathcal{N}(\mathbf{z};\mathbf{x},\mathbf{v})\tag{1}$$
为了计算网络输出的 Data Uncertainty，通过 Assumed Density Filtering(ADF) 来传递输入数据的噪音。网络的联合概率分布为：
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
基于高斯分布的假设，以上解为：
$$\begin{align}
\mathbf{\mu}^{(i)}=\mathbb{E} _ {q(\mathbf{z}^{(i-1)})}[\mathbf{f}^{(i)}(z^{(i-1)})]\\
\mathbf{v}^{(i)}=\mathbb{V} _ {q(\mathbf{z}^{(i-1)})}[\mathbf{f}^{(i)}(z^{(i-1)})]\\
\end{align}\tag{8}$$

## 2.&ensp;Epistemic Uncertainty(Model Uncertainty)
　　Model Uncertainty 表征的是模型的不确定性，对于训练集 \\(\\mathrm{D}=\\{\\mathbf{X},\\mathbf{Y}\\}\\)，模型的不确定性即为权重参数的分布 \\(p(\\mathbf{\\omega} | \\mathbf{X},\\mathbf{Y})\\)。可采用 Monte-Carlo 采样方法来近似估计模型权重分布，具体的采样通过 Dropout 实现：
$$p(\omega|\mathbf{X},\mathbf{Y})\approx q(\mathbf{\omega};\mathbf{\Phi})=Bern(\mathbf{\omega};\mathbf{\Phi}) \tag{9}$$
其中 \\(\\mathbf{\\Phi}\\) 是 Bernolli(dropout) rates，由此模型的不确定性即为 T 次采样的方差：
$$\mathbf{Var} _ {p(\mathbf{y}|\mathbf{x})} ^ {model}(\mathbf{y})=\sigma _ {model} = \frac{1}{T}\sum _ {t=1} ^ T(\mathbf{y} _ t-\bar{\mathbf{y}}) ^ 2\tag{10}$$
其中 \\(\\{\\mathbf{y} _ t\\} _ {t=1} ^ T\\) 是不同权重 \\(\\omega ^ t\\sim q(\\omega;\\mathbf{\\Phi})\\) 采样下的输出。  
　　这种模型不确定性的计算方式，直观的理解为：当模型对某些数据预测比较好，误差比较小的时候，那么模型对这些数据的冗余度肯定是较高的，所以去掉模型的一部分网络，模型对这些数据的预测与原模型应该会有较高的一致性。

## 3.&ensp;Total Uncertainty
<img src="framework.png" width="95%" height="95%" title="图 1. Framework">
　　如图 2.2 所示，结合蒙特卡洛采样与 ADF 方法，网络预测的结果与对应的总的 Uncertainty 可计算为：
$$\left\{\begin{array}{l}
\mu = \frac{1}{T}\sum _ {t=1} ^ T \mathbf{\mu} _ t ^ {(l)}\\
\sigma _ {tot} = \frac{1}{T}\sum _ {t=1} ^ {T} \mathbf{v} _ t ^ {(l)} + \frac{1}{T}\sum _ {t=1} ^ T\left(\mathbf{\mu} _ t ^ {(l)}-\bar{\mathbf{\mu}}\right) ^ 2
\end{array}\tag{11}\right.$$
其中 \\(\\{\\mathbf{\\mu} _ t ^ {(l)},\\mathbf{v} _ t ^ {(l)}\\} _ {t=1} ^ T\\) 是 ADF 网络 \\(T\\) 次蒙特卡洛采样结果。**由此可见，不同于以往将 Model Uncertainty 和 Data Uncertainty 完全作独立假设的方式，本文方法是将二者联合来估计的。这也比较好理解，如果数据噪音很大，那么模型的不确定性也会很大，所以二者不可能是完全独立的**。  
　　该方法可归纳为：

1. 将现有的网络转换为 ADF 网络形式；
2. 手机 \\(T\\) 次蒙特卡洛采样的网络输出；
3. 计算网络输出的 Mean 和 Variance；

其中步骤一不用作任何额外训练的操作，神经网络中的每个操作都有对应的 ADF 操作，代码可参考<a href="#2" id="2ref">[2]</a>。比如基于 Pytorch的 2D 卷积：
```
class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True,
        keep_variance_fn=None, padding_mode='zeros'):
        self._keep_variance_fn = keep_variance_fn
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        def forward(self, inputs_mean, inputs_variance):
            outputs_mean = F.conv2d(inputs_mean, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            outputs_variance = F.conv2d(inputs_variance, self.weight ** 2, None, self.stride, self.padding, self.dilation, self.groups)
            if self._keep_variance_fn is not None:
                outputs_variance = self._keep_variance_fn(outputs_variance)
                return outputs_mean, outputs_variance
```
Softmax:
```
class Softmax(nn.Module):
    def __init__(self, dim=1, keep_variance_fn=None):
        super(Softmax, self).__init__()
        self.dim = dim
        self._keep_variance_fn = keep_variance_fn

        def forward(self, features_mean, features_variance, eps=1e-5):
            """Softmax function applied to a multivariate Gaussian distribution.
            It works under the assumption that features_mean and features_variance 
            are the parameters of a the indepent gaussians that contribute to the 
            multivariate gaussian. 
            Mean and variance of the log-normal distribution are computed following
            https://en.wikipedia.org/wiki/Log-normal_distribution."""""
                    
            log_gaussian_mean = features_mean + 0.5 * features_variance
            log_gaussian_variance = 2 * log_gaussian_mean
                    
            log_gaussian_mean = torch.exp(log_gaussian_mean)
            log_gaussian_variance = torch.exp(log_gaussian_variance)
            log_gaussian_variance = log_gaussian_variance*(torch.exp(features_variance)-1)
                    
            constant = torch.sum(log_gaussian_mean, dim=self.dim) + eps
            constant = constant.unsqueeze(self.dim)
            outputs_mean = log_gaussian_mean/constant
            outputs_variance = log_gaussian_variance/(constant**2)
                    
            if self._keep_variance_fn is not None:
                outputs_variance = self._keep_variance_fn(outputs_variance)
                return outputs_mean, outputs_variance
```

## 4.&ensp;Experiments
　　本文在 Steering Angle Prediction，Object Future Motion Prediction，Object Recognition，Closed-Loop Control of a Quadrotor 等任务上作了 Uncertainty 的估计，用 KL，NLL 来评估预测好坏。KL 本质上用于描述两个分布的距离。NLL(Negative Log-likelihood) 数学形式为 \\(\\frac{1}{2}\\mathrm{log}(\\sigma _ {tot})+\\frac{1}{2\\sigma _ {tot}}(y _ {gt}-y _ {pred}) ^ 2\\)，即为 {%post_link Heteroscedastic-Aleatoric-Uncertainty Heteroscedastic Aleatoric Uncertainty%} 中预测 Uncertainty 方法中的 Loss 项。  

## 5.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Loquercio, Antonio , Segù, Mattia, and D. Scaramuzza . "A General Framework for Uncertainty Estimation in Deep Learning." (2019).  
<a id="2" href="#2ref">[2]</a> https://github.com/mattiasegu/uncertainty_estimation_deep_learning/blob/master/contrib/adf.py

