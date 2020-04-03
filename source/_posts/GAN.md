---
title: GAN
date: 2020-03-08 16:45:51
tags: ["Deep Learning"]
categories: Deep Learning
mathjax: true
---

　　Generative Adversarial Nets(GAN) 能将某个分布的数据映射到另一组数据形成的分布空间内。这在某些领域非常有用，如：图像去噪，图像去雨雾，图像去模糊，图像低光照增强等。**自动驾驶中，图像去雨雾与低光照增强非常关键，GAN 能在没有模拟器的情况下，根据有限的数据，自动生成某一分布的数据，为后续感知做准备**。目前还没看到针对点云的 GAN，未来 3D GAN 可能会有大进展。  
　　本文介绍几个 GAN 的基础性工作。

## 1.&ensp;GAN 基础网络
### 1.1.&ensp;Generative Adversarial Nets<a href="#1" id="1ref"><sup>[1]</sup></a>
　　对抗网络由生成模型和判别模型构成。生成模型输入随机噪声，输出以假乱真的图像，判别模型则对图像作分类。其优化函数为：
$$ \min\limits _ G \max\limits _ D V(D,G) = E _ {x\sim p _ {data}(x)}[log(D(x))] + E _ {x\sim p _ z(z)}[log(1-D(G(z)))] \tag{1}$$
该优化过程有两部分组成：

1. **优化判别模型**  
$$ \max\limits _ D V(D,G) = E _ {x\sim p _ {data}(x)}[log(D(x))] + E _ {x\sim p _ z(z)}[log(1-D(G(z)))] \tag{2}$$
其中第一项表示输入为真样本时，那么判别模型输出越大越好，即越接近 1；而对于已经生成的假样本 \\(G(z)\\)，判别模型输出越小越好，即接近 0。
2. **优化生成模型**  
$$ \min\limits _ GV(D,G) =E _ {x\sim p _ z(z)}[log(1-D(G(z)))] \tag{3}$$
优化生成模型时，希望生成的假样本接近真样本，所以生成的假样本经过判别模型后越大越好，即\\(D(G(z))\\)要接近 1。由此统一成上式。

　　对抗网络的优化由这两步迭代组成。

### 1.2.&ensp;Conditional Generative Adversarial Nets<a href="#2" id="2ref"><sup>[2]</sup></a>
　　条件对抗网络中的生成模型输入不在是随机噪声，而是特定的数据分布，如真值标签。其优化函数为：
$$ \min\limits _ G \max\limits _ D V(D,G) = E _ {x\sim p _ {data}(x)}[log(D(x|y))] + E _ {x\sim p _ z(z)}[log(1-D(G(z|y)))] \tag{4}$$
　　其优化过程与 GAN 类似。

### 1.3.&ensp;Cycle-Consistent Adversarial Nets<a href="#3" id="3ref"><sup>[3]</sup></a>
　　Cycle GAN 使得高分辨率图像的 domain-transfer 成为可能。对于两个图像分布 \\(X,Y\\)，设计两个映射函数(生成模型): \\(G:X\\to Y\\) 和 \\(F:Y\\to X\\)；设计两个判别模型: \\(D _ X\\) 和 \\(D _ Y\\)，\\(D _ X\\) 用于判别 \\(x\\) 与 \\(F(y)\\), \\(D _ Y\\) 用于判别 \\(y\\) 与 \\(G(x)\\)。为了还原高分辨率图像，设计两部分 Loss：

1. **Adversarial Loss**  
就是传统的对抗网络 Loss:
$$\begin{align}
\mathcal{L} _ {GAN}&=\mathcal{L} _ {GAN}(G, D _ Y,X,Y)+\mathcal{L} _ {GAN}(F, D _ X,Y,X)\\
&= E _ {y\sim p _ {data}(y)}[log(D _ Y(y))] + E _ {x\sim p _ {data}(x)}[log(1-D _ Y(G(x)))]\\
&+ E _ {x\sim p _ {data}(x)}[log(D _ X(x))] + E _ {y\sim p _ {data}(y)}[log(1-D _ X(F(Y)))]
\end{align} \tag{5}$$
2. **Cycle Consistency Loss**  
为了保证映射网络的映射准确性，考虑到 \\(x\\to G(x)\\to F(G(x))\\approx x \\) 以及 \\(y\\to F(y)\\to G(F(y))\\approx y \\)，设计 cycle loss：
$$\mathcal{L} _ {cyc}(G,F)= E _ {x\sim p _ {data}(x)}\Vert F(G(x))-x\Vert + E _ {y\sim p _ {data}(y)}\Vert G(F(y))-y\Vert \tag{6}$$

总的 Loss 为：
$$\mathcal{L} _ (G,F,D _ X, D _ Y)=\mathcal{L} _ {GAN}(G, D _ Y,X,Y)+\mathcal{L} _ {GAN}(F, D _ X,Y,X)+\lambda \mathcal{L} _ {cyc}(G,F) \tag{7}$$

## 2.&ensp;其它资料
　　上面介绍了三个 GAN 基本网络，尤其是 Cycle-GAN，是高分辨率图像无监督 domain-transfer 的基础，应用相当广泛。本文介绍相对较简单，<a href="#4" id="4ref">[4]</a> 详细介绍了 GAN 的来龙去脉。代码则可以参考 <a href="#5" id="5ref">[5]</a> ，收录的 GAN 网络非常详细。

## 3.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.  
<a id="2" href="#2ref">[2]</a> Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." arXiv preprint arXiv:1411.1784 (2014).  
<a id="3" href="#3ref">[3]</a> Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017.  
<a id="4" href="#4ref">[4]</a> http://www.gwylab.com/note-gans.html  
<a id="5" href="#5ref">[5]</a> https://github.com/eriklindernoren/PyTorch-GAN  
