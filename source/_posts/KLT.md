---
title: KLT 光流算法详解
date: 2019-06-19 18:58:47
tags: ["tracking", "ADAS", "autonomous driving"]
categories: Scene Flow
mathjax: true
---

　　光流（Optical Flow）是物体在三维空间中的运动（运动场）在二维图像平面上的投影，由物体与相机的相对速度产生，反映了微小时间内物体对应的图像像素的运动方向和速度。  
　　KLT 是基于光流原理的一种特征点跟踪算法，本文首先介绍光流原理，然后介绍 KLT 及相关 KLT 变种算法。

## 1.&ensp;Optical Flow
　　光流法假设：

- 亮度恒定，图像中物体的像素亮度在连续帧之间不会发生变化；
- 短距离(短时)运动，相邻帧之间的时间足够短，物体运动较小；
- 空间一致性，相邻像素具有相似的运动；

　　记 \\(I(x,y,t)\\) 为 \\(t\\) 时刻像素点 \\((x,y)\\) 的像素值，那么根据前两个假设，可得到：
$$I(x,y,t)=I(x+dx,y+dy,t+dt)$$
一阶泰勒展开：
$$I(x+dx,y+dy,t+dt)=I(x,y,t)+\frac{\partial I}{\partial x}dx+\frac{\partial I}{\partial y}dy+\frac{\partial I}{\partial t}dt$$
由此可得：
$$\frac{\partial I}{\partial x}dx+\frac{\partial I}{\partial y}dy+\frac{\partial I}{\partial t}dt=0 \iff \frac{\partial I}{\partial x}\frac{dx}{dt}+\frac{\partial I}{\partial y}\frac{dy}{dt}=-\frac{\partial I}{\partial t}$$
记 \\(\\left(\\frac{dx}{dt},\\frac{dy}{dt}\\right)=(u,v)\\)，即为所要求解的像素光流；\\(\\left(\\frac{\\partial I}{\\partial x},\\frac{\\partial I}{\\partial y}\\right)=(I_x,I_y)\\) 为像素灰度空间微分；\\(\\frac{\\partial I}{\\partial t}=I_x\\) 为像素坐标点的时间灰度微分。整理成矩阵形式：
$$\begin{bmatrix}
I_x &I_y\\
\end{bmatrix}
\begin{bmatrix}
u\\
v\\
\end{bmatrix}=-I_t
$$
该式表示相同坐标位置的时间灰度微分是空间灰度微分与这个位置上相对于观察者的速度的乘积。由空间一致性假设，对于周围多个点，有：
$$\begin{bmatrix}
I_{x1} &I_{y1}\\
I_{x2} &I_{y2}\\
I_{x3} &I_{y3}\\
\vdots &\vdots \\
\end{bmatrix}
\begin{bmatrix}
u\\
v\\
\end{bmatrix}=-
\begin{bmatrix}
I_{t1}\\
I_{t2}\\
\vdots\\
\end{bmatrix} \iff A\vec{u}=b
$$
这是标准的线性方程组，可用最小二乘法求解 \\(\\vec{u}=\\left(A^ TA\\right)^ {-1}A^ Tb\\)，也可以迭代求解。这种方式得到的光流，称为 Lucas-Kanade 算法。

## 2.&ensp;KLT
　　KLT 算法本质上也基于光流的三个假设，不同于前述直接比较像素点灰度值的作法，KLT 比较像素点周围的窗口像素，来寻找最相似的像素点。由光流假设，在很短时间 \\(\\tau\\) 内，前后两帧图像满足：
$$J(A\mathrm{x}+d)=I(\mathrm{x}), 其中 A=1+D=1+\begin{bmatrix}
d_{xx} & d_{xy}\\
d_{yx} & d_{yy}\\
\end{bmatrix}$$
像素位移(displacement)向量满足仿射运动模型(Affine Motion) \\(\delta=Dx+d\\)，其中 \\(D\\) 称为变形矩阵(Deformation Matrix)，\\(d\\) 称为位移向量(Displacement Vector)。\\(D\\) 表示两个像素窗口块运动后的变形量，所以当窗口较小时，会比较难估计。通常 \\(D\\) 可以用来衡量两个像素窗口的相似度，即衡量特征点有没有漂移。而对于光流跟踪量，一般只考虑平移模型(Translation Model)：
$$J(\mathrm{x}+d)=I(\mathrm{x})$$
　　为了普遍性，我们用仿射运动模型来推到 KLT 算法原理。在像素窗口下，构造误差函数：
$$\epsilon=\iint_W [J(A\mathrm{x}+d)-I(x)]^2 w(\mathrm{x})d\mathrm{x}$$
其中 \\(w(\\mathrm{x})\\) 是权重函数，可定义为高斯形式。上式分别对变量 \\(D\\) 和 \\(d\\) 求导：
$$\left\{\begin{array}{l}
\frac{\partial \epsilon}{\partial D}=2\iint_W[J(A\mathrm{x}+d)-I(\mathrm{x})]g\,\mathrm{x}^T\,w\,d\mathrm{x}&=0\\
\frac{\partial \epsilon}{\partial d}=2\iint_W[J(A\mathrm{x}+d)-I(\mathrm{x})]g\,w\,d\mathrm{x}&=0\\
\end{array}\right.$$
其中 \\(g=\\left(\\frac{\\partial J}{\\partial x},\\frac{\\partial J}{\\partial y}\\right)^ T\\)。记光流 \\(u=D\\mathrm{x}+d\\)，则对运动后的像素点进行泰勒展开：
$$J(A\mathrm{x}+d)=J(x)+g^T(u)$$
仿射运动模型结果可见<a href="#1" id="1ref">[1]</a><a href="#5" id="5ref">[5]</a>，这里给出平移运动模型结果。令 \\(D=0\\)：
$$\begin{align}
&\iint_W[J(A\mathrm{x}+d)-I(\mathrm{x})]g\,w\,d\mathrm{x}=0\\
\iff &\iint_W[J(\mathrm{x})-I(\mathrm{x})]g\,w\,d\mathrm{x}=-\iint_Wg^T\,\mathrm{d}\,g\,w\,d\mathrm{x}=-\left[\iint_Wg\,g^T\,w\,d\mathrm{x}\right]\mathrm{d}\\
\iff &Z\mathrm{d}=e
\end{align}$$
其中 \\(Z\\) 是 \\(2\\times 2\\) 矩阵，\\(e\\) 是 \\(2\\times 1\\) 向量。这是线性方程组优化问题，当 \\(Z\\) 可逆时，这个方程可容易求解。因为推导过程用到了泰勒展开，所以只有当像素位移较小时，才成立。实际操作中，一般迭代式的来求解，每次用上次结果做初始化，进一步求解(In a Newton-Raphson Fasion)。

## 3.&ensp;Pyramidal Iterative KLT 
　　以上标准的迭代式 KLT 计算过程只在位移较小时成立（泰勒展开），所以需要更优的金字塔式迭代求解。图像金字塔有多重定义方式，这里定义：
$$\begin{align}
I^L(x,y)&=\frac{1}{4}I^{L-1}(2x,2y)\\
&+\frac{1}{8}\left(I^{L-1}(2x-1,2y)+I^{L-1}(2x+1,2y)+I^{L-1}(2x,2y-1)+I^{L-1}(2x,2y+1)\right)\\
&+\frac{1}{16}\left(I^{L-1}(2x-1,2y-1)+I^{L-1}(2x+1,2y+1)+I^{L-1}(2x-1,2y+1)+I^{L-1}(2x+1,2y-1)\right)
\end{align}$$
　　特征点跟踪有两个关键指标：**准确性(accuracy)**，以及**鲁棒性(robustness)**。大的窗口，对大的运动量比较鲁棒，但是为了提高准确性，又不得不减小窗口。所以窗口的选择需要权衡跟踪准确性与鲁棒性。金字塔迭代 KLT 则能有效弱化窗口的局限性。这里介绍平移模型下金字塔迭代 KLT 算法，仿射模型算法过程可见<a href="#1" id="1ref">[1]</a><a href="#5" id="5ref">[5]</a>。  
　　定义金字塔迭代 KLT 算法的目标：图像 \\(I\\) 中某坐标点 \\(\\mathrm{x}\\)，在图像 \\(J\\) 中找到其对应点 \\(\\mathrm{\hat{x}}\\)。算法流程为：

> 建立图像金字塔：\\(\\{I^ L\\}_ {L=0,...,L_m}\\)，\\(\\{J^ L\\}_ {L=0,...,L_m}\\)  
> 初始化光流在金字塔之间的传递值：\\(g^ {L_m}=[g_x^ {L_m},g_y^ {L_m}]^ T=[0,0]^ T\\)  
> **for \\(L=L_m\\) down to 0 with step of -1**
>
>> 计算图像 \\(I^ L\\) 中的 \\(\\mathrm{x}\\) 坐标: \\(\\mathrm{x}^ L=[x,y]^ T=\\mathrm{x}/2^ L\\)  
>> 计算空间梯度矩阵 \\(Z\\)  
>> 初始化 KLT 迭代值：\\(v^ 0=[0,0]^ T\\)  
>> **for \\(k=1\\) to \\(K\\) with step of 1** or until \\(\\Vert\\eta^ k\\Vert\\) < accuracy threshold
>>
>>> 计算图像差矩阵 \\(I^ L(\\mathrm{x}^ L)-J^ L(\\mathrm{x}^ L)=I^ L(x,y)-J^ L(x+g_x^ L+v_x^ {k-1},y+g_y^ L+v_y^ {k-1})\\)  
>>> 计算图像差矩阵 \\(e_k\\)  
>>> 计算光流 \\(\\eta^ k=Z^ {-1}e_k\\)  
>>> 更新下次迭代的初值 \\(v^ k=v^ {k-1}+\\eta^ k\\)
>>
>> **end of for-loop on k**  
>> 第 \\(L\\) 层金字塔下光流为：\\(\\mathrm{d}^ L=v^ K\\)  
>> 初始化第 \\(L-1\\) 层金字塔的光流： \\(g^ {L-1}=[g_x^ {L-1}, g_y^ {L-1}]^ T=2(g^ L+\\mathrm{d}^ L)\\)
>
> **end of for-loop on L**
> 最终的光流结果：\\(\\mathrm{d}=g^ 0+\\mathrm{d}^ 0\\)  
> 对应的 \\(J\\) 上的坐标点为：\\(\\hat{\\mathrm{x}}=\\mathrm{x}+\\mathrm{d}\\)

## 4.&ensp;Feature Selection
　　在特征点跟踪之前，特征点的选择也很重要，以上计算过程中，我们期望 \\(Z\\) 可逆，也就是其最小特征值要足够大。如果已经提取了角点，则可进一步做选择。因此特征点选择准则为：

1. 计算图像每个像素(或已提取的角点)的 \\(Z\\) 矩阵，及其最小的特征值 \\(\\lambda_m\\)
2. 从所有 \\(\\lambda_m\\) 中取最大值为 \\(\\lambda_{max}\\)
3. 保留 \\(\\lambda_m\\) 大于一定百分比(10%) \\(\\lambda_{max}\\) 的像素(角点)
4. 在这些像素(角点)中，保留局部最大值
5. 视计算能力，保留其中的子集

以上特征点提取的过程类似于 [Harris 角点](https://blog.csdn.net/u010103202/article/details/73331440)。要注意的是选择特征计算 \\(Z\\) 时，\\(3\\times3\\) 窗口足够，但是跟踪时，一般大于 \\(3\\times3\\)。

## 5.&ensp;Dissimilarity
　　相似性度量决定该特征点是否已经漂移而不能使用了，即外点检测(Outlier Detection)，所以非常重要。相比于平移模型，仿射模型对特征点的相似性度量更有效果。在长距离跟踪下，相似性度量可能解决不了是否漂移的问题，但是好的相似性度量能从一开始就剔除漂移的特征点。此外，也可用其它更高层面的外点检测技术替代。


<a id="1" href="#1ref">[1]</a> Shi, Jianbo, and Carlo Tomasi. Good features to track. Cornell University, 1993.  
<a id="2" href="#2ref">[2]</a> Birchfield, Stan. "Derivation of kanade-lucas-tomasi tracking equation." unpublished notes (1997).  
<a id="3" href="#3ref">[3]</a> Bouguet, J.-Y.. “Pyramidal implementation of the lucas kanade feature tracker.” (2000).  
<a id="4" href="#4ref">[4]</a> Suhr, Jae Kyu. "Kanade-lucas-tomasi (klt) feature tracker." Computer Vision (EEE6503) (2009): 9-18.  
<a id="5" href="#5ref">[5]</a> Bouguet, Jean-Yves. "Pyramidal implementation of the affine lucas kanade feature tracker description of the algorithm." Intel Corporation 5.1-10 (2001): 4.
