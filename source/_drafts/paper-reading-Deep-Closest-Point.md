---
title: '[paper_reading]-"Deep Closest Point"'
date: 2019-12-05 10:00:53
tags: ["paper reading", "ICP", "Deep Learning", "Point Cloud", "Scene Flow"]
categories: Trash
mathjax: true
---

　　考虑一维情况，假设点 \\(x\\) 在前后帧 Lidar 坐标系下的表示分别为 \\(x _ {t-1}\\) 与 \\(x _ t\\)，本车运动量为 \\(x  _ e\\)。那么在栅格分辨率 \\(r\\) 下，\\(t-1\\) 时刻该点所在的栅格为：
$$ c _ {t-1}=\left\lfloor \frac{x _ {t-1}}{r}\right\rfloor \tag{1}$$
\\(t\\) 时刻该点所在的栅格为：
$$ c _ {t}=\left\lfloor \frac{x _ {t}}{r}\right\rfloor =\left\lfloor \frac{x _ {t-1}-x _ e}{r}\right\rfloor =\left\lfloor \frac{x _ {t-1}}{r}-\frac{x _ e}{r}\right\rfloor \tag{2}$$
\\(t-1\\) 时刻该栅格变换到 \\(t\\) 时刻对应的栅格为：
$$\hat{c} _ {t} = \left\lfloor\frac{(c _ {t-1} + \delta)\cdot r - x _ e}{r}\right\rfloor=\left\lfloor \left\lfloor\frac{x _ {t-1}}{r}\right\rfloor+\delta-\frac{x _ e}{r}\right\rfloor \tag{3}$$
其中 \\(\\delta\\) 为代表栅格的点与栅格编号的残差，一般取 0.5。  
　　比较 \\(c _ t\\) 与 \\(\\hat{c} _ t\\) 是否一致。只考虑小数情况，**容易看出 \\(c _ t\\) 可能比 \\(\\hat{c} _ {t}\\) 大 1，也可能小 1，也可能相等**。举例如下：

1. 大于 1 的情况  
\\(\\frac{x _ {t-1}}{r} = 0.9, \\frac{x _ e}{r}=0.6\\)；
2. 小于 1 的情况  
\\(\\frac{x _ {t-1}}{r} = 0.1, \\frac{x _ e}{r}=0.2\\)；





# 符号和变量

`ImuModel::Predict`函数是利用当前车辆的运动状态和已知的标定参数去推理当前`IMU`这个传感器的输出量。为了方便表述，我们把`ImuModel::Predict`命名为函数$h$，
$$
h:T_{m}^{v},\omega_v,v_v,\epsilon_v,a_v,b_\omega,b_a\rightarrow \omega_i,a_i
$$
其中，

1. $T_m^v$是地图坐标系到车辆坐标系的变换矩阵，由旋转部分$R_m^v$（旋转矩阵）和平移部分$t_m^v$（平移向量）组成；
2. $T_v^i$是车辆坐标系到IMU坐标系的变换矩阵，由旋转部分$R_v^i$和平移部分$t_v^i$组成，这个参数是标定时确定的，在系统中为常数；
3. $\omega_v$是车辆坐标系的角速度；
4. $v_v$是车辆坐标系的线速度；
5. $\epsilon_v$是车辆坐标系的角加速度；
6. $a_v$是车辆坐标系的线加速度；
7. $b_\omega$是IMU陀螺仪的零偏；
8. $b_a$是IMU加速度计的零偏；
9. $\omega_i$是IMU应该输出的陀螺仪测量（包含零偏）；
10. $a_i$是IMU应该输出的加速度计测量（包含零偏和重力加速度）；
11. 车辆坐标系$F_v$和IMU坐标系$F_i$方向相同，刚性连接，仅相差一个平移向量$r_v^i$；
12. $g$为重力加速度，在这里为常量

# 预测传感器输出量

预测函数$h$其实是分成两部分的：$h_\omega$是对IMU角速度的预测；$h_a$是对IMU加速度的预测。

## 角速度$\omega_i$

因为$F_v$和$F_i$方向相同且刚性连接，所以，
$$
\widetilde{\omega}_i=\omega_v
$$
再将陀螺仪的偏置加进去，也就是，
$$
h_\omega=\omega_v+b_\omega \tag{1}
$$

## 线加速度$a_i$

我们知道，角速度$\omega$乘以旋转半径$r$即为该质点的速度，即
$$
\vec\omega\times \vec{r}=\vec{v}
$$
为了公式简便，我们忽略向量符号$\vec{\bullet}$，虽然公式了没有向量符号，但是要记得都是向量，即
$$
\omega\times r=v \tag{2}
$$
根据哥式定理的速度合成方程，
$$
v_i=v_v+\omega_v\times r_v^i \tag{3}
$$


我们知道，
$$
a_i=\frac{\partial v_i}{\partial t} \tag{4}
$$
和叉乘的偏导求导公式，
$$
\frac{\partial a\times b}{\partial t}=\frac{\partial a}{\partial t}\times b+a\times\frac{\partial b}{\partial t} \tag{5}
$$


我们将方程$(3)$、$(5)$带入$(4)$式，可得，
$$
\begin{align*}
\widetilde{a}_i&=\frac{\partial v_i}{\partial t} \\
&=\frac{\partial v_v}{\partial t}+\frac{\partial \omega_v\times r_v^i}{\partial t} \\
&=a_v+\frac{\partial\omega_v}{\partial t}\times r_v^i+\omega_v\times\frac{\partial r_v^i}{\partial t} \\
&=a_v+\epsilon_v\times r_v^i+\omega_v\times(\omega_v\times r_v^i)
\end{align*}
$$

接下来在把重力的影响和加速度计的偏置加进来，可得，
$$
\begin{align*}
a_i&=\widetilde{a}_i+R_i^m\cdot g+b_a \\
&=\widetilde{a}_i+(R_m^i)^{-1}\cdot g+b_a \\
&=\widetilde{a}_i+(R_m^v\cdot R_v^i)^{-1}\cdot g+b_a \\
&=a_v+\epsilon_v\times r_v^i+\omega_v\times(\omega_v\times r_v^i)+(R_m^v\cdot R_v^i)^{-1}\cdot g+b_a
\end{align*}
$$
因此，
$$
h_a=a_v+\epsilon_v\times r_v^i+\omega_v\times(\omega_v\times r_v^i)+(R_m^v\cdot R_v^i)^{-1}\cdot g+b_a \tag{6}
$$

## 总结

$$
\left\{
\begin{align*}
h_\omega&=\omega_v+b_\omega \\
h_a&=a_v+\epsilon_v\times r_v^i+\omega_v\times(\omega_v\times r_v^i)+(R_m^v\cdot R_v^i)^{-1}\cdot g+b_a
\end{align*}
\right.
$$



# $h$的雅克比矩阵$H$

在我们的系统中，$R_{m}^{v},t_{m}^{v},\omega_v,v_v,\epsilon_v,a_v,b_\omega,b_a$都是系统的状态量，所以我们应该将函数$h$依次对$R_{m}^{v},t_{m}^{v},\omega_v,v_v,\epsilon_v,a_v,b_\omega,b_a$求导。

为了简化公式，我们将去掉系统状态变量的上下标。对于系统变量之外的参数，我们还是保留上下标，简化如下，
$$
\left\{
\begin{align*}
R_{m}^{v}&\rightarrow R \\
t_{m}^{v}&\rightarrow t \\
\omega_v&\rightarrow \omega \\
v_v&\rightarrow v \\
\epsilon_v&\rightarrow \epsilon \\
a_v&\rightarrow a \\
b_\omega&\rightarrow b_\omega \\
b_a&\rightarrow b_a \\
r_v^i&\rightarrow r
\end{align*}
\right.
$$
在李群的求导中，我们使用右扰动求导。

## 公式、性质和方法等预备知识

### 李代数求导方法

这里以右扰动为例

#### 扰动求导的通用公式

扰动模型求导的方法核心是对要求导的变量左乘或是右乘一个微小扰动项，这个微小扰动项会在最后结果上有一个微小的差。根据导数的定义:
$$
\frac{\partial{f}}{\partial{g}}=\lim_{\delta\rightarrow 0}\frac{f(g+\delta)-f(g)}{\delta}
$$
可得右扰动求导的通用公式：
$$
\begin{align*}
\frac{\partial{f}}{\partial{g}}&=\lim_{\delta\rightarrow 0}\frac{f(g\cdot\exp(\delta))-f(g)}{\delta} \tag{7}
\end{align*}
$$
其中，$\delta$属于李代数，$g$属于李群。

#### 李群到李群映射的扰动求导公式

有一个从**李群到李群**的映射$f$
$$
f:G\rightarrow G
$$

> 特殊正交群($SO(3)$)和欧式正交群($SE(3)$)都是李群的一种

> **注意是从：李群到李群的映射**

我们要求如下导数：
$$
\frac{\partial{f(g)}}{\partial{g}}
$$
在函数$f$的自变量上右乘一个微小扰动量$\exp(\delta)$，会使得函数值上产生一个微小扰动量$\exp(\epsilon)$：
$$
f(g\cdot\exp(\delta))=f(g)\cdot\exp(\epsilon) \tag{8}
$$
那么根据导数的定义：
$$
\frac{\partial{f}}{\partial{g}}\equiv \frac{\partial{\epsilon}}{\partial\delta}|_{\delta=0}
$$
根据式$(8)$，将$\epsilon$转换为$\delta$的表达式。将式$(8)$左右两边求$\log$：
$$
\epsilon=\log(f^{-1}(g)\cdot f(g\cdot\exp(\delta)))
$$
那么：
$$
\begin{align*}
\frac{\partial{f}}{\partial{g}}&=\frac{\partial{\exp(\epsilon)}}{\partial\exp(\delta)}|_{\delta=0} \\
&=\frac{\partial{\log(f^{-1}(g)\cdot f(g\cdot\exp(\delta)))}}{\partial\delta}|_{\delta=0} \tag{9}
\end{align*}
$$

### 其他性质和函数

-  $\bullet^\wedge$运算符
   $$
   \begin{bmatrix}x_1 \\ x_2 \\ x_3 \end{bmatrix}^\wedge=
   \begin{bmatrix}
   0 & -x_3 & x_2\\
   x_3 & 0 & -x_1\\
   -x_2 & x_1 & 0
   \end{bmatrix}
   $$
   
- $$
  a^\wedge b=a\times b=-b\times a=-b^\wedge a \tag{10}
  $$

- $\exp(\phi)\thickapprox I+\phi^\wedge$
  
  证明：
  
  $$
  \begin{align*}
  \exp(\phi)&=\sum^{\infty}_{n=0}\frac{1}{n!}(\phi^\wedge)^n \\
  &=I+\phi^\wedge+\frac{1}{2}(\phi^\wedge)^2+\cdots \\
&\thickapprox I+\phi^\wedge \tag{11}
  \end{align*}
  $$
  
- $\exp(\phi)^{-1}=\exp(-\phi)$

  证明：

  已知：
  $$
  \begin{align*}
  \exp(\phi)=\exp(\theta\cdot a)=\cos(\theta)\cdot I+(1-\cos(\theta))a\cdot a^\intercal+\sin(\theta)a^\wedge
  \end{align*}
  $$
  和：
  $$
  (a^\wedge)^\intercal=-a^\wedge
  $$
  则：
  $$
  \begin{align*}
  \exp(\phi)^{-1}=\exp(\theta\cdot a)^\intercal&=\cos(\theta)\cdot I^\intercal+(1-\cos(\theta))(a\cdot a^\intercal)^\intercal+\sin(\theta)(a^\wedge)^\intercal \\
  &=\cos(\theta)\cdot I+(1-\cos(\theta))a\cdot a^\intercal-\sin(\theta)a^\wedge \\
  \\
  \exp(-\phi)=\exp(-\theta\cdot a)&=\cos(-\theta)\cdot I+(1-\cos(-\theta))a\cdot a^\intercal+\sin(-\theta)a^\wedge \\
  &=\cos(\theta)\cdot I+(1-\cos(\theta))a\cdot a^\intercal-\sin(\theta)a^\wedge
  \end{align*}
  $$
  因此：
  $$
  \exp(\phi)^{-1}=\exp(-\phi) \tag{12}
  $$

  

## 将函数$h_\omega$对求所有状态变量求导

易得，
$$
\left\{
\begin{align*}
\frac{\partial h_\omega}{\partial R}&=0 \\
\frac{\partial h_\omega}{\partial t}&=0 \\
\frac{\partial h_\omega}{\partial \omega}&=I_{3\times3} \\
\frac{\partial h_\omega}{\partial v}&=0 \\
\frac{\partial h_\omega}{\partial \epsilon}&=0 \\
\frac{\partial h_\omega}{\partial a}&=0 \\
\frac{\partial h_\omega}{\partial b_\omega}&=I_{3\times3} \\
\frac{\partial h_\omega}{\partial b_a}&=0
\end{align*}
\right.
$$

## 将函数$h_a$对求所有状态变量求导

### 将函数$h_a$对$R$求导

$$
\begin{align*}
\frac{\partial h_a}{\partial R}&=\frac{\partial}{\partial R}(a+\epsilon\times r+\omega\times(\omega\times r)+(R\cdot R_v^i)^{-1}\cdot g+b_a) \\
&=\frac{\partial}{\partial R}\left[(R\cdot R_v^i)^{-1}\cdot g\right] \tag{13}
\end{align*}
$$

由于，式$(13)$不是李群到李群的映射，所以只能将式$(7)$的方法带入，
$$
\begin{align*}
\frac{\partial h_a}{\partial R}
&=\frac{\partial}{\partial R}\left[(R\cdot R_v^i)^{-1}\cdot g\right] \\
&=\lim_{\delta\rightarrow 0}\frac{[R\cdot\exp(\delta)\cdot R_v^i]^{-1}\cdot g-(R\cdot R_v^i)^{-1}\cdot g}{\delta} \\
&=\lim_{\delta\rightarrow 0}\frac{(R_v^i)^{-1}\cdot(\exp(\delta))^{-1}\cdot R^{-1}\cdot g-(R_v^i)^{-1}\cdot R^{-1}\cdot g}{\delta} \\
&=\lim_{\delta\rightarrow 0}\frac{(R_v^i)^{-1}\cdot[(\exp(\delta))^{-1}-I]\cdot R^{-1}\cdot g}{\delta} \tag{14}\\
\end{align*}
$$
将式$(10)$、式$(11)$、式$(12)$带入式$(14)$可得，
$$
\begin{align*}
\frac{\partial h_a}{\partial R}
&=\lim_{\delta\rightarrow 0}\frac{(R_v^i)^{-1}\cdot[(\exp(\delta))^{-1}-I]\cdot R^{-1}\cdot g}{\delta}\\
&=\lim_{\delta\rightarrow 0}\frac{(R_v^i)^{-1}\cdot(\exp(-\delta)-I)\cdot R^{-1}\cdot g}{\delta}\\
&=\lim_{\delta\rightarrow 0}-\frac{(R_v^i)^{-1}\cdot\delta^\wedge\cdot R^{-1}\cdot g}{\delta}\\
&=\lim_{\delta\rightarrow 0}\frac{(R_v^i)^{-1}\cdot(R^{-1}\cdot g)^\wedge\cdot \delta}{\delta}\\
&=(R_v^i)^{-1}\cdot(R^{-1}\cdot g)^\wedge\\
\end{align*}
$$

### 将函数$h_a$对$\omega$求导

$$
\begin{align*}
\frac{\partial h_a}{\partial\omega}&=\frac{\partial}{\partial \omega}\left[a+\epsilon\times r+\omega\times(\omega\times r)+(R\cdot R_v^i)^{-1}\cdot g+b_a\right] \\
&=\frac{\partial}{\partial \omega}\left[\omega\times(\omega\times r)\right] \tag{15}
\end{align*}
$$

将式$(5)$带入式$(15)$中，
$$
\begin{align*}
\frac{\partial h_a}{\partial\omega}
&=\frac{\partial}{\partial \omega}\left[\omega\times(\omega\times r)\right] \\
&=\frac{\partial \omega}{\partial \omega}\times(\omega\times r)+\omega\times\frac{\partial\omega\times r}{\partial\omega} \\
&=-(\omega\times r)^\wedge+\omega^\wedge\cdot(-r^\wedge) \\
&=-(\omega\times r)^\wedge-\omega^\wedge\cdot r^\wedge
\end{align*}
$$

### 将函数$h_a$对$\epsilon$求导

$$
\begin{align*}
\frac{\partial h_a}{\partial\epsilon}&=\frac{\partial}{\partial \epsilon}\left[a+\epsilon\times r+\omega\times(\omega\times r)+(R\cdot R_v^i)^{-1}\cdot g+b_a\right] \\
&=\frac{\partial}{\partial \epsilon}(\epsilon\times r) \\
&=-r^\wedge
\end{align*}
$$

### 将函数$h_a$对$a$求导

$$
\begin{align*}
\frac{\partial h_a}{\partial a}&=\frac{\partial}{\partial a}\left[a+\epsilon\times r+\omega\times(\omega\times r)+(R\cdot R_v^i)^{-1}\cdot g+b_a\right] \\
&=\frac{\partial}{\partial  a}(a) \\
&=I_{3\times3}
\end{align*}
$$

### 将函数$h_a$对$b_a$求导

$$
\begin{align*}
\frac{\partial h_a}{\partial b_a}&=\frac{\partial}{\partial b_a}\left[a+\epsilon\times r+\omega\times(\omega\times r)+(R\cdot R_v^i)^{-1}\cdot g+b_a\right] \\
&=\frac{\partial}{\partial  b_a}(a) \\
&=I_{3\times3}
\end{align*}
$$

### 总结

因此，
$$
\left\{
\begin{align*}
\frac{\partial h_a}{\partial R}&=(R_v^i)^{-1}\cdot(R^{-1}\cdot g)^\wedge \\
\frac{\partial h_a}{\partial t}&=0 \\
\frac{\partial h_a}{\partial \omega}&=-(\omega\times r)^\wedge-\omega^\wedge\cdot r^\wedge \\
\frac{\partial h_a}{\partial v}&=0 \\
\frac{\partial h_a}{\partial \epsilon}&=-r^\wedge \\
\frac{\partial h_a}{\partial a}&=I_{3\times3} \\
\frac{\partial h_a}{\partial b_\omega}&=0 \\
\frac{\partial h_a}{\partial b_a}&=I_{3\times3}
\end{align*}
\right.
$$

## $h$的雅克比矩阵$H$

根据上面的推到，$h$的雅克比矩阵$H$为，
$$
\begin{align*}
H=&
\begin{bmatrix}
\frac{\partial h_\omega}{\partial R} &
\frac{\partial h_\omega}{\partial t} &
\frac{\partial h_\omega}{\partial \omega} &
\frac{\partial h_\omega}{\partial v} &
\frac{\partial h_\omega}{\partial \epsilon} &
\frac{\partial h_\omega}{\partial a} &
\frac{\partial h_\omega}{\partial b_\omega} &
\frac{\partial h_\omega}{\partial b_a} \\

\frac{\partial h_a}{\partial R} &
\frac{\partial h_a}{\partial t} &
\frac{\partial h_a}{\partial \omega} &
\frac{\partial h_a}{\partial v} &
\frac{\partial h_a}{\partial \epsilon} &
\frac{\partial h_a}{\partial a} &
\frac{\partial h_a}{\partial b_\omega} &
\frac{\partial h_a}{\partial b_a}
\end{bmatrix} \\
=&
\begin{bmatrix}
0 &
0 &
I_{3\times3} &
0 &
0 &
0 &
I_{3\times3} &
0 \\

(R_v^i)^{-1}\cdot(R^{-1}\cdot g)^\wedge &
0 &
-(\omega\times r)^\wedge-\omega^\wedge\cdot r^\wedge &
0 &
-r^\wedge &
I_{3\times3} &
0 &
I_{3\times3}
\end{bmatrix}
\end{align*}
$$

