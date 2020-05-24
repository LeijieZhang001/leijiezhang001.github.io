---
title: 非线性最小二乘
date: 2020-05-18 09:19:54
tags: ["Optimization"]
categories: SLAM
mathjax: true
---
　　非线性最小二乘(Non-linear Least Squares)问题应用非常广泛，尤其是在 SLAM 领域。{% post_link LOAM LOAM%}，{% post_link paper-reading-the-Normal-Distributions-Transform The Normal Distributions Transform%}，{% post_link [paper_reading]-Stereo-RCNN-based-3D-Object-Detection-for-Autonomous-Driving Stereo-RCNN%}，{% post_link [paper_reading]-Stereo-Vision-based-Semantic-3D-Object-and-Ego-motion-Tracking-for-Autonomous-Driving Stereo Vision-based Semantic and Ego-motion Tracking for Autonomous Driving%} 等均需要求解非线性最小二乘问题。其中 {% post_link LOAM LOAM%} 作为非常流行的激光 SLAM 框架，其后端是一个典型的非线性最优化问题，本文会作为实践进行代码级讲解。  

## 1.&ensp;问题描述
　　在前端观测-后端优化框架下，设观测数据对集合为：\\(\\{y _ i,z _ i\\} _ {i=1}^m\\)，待求解的变量参数 \\(x\\in\\mathbb{R}^n\\) 定义了观测数据对的映射关系，即 \\(z _ i=h(y _ i;x)\\)，由此得到有 \\(m\\) 个参数方程 \\(F(x)=[f _ 1(x),...,f _ m(x)]^T\\)，其中 \\(f _ i(x) = z _ i-h(y _ i;x)\\)。我们要找到最优的参数 \\(x\\) 来描述观测数据对之间的关系，即求解的最优化问题为：
$$\begin{align}
\mathop{\arg\min}\limits _ x \frac{1}{2}\Vert F(x)\Vert ^2 \iff \mathop{\arg\min}\limits _ x\frac{1}{2}\sum _ i \rho _ i\left(\Vert f _ i(x)\Vert ^ 2\right)\\
L\leq x \leq U
\end{align}\tag{1}$$
其中 \\(f _ i(\\cdot)\\) 为 Cost Function，\\(\\rho _ i(\\cdot)\\) 为 Loss Function，即核函数，用来减少离群点对非线性最小二乘优化的影响；\\(L,U\\) 分别为参数 \\(x\\) 的上下界。当核函数 \\(\\rho _ i(x)=x\\) 时，就是常见的非线性最小二乘问题。  
　　《视觉 SLAM 十四讲》<a href="#1" id="1ref"><sup>[1]</sup></a>在 SLAM 的状态估计问题中，从概率学角度导出了最大似然估计求解状态的方法，并进一步引出了最小二乘问题。回过头来看，本文很多内容在《视觉 SLAM 十四讲》中已经有非常清晰的描述，可作进一步参考。

## 2.&ensp;问题求解 
　　根据 \\(F(x)\\) 求得雅克比矩阵(Jacobian)：\\(J(x) \\in\\mathbb{R}^{m\\times n}\\)，即 \\(J _ {ij}(x)=\\frac{\\partial f _ i(x)}{\\partial x _ j}\\)。目标函数的梯度向量为 \\(g(x) = \\nabla\\frac{1}{2}\\Vert F(x)\\Vert ^ 2=J(x)^TF(x)\\)。在 \\(x\\) 处将目标函数线性化：\\(F(x+\\Delta x)\\approx F(x)+J(x)\\Delta x\\)。由此非线性最小二乘问题可转换为线性最小二乘求解残差量 \\(\\Delta x\\) 来近似求解：
$$\mathop{\arg\min}\limits _ {\Delta x}\frac{1}{2}\Vert J(x)\Delta x+F(x)\Vert ^ 2\tag{2}$$
根据如何控制 \\(\\Delta x\\) 的大小，非线性优化算法可分为两大类：

- Line Search
  - Gradient Descent
  - Gaussian-Newton
- Trust Region
  - Levenberg-Marquardt
  - Dogleg
  - Inner Iterations
  - Non-monotonic Steps

Line Search 首先确定迭代方向，然后最小化 \\(\\Vert f(x+\\alpha \\Delta x)\\Vert ^2\\) 确定迭代步长；Trust Region 则划分一个局部区域，在该区域内求解最优值，然后根据近似程度，扩大或缩减该局部区域范围。Trust Region 相比 Linear Search，数值迭代会更加稳定。这里介绍几种有代表性的方法：属于 Line Search 的梯度下降法，高斯牛顿法，以及属于 Trust Region 的 LM 法。

### 2.1.&ensp;梯度下降法
　　将目标函数式(1)在 \\(x\\) 附近泰勒展开：
$$ \Vert F(x+\Delta x)\Vert ^2 \approx \Vert F(x)\Vert ^2 + J(x)\Delta x+\frac{1}{2}\Delta x^TH\Delta x \tag{3}$$
其中 \\(H\\) 是二阶导数(Hessian 矩阵)。  
　　如果保留一阶导数，那么增量的解就为：
$$\Delta x = -\lambda J^T(x) \tag{4}$$
其中 \\(\\lambda\\) 为步长，可预先由相关策略设定。  
　　如果保留二阶导数，那么增量方程为：
$$\mathop{\arg\min}\limits _ {\Delta x} \Vert F(x)\Vert ^2+J(x)\Delta x+\frac{1}{2}\Delta x^TH\Delta x\tag{5}$$
对 \\(\\Delta x\\) 求导即可求解增量的解为：
$$\Delta x = -H^{-1}J^T \tag{6}$$
　　一阶梯度法又称为最速下降法，二阶梯度法又称为牛顿法。一阶和二阶法都是将函数在当前值下泰勒展开，然后线性得求解增量值。最速下降法过于贪心，容易走出锯齿路线，反而增加迭代步骤。牛顿法需要计算 \\(H\\) 矩阵，计算量较大且困难。

### 2.2.&ensp;高斯牛顿法
　　 将式(2)对 \\(\\Delta x\\) 求导并令其为零，可得：
$$\begin{align}
&J(x)^TJ(x)\Delta x=-J(x)^TF(x)\\
\iff & H\Delta x=g
\end{align}\tag{7}$$
相比牛顿法，高斯牛顿法不用计算 \\(H\\) 矩阵，直接用 \\(J^TJ\\) 来近似，所以节省了计算量。但是高斯牛顿法要求 \\(H\\) 矩阵是可逆且正定的，而实际计算的 \\(J^TJ\\) 是半正定的，所以 \\(J^TJ\\) 会出现奇异或病态的情况，此时增量的稳定性就会变差，导致迭代发散。另一方面，增量较大时，目标近似函数式(2)就会产生较大的误差，也会导致迭代发散。这是高斯牛顿法的缺陷。高斯牛顿法的步骤为：

1. 根据式 (7) 求解迭代步长 \\(\\Delta x\\)；
2. 变量迭代：\\(x ^ * \\leftarrow x+\\Delta x\\)；
3. 如果 \\(\\Vert F(x ^ * )-F(x)\\Vert < \\epsilon\\)，则收敛，退出迭代，否则重复步骤 1.；

高斯牛顿法简单的将 \\(\\alpha\\) 置为 1，而其它 Line Search 方法会最小化 \\(\\Vert f(x+\\alpha \\Delta x)\\Vert ^2\\) 来确定 \\(\\alpha\\) 值。

### 2.3.&ensp;LM 法
　　Line Search 依赖线性化近似有较高的拟合度，但是有时候线性近似效果较差，导致迭代不稳定；Region Trust 就是解决了这种问题。高斯牛顿法中采用的近似二阶泰勒展开只在该点附近有较好的近似结果，对 \\(\\Delta x\\) 添加一个信赖域区域，就变为 Trust Region 方法。其最优化问题转换为：
$$\begin{align}
\mathop{\arg\min}\limits _ x \frac{1}{2}\Vert J(x)\Delta x+F(x)\Vert ^2 \\
\Vert D(x)\Delta x\Vert ^2 \leq \mu\\
L\leq x \leq U\\
\end{align}\tag{8}$$
用 Lagrange 乘子将其转换为无约束优化问题：
$$\mathop{\arg\min}\limits _ {\Delta x}\frac{1}{2}\Vert J(x)\Delta x+F(x)\Vert ^ 2+\frac{1}{\mu}\Vert D(x)\Delta x\Vert ^2 \tag{9}$$
其中 Levenberg 提出的方法中 \\(D=I\\)，相当于把 \\(\\Delta x\\) 约束在球中；Marquart 提出的方法中将 \\(D\\) 取为非负数对角阵，通常为 \\(J(x)^TJ(x)\\) 的对角元素平方根。  
　　对于信赖域区域 \\(\\mu\\) 的定义，一个比较好的方式是根据近似模型与实际函数之间的差异来确定这个范围：如果差异小，那么增大信赖域；反之减小信赖域。因此，考虑：
$$\rho = \frac{\Vert F(x+\Delta x)\Vert ^2-\Vert F(x)\Vert ^2}{\Vert J(x)\Delta x+F(x)\Vert ^2-\Vert F(x)\Vert ^2} \tag{10}$$
　　 将式(9)对 \\(\\Delta x\\) 求导并令其为零，可得：
$$\begin{align}
&\left(J(x)^TJ(x)+\frac{2}{\mu}D^T(x)D(x)\right)\Delta x=-J(x)^TF(x)\\
\iff & (H+\lambda D^TD)\Delta x=g
\end{align}\tag{11}$$
当 \\(\\lambda\\) 较小时，接近于高斯牛顿法；当 \\(\\lambda\\) 较大时，接近于最速下降法。LM 法的步骤为：

1. 根据式(11)求解迭代步长 \\(\\Delta x\\);
2. 根据式(10)求解 \\(\\rho\\);
3. 若 \\(\\rho > \\eta _ 1\\)，则 \\(\\mu = 2\\mu\\);
4. 若 \\(\\rho < \\eta _ 2\\)，则 \\(\\mu = 0.5\\mu\\);
5. 若 \\(\\rho > \\epsilon\\)，则 \\(x ^ * \\leftarrow x+\\Delta x\\)；
6. 如果满足收敛条件，则结束，否则继续步骤1.；

## 3.&ensp;Ceres 实践
　　Ceres 是谷歌开发的一个用于非线性优化的库，使用 Ceres 库有以下几个步骤：

- 构建 Cost Function，式(1)中的 \\(\\rho _ i\\left(\\Vert f _ i(x)\\Vert ^ 2\\right)\\) 即为代码中需要增加的 ResidualBlock；
- 累加的 Cost Function 构成最终的 Loss Function 目标函数；
- 配置求解器参数并求解问题；

### 3.1.&ensp;例子-曲线拟合
　　以下代码为拟合曲线参数的简单例子：

  ```c
// copy from http://zhaoxuhui.top/blog/2018/04/04/ceres&ls.html
#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>

using namespace std;
using namespace cv;
using namespace ceres;

//vector,用于存放x、y的观测数据
//待估计函数为y=3.5x^3+1.6x^2+0.3x+7.8
vector<double> xs;
vector<double> ys;

//定义CostFunctor结构体用于描述代价函数
struct CostFunctor{
  
  double x_guan,y_guan;
  
  //构造函数，用已知的x、y数据对其赋值
  CostFunctor(double x,double y)
  {
    x_guan = x;
    y_guan = y;
  }
  
  //重载括号运算符，两个参数分别是估计的参数和由该参数计算得到的残差
  //注意这里的const，一个都不能省略，否则就会报错
  template <typename T>
  bool operator()(const T* const params,T* residual)const
  {
    residual[0]=y_guan-(params[0]*x_guan*x_guan*x_guan+params[1]*x_guan*x_guan+params[2]*x_guan+params[3]);
    return true;  
  }
};

//生成实验数据
void generateData()
{
  RNG rng;
  double w_sigma = 1.0;
  for(int i=0;i<100;i++)
  {
    double x = i;
    double y = 3.5*x*x*x+1.6*x*x+0.3*x+7.8;
    xs.push_back(x);
    ys.push_back(y+rng.gaussian(w_sigma));
  }
  for(int i=0;i<xs.size();i++)
  {
    cout<<"x:"<<xs[i]<<" y:"<<ys[i]<<endl;
  }
}

//简单描述我们优化的目的就是为了使我们估计参数算出的y'和实际观测的y的差值之和最小
//所以代价函数(CostFunction)就是y'-y，其对应每一组观测值与估计值的残差。
//由于我们优化的是残差之和，因此需要把代价函数全部加起来，使这个函数最小，而不是单独的使某一个残差最小
//默认情况下，我们认为各组的残差是等权的，也就是核函数系数为1。
//但有时可能会出现粗差等情况，有可能不等权，但这里不考虑。
//这个求和以后的函数便是我们优化的目标函数
//通过不断调整我们的参数值，使这个目标函数最终达到最小，即认为优化完成
int main(int argc, char **argv) {
  
  generateData();
  
  //创建一个长度为4的double数组用于存放参数
  double params[4]={1.0};

  //第一步，创建Problem对象，并对每一组观测数据添加ResidualBlock
  //由于每一组观测点都会得到一个残差，而我们的目的是最小化所有残差的和
  //所以采用for循环依次把每个残差都添加进来
  Problem problem;
  for(int i=0;i<xs.size();i++)
  {
    //利用我们之前写的结构体、仿函数，创建代价函数对象，注意初始化的方式
    //尖括号中的参数分别为误差类型，输出维度(因变量个数)，输入维度(待估计参数的个数)
    CostFunction* cost_function = new AutoDiffCostFunction<CostFunctor,1,4>(new CostFunctor(xs[i],ys[i]));
    //三个参数分别为代价函数、核函数和待估参数
    problem.AddResidualBlock(cost_function,NULL,params);
  }
  
  //第二步，配置Solver
  Solver::Options options;
  //配置增量方程的解法
  options.linear_solver_type=ceres::DENSE_QR;
  //是否输出到cout
  options.minimizer_progress_to_stdout=true;
  
  //第三步，创建Summary对象用于输出迭代结果
  Solver::Summary summary;
  
  //第四步，执行求解
  Solve(options,&problem,&summary);
  
  //第五步，输出求解结果
  cout<<summary.BriefReport()<<endl;
  
  cout<<"p0:"<<params[0]<<endl;
  cout<<"p1:"<<params[1]<<endl;
  cout<<"p2:"<<params[2]<<endl;
  cout<<"p3:"<<params[3]<<endl;
  return 0;
}
  ```

### 3.2.&ensp;例子-LOAM
　　{% post_link LOAM LOAM%} 前端提取线和面特征，后端最小化线和面的匹配误差。其源码实现了整个最优化过程，ALOAM<a href="#2" id="2ref"><sup>[2]</sup></a> 将后端代码用 Ceres 实现，这里对其作理解与分析。

  ```c
struct LidarEdgeFactor
{
	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<T, 3, 1> de = lpa - lpb;

		residual[0] = nu.x() / de.norm();
		residual[1] = nu.y() / de.norm();
		residual[2] = nu.z() / de.norm();

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeFactor, 3, 4, 3>(
			new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};
  ```
　　对于 Point2Line 误差，为了衡量该线特征上的点是否在地图对应的线特征上，在地图线特征上采样两个点，加上该点，组成两个向量，向量叉乘即可描述匹配误差。

```c
struct LidarPlaneFactor
{
	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		//Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
		//Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		residual[0] = (lp - lpj).dot(ljm);

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									   const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneFactor, 1, 4, 3>(
			new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
	double s;
};
```
　　对于 Point2Plane 误差，为了衡量该面特征上的点是否在地图对应的面特征上，在地图面特征上采样一个点，加上该点，组成向量，然后点乘面的法向量即可衡量匹配误差。

### 3.3.&ensp;例子-BA
```c
// copy from https://www.jianshu.com/p/3df0c2e02b4c
#include <cmath>
#include <cstdio>
#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

// Read a Bundle Adjustment in the Large dataset.
class BALProblem {
 public:
  ~BALProblem() {
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] parameters_;
  }

  int num_observations()       const { return num_observations_;               }
  const double* observations() const { return observations_;                   }
  double* mutable_cameras()          { return parameters_;                     }
  double* mutable_points()           { return parameters_  + 9 * num_cameras_; }

  double* mutable_camera_for_observation(int i) {
    return mutable_cameras() + camera_index_[i] * 9;
  }
  double* mutable_point_for_observation(int i) {
    return mutable_points() + point_index_[i] * 3;
  }

  bool LoadFile(const char* filename) {
    FILE* fptr = fopen(filename, "r");
    if (fptr == NULL) {
      return false;
    };

    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    for (int i = 0; i < num_observations_; ++i) {
      FscanfOrDie(fptr, "%d", camera_index_ + i);
      FscanfOrDie(fptr, "%d", point_index_ + i);
      for (int j = 0; j < 2; ++j) {
        FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
      }
    }

    for (int i = 0; i < num_parameters_; ++i) {
      FscanfOrDie(fptr, "%lf", parameters_ + i);
    }
    return true;
  }

 private:
  template<typename T>
  void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
      LOG(FATAL) << "Invalid UW data file.";
    }
  }

  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;

  int* point_index_;
  int* camera_index_;
  double* observations_;
  double* parameters_;
};

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = - p[0] / p[2];
    T yp = - p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    T r2 = xp*xp + yp*yp;
    T distortion = 1.0 + r2  * (l1 + l2  * r2);

    // Compute final projected point position.
    const T& focal = camera[6];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  if (argc != 2) {
    std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
    return 1;
  }

  BALProblem bal_problem;
  if (!bal_problem.LoadFile(argv[1])) {
    std::cerr << "ERROR: unable to open file " << argv[1] << "\n";
    return 1;
  }

  const double* observations = bal_problem.observations();

  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  ceres::Problem problem;
  for (int i = 0; i < bal_problem.num_observations(); ++i) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.

    ceres::CostFunction* cost_function =
        SnavelyReprojectionError::Create(observations[2 * i + 0],
                                         observations[2 * i + 1]);
    problem.AddResidualBlock(cost_function,
                             NULL /* squared loss */,
                             bal_problem.mutable_camera_for_observation(i),
                             bal_problem.mutable_point_for_observation(i));
  }

  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  return 0;
}
```
　　这里使用了 Bundle Adjustment in the Large<a href="#3" id="3ref"><sup>[3]</sup></a> 数据集，观测量为图像坐标系下路标(特征)的像素坐标系，待优化的参数为各路标的 3D 坐标以及相机内外参，这里相机内外参有 9 个，其中位置及姿态 6 个，畸变系数 2 个，焦距 1 个。

## 4.&ensp;Reference
<a id="1" href="#1ref">[1]</a> 高翔. 视觉 SLAM 十四讲: 从理论到实践. 电子工业出版社, 2017.  
<a id="2" href="#2ref">[2]</a> https://github.com/HKUST-Aerial-Robotics/A-LOAM  
<a id="3" href="#3ref">[3]</a> http://grail.cs.washington.edu/projects/bal/  
