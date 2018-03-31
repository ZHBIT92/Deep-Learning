# Deep Learning
* [房价预测](#房价预测)
* [神经网络](#neural_network_Python)
* [人脸识别](#人脸识别)
* [个人博客](zhbit92.github.io)

## 房价预测
通过数据挖掘、可视化、清洗的方法获取房产数据集中与房价有关的特征，建立机器学习模型，多个模型拟合得到最优参数，实现对房产价格的预测。

### 文件描述
* data：相关数据集
* info.txt：有关参数的介绍
* feature_show：可视化
* feature_predict：预测

### 探究数据
#### 查看目标函数与理解相关业务
![](http://p1mjzrkoc.bkt.clouddn.com/blog/180331/H6ibFfcak0.png?imageslim)

#### 数据分类-> 数值特征和类别特征
* 方法一
```
select_dtypes(include=[np.number])
```
* 方法二
```
features = pd.concat([train, test],keys=['train','test'])
numeric_feats = features.dtypes[features.dtypes!="object"].index
categorical_feats = features.dtypes[features.dtypes=="object"].index
```

#### 查看特征与目标变量的关系
##### 数值特征
* 通过`seaborn`的`regplot`函数作箱形图来显示类别特征与目标变量之间的关系

##### 类别特征
* 通过`seaborn`的`boxplot()`函数作箱形图来显示类别特征与目标变量之间的关系

##### 整体关系
* 通过`DataFrame.corr()`方法显示列之间的相关性（或关系），可以用来研究特征与目标变量的亲密程度


* 通过`seaborn`的`heatmap()`函数作热力图显示


##### 缺失值情况


## neural_network_Python

* Andrew NG的Deep Learning的系列课程资料
* [深度学习与神经网络学习笔记(吴恩达)](https://zhbit92.github.io/categories/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%8E%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-%E5%90%B4%E6%81%A9%E8%BE%BE/)
## 人脸识别
