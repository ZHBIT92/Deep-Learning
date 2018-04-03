# Deep Learning
* [Kaggle房价预测](#Kaggle房价预测)  
![mark](http://p1mjzrkoc.bkt.clouddn.com/blog/180403/BL3jk0fGdi.png?imageslim)

## Kaggle房价预测
通过数据挖掘、可视化、清洗的方法获取房产数据集中与房价有关的特征，建立机器学习模型，多个模型拟合得到最优参数，实现对房产价格的预测。
* [文件描述](#文件描述)
* [探究数据](#探究数据)
* [训练模型](#训练模型)


### 文件描述
* data：相关数据集
* info.txt：有关参数的介绍
* feature_show：可视化
* feature_predict：预测

### 探究数据
#### 查看目标函数与理解相关业务
```
count      1460.000000
mean     180921.195890
std       79442.502883
min       34900.000000
25%      129975.000000
50%      163000.000000
75%      214000.000000
max      755000.000000
Name: SalePrice, dtype: float64
```
![mark](http://p1mjzrkoc.bkt.clouddn.com/blog/180331/mFB5CKFKdc.png?imageslim)

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
* 通过`seaborn`的`jointplot`函数作散点图来显示数值特征与目标变量之间的关系（部分举例）
![mark](http://p1mjzrkoc.bkt.clouddn.com/blog/180331/GLL1aDDel3.png?imageslim)

##### 类别特征
* 通过`seaborn`的`boxplot()`函数作箱形图来显示类别特征与目标变量之间的关系（部分举例）
![mark](http://p1mjzrkoc.bkt.clouddn.com/blog/180331/Ba4ajKfg5e.png?imageslim)

##### 整体关系
* 通过`DataFrame.corr()`方法显示列之间的相关性（或关系），可以用来研究特征与目标变量的亲密程度
```
相关性前5
OverallQual    0.790982
GrLivArea      0.708624
GarageCars     0.640409
GarageArea     0.623431
TotalBsmtSF    0.613581
Name: SalePrice, dtype: float64 

相关性-5
YrSold          -0.028923
OverallCond     -0.077856
MSSubClass      -0.084284
EnclosedPorch   -0.128578
KitchenAbvGr    -0.135907
Name: SalePrice, dtype: float64 
```
* 通过`seaborn`的`heatmap()`函数作热力图显示
![mark](http://p1mjzrkoc.bkt.clouddn.com/blog/180331/6JA27IjA93.png?imageslim)

##### 缺失值情况
![mark](http://p1mjzrkoc.bkt.clouddn.com/blog/180331/69JLc3GhGK.png?imageslim)

### 训练模型
* Ridge模型

![mark](http://p1mjzrkoc.bkt.clouddn.com/blog/180331/kLf32b5iLD.png?imageslim)

* Lasso模型

![mark](http://p1mjzrkoc.bkt.clouddn.com/blog/180331/480ajDlgIc.png?imageslim)
