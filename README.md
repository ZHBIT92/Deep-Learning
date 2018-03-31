# Deep Learning
* [房价预测](#房价预测)
* [神经网络](#neural_network_Python)
* [人脸识别](#人脸识别)
* [个人博客](zhbit92.github.io)

## 房价预测
### 步骤
1. 获取数据
2. 探究数据（可视化+清洗）
3. 设计并转换特征和目标变量
4. 建立一个模型
5. 制作并提交预测

### 探究数据
#### 查看目标函数与理解相关业务
```
plt.figure()
plt.subplot(1,2,1)
plt.hist(train.SalePrice, color='blue')
plt.subplot(1,2,2)
plt.hist(target, color='blue')
plt.show()  # 展示
```
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
```
def jointplot(x,y,**kwargs):
    try:
        sns.regplot(x=x,y=y)
    except Exception:
        print(x.value_counts())
  numeric_feats = numeric_feats.drop("SalePrice")
f = pd.melt(train, id_vars=['SalePrice'],value_vars=numeric_feats)
g = sns.FacetGrid(f,col='variable',col_wrap=3,sharex=False,sharey=False,size=5)
g = g.map(jointplot,"value","SalePrice")
```

##### 类别特征
* 通过`seaborn`的`boxplot()`函数作箱形图来显示类别特征与目标变量之间的关系
```
for c in categorical_feats:
    train[c] = train[c].astype('category')
    if train[c].isnull().any():
        train[c] = train[c].cat.add_categories(["Missing"])
        train[c] = train[c].fillna("Missing")
def boxplot(x,y,**kwargs):
    sns.boxplot(x=x,y=y)
f = pd.melt(train,id_vars=['SalePrice'],value_vars=categorical_feats)
g = sns.FacetGrid(f,col='variable',col_wrap=3,sharex=False,sharey=False,size=5)
g = g.map(boxplot,"value","SalePrice")
```
##### 整体关系
* 通过`DataFrame.corr()`方法显示列之间的相关性（或关系），可以用来研究特征与目标变量的亲密程度
```
numeric_features = train.select_dtypes(include=[np.number])  
corr = numeric_features.corr()
```

* 通过`seaborn`的`heatmap()`函数作热力图显示
```
plt.subplots(figsize=(12,10))
corrmat = train.corr()
g = sns.heatmap(train.corr())
```

##### 缺失值情况
* 通过`isnull().sum()`

#### 



## neural_network_Python

* Andrew NG的Deep Learning的系列课程资料
* [深度学习与神经网络学习笔记(吴恩达)](https://zhbit92.github.io/categories/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%8E%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-%E5%90%B4%E6%81%A9%E8%BE%BE/)
## 人脸识别
