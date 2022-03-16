## LDA(线性判别分析)
这里说的LDA不是NLP中的LDA，NLP中的LDA是隐含狄利克雷分布(Latent Dirichlet Allocation，简称LDA)，它是一种处理文档的主题模型。在本文中只讨论线性判别分析，因此后面所有的LDA均指线性判别分析。
LDA是一种有监督的线性分类器，更常见的是为后续的分类做降维处理。因此它的数据集的每个样本是有类别输出的。LDA的思想可以概括为：“投影后类内方差最小，类间方差最大”，即使得不同类别之间的距离越远越好，同一类别之中的距离越近越好。  
可能还是有点抽象，我们先看看最简单的情况。假设我们有两类数据 分别为红色和蓝色，如下图所示，这些数据特征是二维的，我们希望将这些数据投影到一维的一条直线，让每一种类别数据的投影点尽可能的接近，而红色和蓝色数据中心之间的距离尽可能的大。
![LDA投影示意图](./picture/LDA1.png)  
上图中国提供了两种投影方式，哪一种能更好的满足我们的标准呢？从直观上可以看出，右图要比左图的投影效果好，因为右图的红色数据和蓝色数据各个较为集中，且类别之间的距离明显。左图则在边界处数据混杂。以上就是LDA的主要思想了，当然在实际应用中，我们的数据是多个类别的，我们的原始数据一般也是超过二维的，投影后的也一般不是直线，而是一个低维的超平面。  

#### LDA算法流程
具体的公式原理，可以参考末尾的参考文献。现在归纳一下LDA降维的算法流程：
输入：数据集D = {(x1,y1),(x2,y2),...,(xm,ym)}，其中任意样本xi为n维向量，yi∈{C1,C2,...,CK}，降维到的维度d。  
输出：降维后的样本集D'。
>1)计算类内散度矩阵$S_w$
2)计算类间散度矩阵$S_b$
3)计算矩阵$S_w^{-1}S_b$
4)计算$S_w^{-1}S_b$最大的d个特征值和对应的d个特征向量($w_1,w_2,...,w_d$)，得到投影矩阵W
5)对样本集中的每一个样本特征$x_i$，转化为新的样本特征$z_i=W^Tx_i$
6)得到输出样本集D'={$(z_1,y_1),(z_2,y_2),...,(z_m,y_m)$}

#### 代码实现
```python
import numpy as np

def lda(data, target, n_dim):
    """
    data:(n_samples, n_features)
    target:数据类别
    n_dim:目标维度
    return:
    (n_samples, n_dim)
    """
    clusters = np.unique(target)

    if n_dim > len(clusters) - 1:
        print("K is too much")
        print("please input again")
        exit(0)
    
    #类内散度矩阵
    Sw = np.zeros((data.shape[1], data.shape[1]))
    for i in clusters:
        datai = data[target == i]
        datai = datai - datai.mean(0)
        Swi = np.mat(datai).T * np.mat(datai) #np.mat 将输入解释为矩阵
        Sw += Swi
    
    #类间散度矩阵
    Sb = np.zeros((data.shape[1], data.shape[1]))
    u = data.mean(0) #所有样本的平均值
    for i in clusters:
        Ni = data[target == i].shape[0]
        ui = data[target == i].mean(0) #某个类别的平均值
        Sbi = Ni*np.mat(ui-u).T * np.mat(ui-u)
        Sb += Sbi
    S = np.linalg.inv(Sw) * Sb #np.linalg.inv 矩阵求逆
    eigVals, eigVects = np.linalg.eig(S) #求特征值，特征向量
    eigInd = np.argsort(eigVals)
    eigInd = eigInd[:(-n_dim-1):-1]
    w = eigVects[:,eigInd]
    data_ndim = np.dot(data, w)
```

#### 小结
LDA算法既可以用来降维，也可以用来分类，就目前来说，主要还是用于降维。下面总结下LDA算法的优缺点：  
LDA算法的主要优点有：  
>1）在降维的过程中可以使用类别的先验知识。
2）LDA在样本分类信息依赖均值而不是方差的时候，比PCA之类的算法较优。

LDA算法的主要缺点有：
>1）LDA不适合对非高斯分布样本进行降维，PCA也有这个问题。
2）LDA降维最多降到类别数k-1的维数，如果我们降维的维度大于k-1，则不能使用LDA。当然目前有一些LDA的进化版算法可以绕过这个问题。
3）LDA在样本分类信息依赖方差而不是均值的时候，降维效果不好。
4）LDA可能过度拟合数据。

## 参考文献
https://www.cnblogs.com/pinard/p/6244265.html
https://github.com/heucoder/dimensionality_reduction_alo_codes/blob/master/codes/LDA/LDA.py