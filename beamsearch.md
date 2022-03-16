## Beam search(集束算法)
#### 1、算法思想
beam search尝试在广度优先搜索基础上，引入搜索空间约束(设置beam width超参数)，达到减少内存消耗的目的。简单来说，每次选取概率最大的beam width个词组作为结果，并将他们分别传入下一个时刻的解码阶段进行解码得到新的组合序列，再从新的组合序列中选取最大的beam width个词组，一直循环到结束。  
其中，设此表大小为N，beam width为K，解码所需时间步为T：  
beam width超参，表示每次保存的概率最大的个数；  
beam width=1时，等同于greedy search贪心算法，时间复杂度为O(NT)；
beam width=2时，表示每次保存2个概率最大结果，时间复杂度为O(2NT)，局部最优；
beam width=N时，当与候选词表大小相同时，就是计算全部的概率，即维特比算法，时间复杂度为O(NNT)，此时为全局最优。  
![](https://github.com/swx-10/some-knowledge/blob/master/picture/beamsearch.png)
以上图中的"我爱你"为例，过程如下：  
整个词表共有【"我","爱","你"】三个词，这个就是搜索空间，也是每一步参与组合的词语集合。
>第一步："我"和"你"的预测分数最高，因此会保留"我"和"你"；  
第二步：分别利用"我"和"你"，与"我"、"爱"、"你"三个词组合，生成序列，通过计算发现"我爱"和"你爱"的得分最高，因此保留"我爱"和"你爱"；
第三步：重复步骤二，分别利用"我爱"和"你爱",与"我"、"爱"、"你"三个词组合，生成序列，通过计算发现，"我爱你"和"你爱我"得分最高，因此，最终输出这两个结果，任务结束。  

#### 2、实现
根据上述思想，对于输入状态矩阵data，设置beam_width为k，设定概率得分函数作为路径得分函数，可以快速地实现，但其中路径得分函数十分关键，不同的函数会得到不同的结果，下面介绍主流的三种方法。
##### 1）概率连乘最大化
概率连乘最大化是最为简单粗暴的方法，直接将每一步的概率相乘即可，但如果一个句子很长，那么这个句子的概率会很低，因为每个概率$P(y∣x)∈(0,1]$，每个条件概率都是小于1，多个小于1的数值相乘，会造成数值下溢，这个评分函数会倾向于短的输出，因为短句子的概率由更少数的小于1的数字乘积得到。
![连乘概率最大化](https://github.com/swx-10/some-knowledge/blob/master/picture/%E8%BF%9E%E4%B9%98%E6%A6%82%E7%8E%87%E6%9C%80%E5%A4%A7%E5%8C%96.png)
##### 2）概率对数最大化
为了解决数值累积下溢出的问题，一般不能最大化这个乘积，而是取log值，最大化$log^{P(y∣x)}$求和的概率值，最大化$log^{P(y∣x)}$等价于最大化$P(y∣x)$。  
对数函数（log函数）是严格单调递增的函数，而且$logP(y|x)$为负值，如果句子很长，加起来的项很多，结果越为负。所以这个目标函数也会影响较长的输出，因为长句子一直累加目标函数的值会十分小。
![概率对数和最大化](https://github.com/swx-10/some-knowledge/blob/master/picture/%E6%A6%82%E7%8E%87%E5%AF%B9%E6%95%B0%E5%92%8C%E6%9C%80%E5%A4%A7%E5%8C%96.png)
##### 3）加入句子长度惩罚的概率对数和最大化
为了进一步降低概率对数和最大化带来的影响，可以进一步引入归一化的思想，降低输出长句子的惩罚。
![加入句子长度惩罚的概率对数和最大化](https://github.com/swx-10/some-knowledge/blob/master/picture/%E5%8A%A0%E5%85%A5%E5%8F%A5%E5%AD%90%E9%95%BF%E5%BA%A6%E6%83%A9%E7%BD%9A%E7%9A%84%E6%A6%82%E7%8E%87%E5%AF%B9%E6%95%B0%E5%92%8C%E6%9C%80%E5%A4%A7%E5%8C%96.png)
如上述例子所示，引入超参$α∈[0,1]$，一般取$α=0.7$。当$α=0$时，不归一化，当$α=1$时，进行标准的长度归一化。

对概率对数和最大化进行实现：
```python
from math import log

#beam search
def beam_search(data, k):
    #始终维护一个长度为k的sequence
    sequences = [[list(), 1.0]] #[[[], 1.0]],初始长度是1 内部列表 第一个元素是所选的index列表，第二个元素是概率的乘积

    for idx,row in enumerate(data):
        all_candidates = list()
        #每一步组合新序列
        for i in range(len(sequences)):
            seq,score = sequences[i]
            for j in range(len(row)):
                candidate = [seq+[j], score-log(row[j])]
                all_candidates.append(candidate)
        print('step %s'%idx, all_candidates)
        #由于概率值小于1，所以-log(score)随着score的增加而减小，整体score越小，概率越大，直接进行排序
        ordered = sorted(all_candidates, key= lambda tup:tup[1])
        #每次输入k个概率最大的序列
        sequences = ordered[:k] #只取前k个
    return sequences
```
### 总结
beam search实际上是增加了搜索空间，但也只能做到局部最优解，不一定是全局最优解。  
理论上来说，只有beam width等于词表的大小时才能找到全局最优解，但不存在，应用上十分困难，工业落地上选用beam search，是效果和性能的一种妥协。

### 参考文献
https://mp.weixin.qq.com/s/_cecu05STwzF5n3fpiB3yQ
