## TextRank：一种用于文本的基于图的排序算法
TextRank的思想比较简单，即通过词之间的相邻关系构建网络，然后用PageRank迭代计算每个点的rank值，排序rank值即可得到关键词。

### PageRank
PageRank算法用于解决互联网网页的价值排序问题，网页之间的链接关系即为图的边迭代计算公式如下：  
![PageRank公式](.%5Cpicture%5CTextRank%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F.png)  
其中，$S(V_i)$表示节点$V_i$的值$In(V_i)$表示节点Vi的前驱节点集合，$Out(V_j)$表示节点Vj的后继节点集合，d为阻尼因子用于做平滑。  

TextRank算法是一种基于图的用于关键词抽取和文档摘要的排序算法，由谷歌的网页重要性排序算法PageRank算法改进而来，它利用一篇文档内部的词语间的共现信息(语义)便可以抽取关键词，它能够从一个给定的文本中抽取出该文本的关键词、关键词组，并使用抽取式的自动文摘方法抽取出该文本的关键句。TextRank算法与PageRank算法的相似之处是：
>用句子替代网页；
任意两个句子的相似性等价于网页转换概率；
相似性得分存储再一个方形矩阵中，类似于PageRank的转移概率矩阵。  

TextRank将文档看作一个词的网络，该网络中的来凝结表示词与词之间的语义关系。
TextRank算法计算公式：
![TextRank计算公式](.\picture\TextRank计算公式.png)  
TextRank算法论文：[TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
TextRank算法主要包括：关键词抽取、关键短语抽取、关键句抽取。

TextRank图算法：
```python
from collections import defaultdict
import sys
class TextRank_Graph:
    def __init__(self):
        self.graph = defultdict(list)
        self.d = 0.85 #阻尼系数，一般为0.85
        self.min_diff = 1e-5 #设定收敛阈值

    #添加节点之间的边
    def add_edge(self, start, end, weight):
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    #节点排序
    def rank(self):
        #默认初始化权重
        weight_default = 1.0 / (len(self.graph)) or 1.0
        #nodeweight_dict，存储节点的权重
        nodeweight_dict = defaultdict(float)
        #outsum_node_dict，存储节点的出度权重
        outsum_node_dict = defaultdict(float)
        #根据图中的边，更新节点权重
        for node, out_edge in self.graph.items():
            #是 [('是', '全国', 1), ('是', '调查', 1), ('是', '失业率', 1), ('是', '城镇', 1)]
            nodeweight_dict[node] = weight_default
            outsum_node_dict[node] = sum((edge[2] for edge in out_edge), 0.0)
        #初始状态下的textrank重要性权重
        sorted_keys = sorted(self.graph.keys())
        #设定迭代次数
        step_dict = [0]
        for step in range(1, 1000):
            for node in sorted_keys:
                s = 0
                #计算公式(edge_weight/outsum_node_dict[edge_node])*node_weight[edge_node]
                for e in self.graph[node]:
                    s += e[2] / outsum_node_dict[e[1]] * nodeweight_dict[e[1]]
                #计算公式：(1-d) + d*s
                nodeweight_dict[node] = (1 - self.d) + self.d * s
            step_dict.append(sum(nodeweight_dict.values()))

            if abs(step_dict[step] - step_dict[step - 1]) <= self.min_diff:
                break
        
        #利用Z-score进行权重归一化，也称为离差标准化，是对原始数据的线性变换，使结果映射到[0 - 1]之间。
        #先设定最大值与最小值均为系统存储的最大值和最小值
        (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])
        for w in nodeweight_dict.values():
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w

        for n,w in nodeweight_dict.items():
            nodeweight_dict[n] = (w - min_rank/10.0) / (max_rank - min_rank/10.0)

        return nodeweight_dict
```
