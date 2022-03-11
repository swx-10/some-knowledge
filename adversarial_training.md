# 对抗训练(提升模型的泛化性)
在对抗训练的过程中，给样本添加一些微小的扰动（扰动很小，但是很可能会造成误分类），然后使神经网络适应这种改变，从而对对抗样本具有鲁棒性。
对抗训练可以概括为如下的最大最小化公式：  
$$\min_{\theta}E_{(Z,y)~D}[max_{\lVert\delta\lVert\leq\epsilon}L(f_\theta(X+\delta),y)]$$
内层（中括号内）是一个最大化，其中$X$表示样本的输入表示，$\delta$表示叠加在输入上的扰动，$f_\theta()$是神经网络函数，$y$是样本的标签，$L(f_\theta(X+\delta),y)$则表示在样本$X$上叠加一个扰动$\delta$，再经过神经网络函数，与标签$y$比较得到的损失。$max(L)$是优化目标，即寻找使损失函数最大的扰动，简单来讲就是添加的扰动要尽量让神经网络迷惑。  
外层就是对神经网络进行优化的最小化公式，即当扰动固定的情况下，我们训练神经网络模型使得在训练数据上的损失最小，也就是说，使模型具有一定的鲁棒性能够适应这种扰动。  
这个公式是一个一般性的公式，并没有讲如何设计扰动。理想情况下，最好是能直接求出$\delta$，但在神经网络模型中这是不太可行的。所以大家就提出各种各样的扰动的近似求解的方法。事实上，对抗训练的研究基本上就是在寻找合适的扰动，使得模型具有更强的鲁棒性。

## NLP领域
在NLP领域，输入的是文字，本质上是one hot向量，而两个不同的one hot向量，其欧氏距离恒为$\sqrt{2}$，因此对于理论上不存在什么“小扰动”。
一个自然的想法是像论文[《Adversarial Training Methods for Semi-Supervised Text Classification》](https://arxiv.org/abs/1605.07725)一样，将扰动加到Enbedding层。这个思路在操作上没有问题，但问题是，扰动后的Embedding向量不一定能匹配上原来的Embedding向量表，这样一来对Embedding层的扰动就无法对应上真实的文本输入，这就不是真正意义上的对抗样本了，因为对抗样本依然能对应一个合理的原始输入。  
那么，在Embedding层做对抗扰动还有没有意义呢？有！实验结果显示，在很多任务中，在Embedding层进行对抗扰动能有效提高模型的性能。
## 思路分析
对于CV任务来说，一般输入张量的shape是$(b,h,w,c)$，这时候给原始输入加上一个shape相同的全零初始化的Variable，比如叫做$\Delta x$，那么我们可以直接求loss对$x$的梯度，然后根据梯度给$\Delta x$赋值，来实现对输入的干扰，完成干扰之后再执行常规的梯度下降。
对于NLP任务来说，原则上也要对Embedding层的输出进行同样的操作，Embedding层的输出shape为$(b,n,d)$，所以也要在Embedding层的输出加上一个shape为$(b,n,d)$的Variable，然后进行上述步骤。但这样一来，我们需要拆解、重构模型，对使用者不够友好。
我们可以退而求其次。Embedding层的输出是直接取自于Embedding参数矩阵的，因此我们可以直接对Embedding参数矩阵进行扰动。这样得到的对抗样本的多样性会少一些（因为不同样本的同一个token共用了相同的扰动），但仍然能起到正则化的作用，而且这样实现起来容易得多。
## 代码参考
Keras下基于FGM方式对Embedding层进行对抗训练的实现参考[苏剑林实现代码](https://github.com/bojone/keras_adversarial_training)  
核心代码如下：  
```python
def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。"""
    if model.train_function is None: #如果还没有训练函数
        model._make_train_function() #手动make
    old_train_function = model.train_function #备份旧的训练函数

    #查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    #求Embedding梯度
    embedding = embedding_layer.embeddings #Embedding矩阵
    gradients = K.gradients(model.total_loss,[embeddings]) #Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0] #转为dense tensor

    #封装函数
    inputs = (model._feed_inputs +model._feed_targets +model._feed_sample_weights) #所有输入层
    embedding_gradients = K.function(inputs = inputs, outputs = [gradients], name = 'embedding_gradients',) #封装为函数

    def train_function(inputs): #重新定义训练函数
        grads = embedding_gradients(inputs)[0] #Embedding梯度
        delta = epsilon*grads / (np.sqrt((grads**2).sum()) + 1e-8) #计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta) #注入扰动
        outputs = old_train_function(inputs) #梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta) #删除扰动
        return outputs

    model.train_function = train_function #覆盖原训练函数
```
定义好上述函数后，给Keras模型增加对抗训练就只需要一行代码了：
```python
#写好函数后，启用对抗训练只需一行代码
adversarial_training(model, 'Embedding-Token', 0.5)
```
需要指出的是，由于每一步算对抗扰动也需要计算梯度，因此每一步训练一共算了两次梯度，因此每步的训练时间会翻倍。