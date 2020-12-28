# 生成数据集
from mxnet import autograd, nd

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random_normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random_normal(scale=0.01, shape=labels.shape)

# 读取数据
from mxnet.gluon import data as gdata

batch_size = 10
# 将训练数据的特征和标签组合起来
dataset = gdata.ArrayDataset(features, labels)
# 随机读取小批量数据，此处的batch_sizec = 10
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
for X, y in data_iter:
    print(X, y)
    break

#定义模型
from mxnet.gluon import nn

# Sequential实例类似一个容器，net.add()则是往其中添加层
net = nn.Sequential()
net.add(nn.Dense(1))

# 初始化模型参数
from mxnet import init

# 指定权重参数每个元素将在初始化时随机采样于均值为0， 标准差为0.01的正态分布
net.initialize(init.Normal(sigma=0.01))

# 定义损失函数
from mxnet.gluon import loss as gloss

# 平方损失函数--README.md-1
loss = gloss.L2Loss()

# 定义优化算法
from mxnet import gluon

# 小批量随机梯度下降--README.md-2
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.03})

# 训练模型
num_epoch = 10
for epoch in range(1, num_epoch+1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
