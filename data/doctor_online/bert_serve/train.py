import pandas as pd
from sklearn.utils import shuffle
from functools import reduce
from collections import Counter
from bert_chinese_encode import get_bert_encode
import torch
import torch.nn as nn

from finetuning_net import Net
import torch.optim as optim
# 导入制图工具包
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import time

def data_loader(data_path, batch_size, split=0.2):
    """
    description: 从持久化文件中加载数据, 并划分训练集和验证集及其批次大小
    :param data_path: 训练数据的持久化路径
    :param batch_size: 训练和验证数据集的批次大小
    :param split: 训练集与验证的划分比例
    :return: 训练数据生成器, 验证数据生成器, 训练数据数量, 验证数据数量
    """
    # 使用pd进行csv数据的读取
    data = pd.read_csv(data_path, header=None, sep="\t")
    # 打印整体数据集上的正负样本数量
    print("数据集的正负样本数量：")
    print(dict(Counter(data[0].values)))
    # 打乱数据集的顺序
    data = shuffle(data).reset_index(drop=True)
    #划分训练集合验证集
    split_point = int(len(data)*split)
    valid_data = data[:split_point]
    train_data = data[split_point:]

    # 验证数据集中的数据总数至少能够满足一个批次
    if len(valid_data) < batch_size:
        raise ("Batch size or split not match!")


    def _loader_generator(data):
        """
        description: 获得训练集/验证集的每个批次数据的生成器
        :param data: 训练数据或验证数据
        :return: 一个批次的训练数据或验证数据的生成器
        """
        # 以每个批次的间隔遍历数据集
        for batch in range(0, len(data), batch_size):
            # 预定于batch数据的张量列表
            batch_encoded = []
            batch_labels = []
            # 将一个bitch_size大小的数据转换成列表形式，[[label, text_1, text_2]]
            # 并进行逐条遍历
            for item in data[batch:batch+batch_size].values.tolist():
                # 每条数据中都包含两句话, 使用bert中文模型进行编码
                encoded = get_bert_encode(item[1], item[2])
                # 将编码后的每条数据装进预先定义好的列表中
                batch_encoded.append(encoded)
                # 同样将对应的该batch的标签装进labels列表中
                batch_labels.append([item[0]])
                # 使用reduce高阶函数将列表中的数据转换成模型需要的张量形式
                # encoded的形状是(batch_size, 2*max_len, embedding_size)
                encoded = reduce(lambda x, y:torch.cat((x, y), dim=0), batch_encoded)
                labels = torch.tensor(reduce(lambda x, y: x + y, batch_labels))
                # 以生成器的方式返回数据和标签

                yield (encoded, labels)

    # 对训练集和验证集分别使用_loader_generator函数, 返回对应的生成器
    # 最后还要返回训练集和验证集的样本数量
    return _loader_generator(train_data), _loader_generator(valid_data), len(train_data), len(valid_data)

# 数据所在路径
data_path = "./train_data.csv"
# 定义batch_size大小
batch_size = 32
# 构建模型训练函数
# 定义embedding_size, char_size
embedding_size = 768
char_size = 20

# 实例化微调网络
net = Net(embedding_size, char_size)
# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义SGD优化方法
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train(train_data_labels):
    """
    description: 训练函数, 在这个过程中将更新模型参数, 并收集准确率和损失
    :param train_data_labels: 训练数据和标签的生成器对象
    :return: 整个训练过程的平均损失之和以及正确标签的累加数
    """
    # 定义训练过程的初始损失和准确率累加数
    train_running_loss = 0.0
    train_running_acc = 0.0

    # 循环遍历训练数据和标签生成器, 每个批次更新一次模型参数
    for train_tensor, train_labels in train_data_labels:
        # 初始化该批次的优化器
        optimizer.zero_grad()
        # 使用微调网络输出
        train_outputs = net(train_tensor)
        # 得到该批次下的平均损失
        train_loss = criterion(train_outputs, train_labels)
        # 将该批次的平均损失加到train_running_loss中
        train_running_loss += train_loss.item()
        # 损失反向传播
        train_loss.backward()
        # 优化器更新模型
        optimizer.step()
        # 将该批次中正确的标签数量进行累加，以便以后计算准确率
        train_running_acc += (train_outputs.argmax(1) == train_labels).sum().item()
    return train_running_loss, train_running_acc

def valid(valid_data_labels):
    """
    description: 验证函数, 在这个过程中将验证模型的在新数据集上的标签, 收集损失和准确率
    :param valid_data_labels: 验证数据和标签的生成器对象
    :return: 整个验证过程的平均损失之和以及正确标签的累加数
    """

    # 定义训练过程的初始损失和准确率累加数
    valid_running_loss = 0.0
    valid_running_acc = 0.0
    # 循环遍历验证数据和标签生成器
    for valid_tensor, valid_labels in valid_data_labels:
        # 不自动更新梯度
        with torch.no_grad():
            # 使用微调网络获得输出
            valid_outputs = net(valid_tensor)
            # 得到该批次下的平均损失
            valid_loss = criterion(valid_outputs, valid_labels)
            # 将该批次的平均损失加到valid_running_loss中
            valid_running_loss += valid_loss.item()
            # 将该批次中正确的标签数量进行累加, 以便之后计算准确率
            valid_running_acc += (valid_outputs.argmax(1) == valid_labels).sum().item()

    return valid_running_loss, valid_running_acc


def start_training():
    # 定义训练轮数
    epochs = 20

    # 定义盛装每轮次的损失和准确率列表, 用于制图
    all_train_losses = []
    all_valid_losses = []
    all_train_acc = []
    all_valid_acc = []

    # 进行指定轮次的训练
    for epoch in range(epochs):
        # 打印轮次
        print("Epoch:", epoch + 1)
        # 通过数据加载器获得训练数据和验证数据生成器, 以及对应的样本数量
        train_data_labels, valid_data_labels, train_data_len, valid_data_len = data_loader(data_path, batch_size)
        # 调用训练函数进行训练
        train_running_loss, train_running_acc = train(train_data_labels)
        # 调用验证函数进行验证
        valid_running_loss, valid_running_acc = valid(valid_data_labels)
        # 计算每一轮的平均损失, train_running_loss和valid_running_loss是每个批次的平均损失之和
        # 因此将它们乘以batch_size就得到了该轮的总损失, 除以样本数即该轮次的平均损失
        train_average_loss = train_running_loss * batch_size / train_data_len
        valid_average_loss = valid_running_loss * batch_size / valid_data_len

        # train_running_acc和valid_running_acc是每个批次的正确标签累加和,
        # 因此只需除以对应样本总数即是该轮次的准确率
        train_average_acc = train_running_acc / train_data_len
        valid_average_acc = valid_running_acc / valid_data_len
        # 将该轮次的损失和准确率装进全局损失和准确率列表中, 以便制图
        all_train_losses.append(train_average_loss)
        all_valid_losses.append(valid_average_loss)
        all_train_acc.append(train_average_acc)
        all_valid_acc.append(valid_average_acc)
        # 打印该轮次下的训练损失和准确率以及验证损失和准确率
        print("Train Loss:", train_average_loss, "|", "Train Acc:", train_average_acc)
        print("Valid Loss:", valid_average_loss, "|", "Valid Acc:", valid_average_acc)

    print('Finished Training')

    # 创建第一张画布
    plt.figure(0)

    # 绘制训练损失曲线
    plt.plot(all_train_losses, label="Train Loss")
    # 绘制验证损失曲线, 颜色为红色
    plt.plot(all_valid_losses, color="red", label="Valid Loss")
    # 定义横坐标刻度间隔对象, 间隔为1, 代表每一轮次
    x_major_locator = MultipleLocator(1)
    # 获得当前坐标图句柄
    ax = plt.gca()
    # 设置横坐标刻度间隔
    ax.xaxis.set_major_locator(x_major_locator)
    # 设置横坐标取值范围
    plt.xlim(1, epochs)
    # 曲线说明在左上方
    plt.legend(loc='upper left')
    # 保存图片
    plt.savefig("./loss.png")

    # 创建第二张画布
    plt.figure(1)

    # 绘制训练准确率曲线
    plt.plot(all_train_acc, label="Train Acc")

    # 绘制验证准确率曲线, 颜色为红色
    plt.plot(all_valid_acc, color="red", label="Valid Acc")
    # 定义横坐标刻度间隔对象, 间隔为1, 代表每一轮次
    x_major_locator = MultipleLocator(1)
    # 获得当前坐标图句柄
    ax = plt.gca()
    # 设置横坐标刻度间隔
    ax.xaxis.set_major_locator(x_major_locator)
    # 设置横坐标取值范围
    plt.xlim(1, epochs)
    # 曲线说明在左上方
    plt.legend(loc='upper left')
    # 保存图片
    plt.savefig("./acc.png")

    # 模型保存时间
    time_ = int(time.time())
    # 保存路径
    MODEL_PATH = './model/BERT_net_%d.pth' % time_
    # 保存模型参数
    torch.save(net.state_dict(), MODEL_PATH)


if __name__ == '__main__':
    # 定义实例化模型参数
    embedding_size = 768
    char_size = 20
    dropout = 0.2
    MODEL_PATH = "./model/BERT_net.pth"
    # 初始化微调网络模型
    net = Net(embedding_size, char_size, dropout)
    # 加载模型参数
    net.load_state_dict(torch.load(MODEL_PATH))
    # 使用评估模式
    net.eval()

    text_1 = "左侧的包块是否是普通的淋巴结肿大？"
    text_2 = "五险一金会下调吗"

    # 对原始文本进行编码
    inputs = get_bert_encode(text_1, text_2, mark=102, max_len=10)
    # 使用微调模型进行预测
    outputs = net(inputs)
    # 获得预测结果
    _, predicted = torch.max(outputs, 1)
    # 返回字符串类型的结果
    print(str(predicted.item()))



