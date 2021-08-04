import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """定义微调网络的类"""
    def __init__(self, char_size=20, embedding_size=768, dropout=0.2):
        """
        :param char_size: 输入句子中的字符数量, 因为规范后每条句子长度是max_len, 因此char_size为2*max_len
        :param embedding_size: 字嵌入的维度, 因为使用的bert中文模型嵌入维度是768, 因此embedding_size为768
        :param dropout: 为了防止过拟合, 网络中将引入Dropout层, dropout为置0比率, 默认是0.2
        """
        super(Net, self).__init__()
        # 将char_size和embedding_size传入其中
        self.char_size = char_size
        self.embedding_size = embedding_size
        # 实例化化必要的层和层参数:
        # 实例化Dropout层
        self.dropout = nn.Dropout(p=dropout)
        # 实例化第一个全连接层
        self.fc1 = nn.Linear(char_size*embedding_size, 8)
        # 实例化第二个全连接层
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        # 对输入的张量形状进行变换, 以满足接下来层的输入要求
        x = x.view(-1, self.char_size*self.embedding_size)
        # 使用dropout层
        x = self.dropout(x)
        # 使用第一个全连接层并使用relu函数
        x = F.relu(self.fc1(x))
        # 使用dropout层
        x = self.dropout(x)
        # 使用第二个全连接层并使用relu函数
        x = F.relu(self.fc2(x))
        return x
if __name__ == '__main__':

    embedding_size = 768
    char_size = 20
    dropout = 0.2

    x = torch.randn(1, 20, 768)
    net = Net(char_size, embedding_size, dropout)
    nr = net(x)
    print(nr)




