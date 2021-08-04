import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''

        :param input_size: 输入张量的最后一位的尺寸大小
        :param hidden_size: 隐层张量最后一位的尺寸大小
        :param output_size: 输出张量的最后一位的尺寸大小
        '''

        super(RNN, self).__init__()

        # 传入隐藏层的尺寸大小
        self.hidden_size = hidden_size
        # 构建从输入到隐藏层的线性变化，这个线性层的输入尺寸是input_size+hidden_size
        # 这是因为在循环网络中，每次输入都有两个部分组成，分别是此时可的输入和上一时刻产生的输出
        # 这个线性层的输出尺寸是hidden_size
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        ## 构建从输入到输出层的线性变化, 这个线性层的输入尺寸还是input_size + hidden_size
        # 这个线性层的输出尺寸是output_size.
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # 最后需要对输出做softmax处理, 获得结果.
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        '''

        :param input:输入张量
        :param hidden:初始化隐层张量
        :return:
        '''
        # 首先使用torch.cat将input与hidden进行张量拼接
        combined = torch.cat((input, hidden), 1)
        # 通过输入层到隐层变换获得hidden张量
        hidden = self.i2h(combined)
        # 通过输入到输出层变换获得output张量
        output = self.i2o(combined)
        # 对输出进行softmax处理
        output = self.softmax(output)
        # 返回输出张量和最后的隐层结果
        return output, hidden

    def initHidden(self):
        """隐层初始化函数"""
        # 将隐层初始化成为一个1xhidden_size的全0张量
        return torch.zeros(1, self.hidden_size)

if __name__ == '__main__':
    # 实例化参数
    input_size = 768
    hidden_size = 128
    n_categories = 2

    # 输入参数

    input = torch.rand(1, input_size)
    hidden = torch.rand(1, hidden_size)
    rnn = RNN(input_size, hidden_size, n_categories)
    outputs, hidden = rnn(input, hidden)
    print("outputs:", outputs)
    print("hidden:", hidden)

