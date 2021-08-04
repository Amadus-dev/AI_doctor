import torch
import torch.nn as nn
# 导入bert模型
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-chinese')
# 导入字符映射器
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-chinese')

def get_bert_encode_for_single(text):
    '''
    功能：使用bert-chinese预训练模型对中文文本进行编码
    :param text: 要进行编码的中文文本
    :return: 编码后的张量
    '''

    # 首先使用字符引射器对每个汉字进行映射
    # bert中的tokenizer映射器会加入开始和结束标记，101，102，这里两个标记不需要，采用切片的方式去除
    indexed_tokens = tokenizer.encode(text)[1:-1]
    # 封装成tensor张量
    tokens_tensor = torch.tensor([indexed_tokens])
   #print(tokens_tensor)

    # 预测部分需要是的模型不自动求导
    with torch.no_grad():
        encoded_layers= model(tokens_tensor)

   #print(encoded_layers[0].shape)
    # 模型的输出都是三维张量，第一位是1， 使用[0]来进行降维， 只提取我们需要的后两个维度的张量
    encoded_layers = encoded_layers[0][0]
    return encoded_layers

if __name__ == '__main__':
    text = '卢本伟牛逼！'
    outputs = get_bert_encode_for_single(text)
    print(outputs)
    print(outputs.shape)



