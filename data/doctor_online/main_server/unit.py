import json
import random
import requests

client_id = "FG6ZkhgvwkLN4ivdl5rLHp5o"
client_secret = "948m7X0c0n9FVWQh9MKUM1UxyATKYn48"

def unit_chat(chat_input, user_id="88888"):
    '''
    进行聊天
    :param chat_input:用户输入的聊天语句
    :param user_id: 该用户的ID
    :return: 返回百度Unit机器人的回复语句
    '''
    # 给一个默认的回复
    chat_reply = "不好意思，我们正在学习中，随后回复您。"
    # 得到访问的access_token
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s" % (
        client_id, client_secret)
    res = requests.get(url)
    access_token = eval(res.text)["access_token"]
    # 利用刚刚得到的access_token去访问聊天URL
    unit_chatbot_url = "https://aip.baidubce.com/rpc/2.0/unit/service/chat?access_token=" + access_token

    # 构建我们的访问数据
    post_data = {
                "log_id": str(random.random()),
                "request": {
                    "query": chat_input,
                    "user_id": user_id
                },
                "session_id": "",
                "service_id": "S55225",
                "version": "2.0"
            }
    # 访问第二个URL，得到Unit机器人的回复
    res = requests.post(url=unit_chatbot_url, json=post_data)
    # 得到的res中有一个字段content，里面放着真正的返回信息
    unit_chat_obj = json.loads(res.content)
    # 判断聊天接口的返回是否正常，靠error_code字段，当不等于0时说明访问发生了错误
    if unit_chat_obj["error_code"] != 0:
        return chat_reply

    # 解析数据的过程
    # result -> response_list -> (schema->intent_confidence>0.0)
    unit_chat_obj_result = unit_chat_obj["result"]
    unit_chat_response_list = unit_chat_obj_result["response_list"]

    # 随机选择一个意图置信度大于0的回复作为最终机器人的回复语句
    unit_chat_response_obj = random.choice([unit_chat_response for unit_chat_response in unit_chat_response_list
                                          if unit_chat_response["schema"]["intent_confidence"]>0.0])
    # 还需要进一步的提取数据
    unit_chat_response_action_list = unit_chat_response_obj["action_list"]
    unit_chat_response_action_obj = random.choice(unit_chat_response_action_list)
    unit_chat_response_say = unit_chat_response_action_obj["say"]
    print("sssss")
    return unit_chat_response_say

if __name__ == '__main__':
    while True:
        chat_input = input("请输入: ")
        if chat_input == 'Q' or chat_input == 'q':
            print("Unit回复>> 再见")
            break
        chat_reply = unit_chat(chat_input)
        print("用户输入>>", chat_input)
        print("Unit回复>>", chat_reply)
