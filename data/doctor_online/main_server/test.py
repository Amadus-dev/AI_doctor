import requests
# 定义请求url和传入的data
url = "http://0.0.0.0:5000/v1/main_serve/"

while True:
    chat_input = input("请输入:")
    data = {"uid":"13424", "text": chat_input}
    chat_reply = requests.post(url, data=data)
    print("用户输入 >>>", chat_input)
    print("Unit回复 >>>", chat_reply.text)
    if chat_input == 'q' or chat_input == 'Q':
        break

