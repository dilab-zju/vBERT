import json
import socket
import time

if __name__ == '__main__':
    ip_port = ('127.0.0.1', 9999)

    s = socket.socket()

    s.connect(ip_port)

    while True:
        inp = "{'id': 3, 'seq': 'i love cute dogs', 'domain_name': 'news'}"
        dict_data = eval(inp)
        json_data = json.dumps(dict_data)
        s.sendall(json_data.encode('utf-8'))

        server_reply = s.recv(1024).decode()
        print(server_reply)
        time.sleep(3)

    s.close()       # 关闭连接
