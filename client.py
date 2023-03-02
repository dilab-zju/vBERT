import requests
import json

if __name__ == '__main__':

    dic = {
        "id": 3,
        "domain_name": "news",
        "seq": "i love cute dogs"
    }
    url = "http://0.0.0.0:9989/inference"
    res = requests.post(url, json.dumps(dic))
    print(res.text)

    s.close()       
