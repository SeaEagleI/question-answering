# -*- coding: utf-8 -*-
""" 问答系统API的HTTP服务端程序 """
import http.server
from urllib.parse import unquote
import json
import socket

import torch
from transformers import AlbertTokenizer, AlbertConfig, AlbertForQuestionAnswering
from lib import run_prediction

# config
data_dir = 'squad_v2'
model_paths = ['./albert-xxlarge-v2/finetuned_ckpt_4folds{}_epoch2_lr1e-5'.format(fid)  # use fold2~4 for ensemble
               for fid in range(2, 5)]

# setup model
config_class, model_class, tokenizer_class = (
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer
)
config = config_class.from_pretrained(model_paths[0])
tokenizer = tokenizer_class.from_pretrained(model_paths[0], do_lower_case=True)
models = [model_class.from_pretrained(model_path, config=config) for model_path in model_paths]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = [model.to(device) for model in models]


class TestHTTPHandle(http.server.BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(10)
        http.server.BaseHTTPRequestHandler.setup(self)

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        buf = 'Neural Question-Answering API'
        self.protocal_version = 'HTTP/1.1'
        self._set_response()
        buf = bytes(buf, encoding="utf-8")
        self.wfile.write(buf)

    def do_POST(self):
        """ 处理通过POST方式传递过来并接收的文本数据, 通过问答模型计算得到答案并返回 """
        # 获取post提交的数据
        datas = self.rfile.read(int(self.headers['content-length'])).decode('utf-8')  # bytes => url format str (BASE64)
        # print({'url': datas})  # log
        datas_split = datas.split('&')
        token, context, question, return_text = '', '', '', ''
        try:
            for line in datas_split:
                key, value = [unquote(text.replace('+', ' ')) for text in line.split('=')]  # BASE64 => original str
                if key == 'token':
                    token = value
                elif key == 'context' and value != '':
                    context = value
                elif key == 'question':
                    question = value
                else:
                    print(key, value)
            if token != 'qa_en':
                buf = 'Error: 403'
                # print(buf)
                buf = bytes(buf, encoding="utf-8")
                self.wfile.write(buf)
                return
            if len(context) > 0:
                answer_text, score = run_prediction(question, context, models, tokenizer)  # get predict result
                return_text = answer_text  # only return answer_text without score, but record both in api log
                example_data = {
                    "context": context,
                    "question": question,
                    "answer_text": answer_text,
                    "score": score
                }
                print(json.dumps(example_data, indent=4))  # log
        except Exception as e:
            return_text = 'Error: {}\n{}\nData: {}'.format(type(e), e, datas_split)
            print(return_text)
        if token == 'qa_en':
            buf = return_text
        else:
            buf = '403'
        self._set_response()
        buf = bytes(buf, encoding="utf-8")
        self.wfile.write(buf)


class HTTPServerV6(http.server.HTTPServer):
    address_family = socket.AF_INET6


def start_server(ip, port):
    if ':' in ip:
        http_server = HTTPServerV6((ip, port), TestHTTPHandle)
    else:
        http_server = http.server.HTTPServer((ip, int(port)), TestHTTPHandle)
    print('Server started. Press Ctrl+C to terminate api.')
    try:
        http_server.serve_forever()  # 设置一直监听并接收请求
    except KeyboardInterrupt:
        pass
    http_server.server_close()
    print('HTTP server closed')


if __name__ == '__main__':
    start_server('', 20000)  # For IPv4 Network Only
    # start_server('::', 20000) # For IPv6 Network
