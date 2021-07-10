# -*- coding: utf-8 -*-
""" 问答系统API的HTTP客户端程序 """
import requests
import time
import os.path as op
import torch

# params
url = 'http://60.19.58.86:20000/'
token = 'qa_en'
data_dir = 'squad_v2'
features_file = op.join(data_dir, 'features_dev_384')


# get examples, features
# features_and_dataset = torch.load(features_file)
# all_examples, all_features = (
#     features_and_dataset["examples"],
#     features_and_dataset["features"],
# )
# print('loaded {} examples and {} features'.format(len(all_examples), len(all_features)))


# online real-time recognition
def online_predict(context, question):
    datas = {'token': token, 'context': context, 'question': question}
    r = requests.post(url, datas)
    r.encoding = 'utf-8'
    answer_text, score = r.text.split('&')
    return answer_text, float(score)


# test online api
def test():
    # QA Demo from raw text
    context = "New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean. " \
              "It has a total land area of 268,000 square kilometres (103,500 sq mi), and a population of 4.9 million. " \
              "New Zealand's capital city is Wellington, and its most populous city is Auckland."
    questions = ["How many people live in New Zealand?",
                 "What's the largest city?",
                 "Where's New Zealand?",
                 "How large is New Zealand?",
                 "What's the name of the ocean island?",
                 "Which city has the largest area in New Zealand?",
                 ]
    # QA Demo from dataset
    # start_idx, end_idx = 3347, 3355
    # questions = [all_examples[idx].question_text for idx in range(start_idx, end_idx + 1)]
    # # golds = [all_examples[idx].answer_text for idx in range(start_idx, end_idx + 1)]
    # context = all_examples[start_idx].context_text
    for i, question in enumerate(questions):
        # input('press any key to run a sample...')
        start_t = time.time()
        answer_text, score = online_predict(context, question)
        result = {
            'answer_text': answer_text,
            # 'gold_text': golds[i],
            'score': '{:.4f}'.format(score),
            'total cost': '{:.0f}ms'.format((time.time() - start_t) * 1e3),
        }
        print(result)

    # Results:
    # 通信开销大约为1000ms, 线上同样本运行时长约为本地1/3,
    # 对小样本来说不划算(本地650ms=>线上1250ms), 对大样本来说较划算(本地3200ms=>线上2200ms)


if __name__ == '__main__':
    test()
