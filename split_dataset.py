# -*- coding: utf-8 -*-
import json
from tqdm import tqdm
import os, os.path as op
from transformers.data.metrics.squad_metrics import normalize_answer

data_dir = 'squad_v2'
trainset_file = op.join(data_dir, 'train-v2.0.json')
train_dir = op.join(data_dir, 'trainset')
devset_file = op.join(data_dir, 'dev-v2.0.json')
dev_dir = op.join(data_dir, 'devset')


# split dataset to dicts {'context': context, 'question': question, 'answer_text': [ans_texts]} from squad_v2 dataset
def split_dataset(dataset_file, dest_dir):
    """
    Dev dataset architecture:
    {
        "data": [
            {
                "title": "Super_Bowl_50",
                "paragraphs": [
                    {
                        "context": " numerals 50.",
                        "qas": [
                            {
                                "answers": [
                                    {
                                        "answer_start": 177,
                                        "text": "Denver Broncos",
                                    },
                                ],
                                "question": "Which NFL team represented the AFC at Super Bowl 50?",
                                "id": "56be4db0acb8001400a502ec"
                                "is_impossible": False
                            }
                        ]
                    }
                ]
            }
        ],
        "version": "2.0"
    }
    """
    data = json.load(open(dataset_file))
    if not op.exists(dest_dir):
        os.makedirs(dest_dir)
    for examples in tqdm(data['data'], desc="split {} to {}".format(dataset_file, dest_dir)):
        title = examples['title']
        for paragraphs in examples['paragraphs']:
            context = paragraphs['context']
            for qa in paragraphs['qas']:
                qas_id = qa['id']
                question = qa['question']
                answer_texts = [answer['text'] for answer in qa['answers']]
                start_characters = [answer['answer_start'] for answer in qa['answers']]
                is_impossible = qa.get('is_impossible', False)
                if len(answer_texts) == 0 and is_impossible:
                    answer_texts = ['']
                    start_characters = None
                sample_data = {
                    'title': title,
                    'qas_id': qas_id,
                    'context': context,
                    'question': question,
                    'is_impossible': is_impossible,
                    'answer_text': answer_texts,
                    'start_character': start_characters,
                }
                dest_file = op.join(dest_dir, qas_id + '.json')
                json.dump(sample_data, open(dest_file, 'w+'), indent=4)
    print('{} json files generated.'.format(len(os.listdir(dest_dir))))


if __name__ == '__main__':
    # split dev set to 11873 json files, each contains only one pair of Q & A
    split_dataset(trainset_file, train_dir)
    split_dataset(devset_file, dev_dir)
