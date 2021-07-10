# coding: utf-8
import os, os.path as op
import json
import csv
from tqdm import tqdm
from transformers.data.metrics.squad_metrics import normalize_answer

# config
DEV_PATH = './squad_v2/dev-v2.0.json'
DEV_DIR_PATH = './squad_v2/devset'
NBEST_PATH = './albert-xxlarge-v2/results-ckpt-44066/nbest_predictions.json'
CSV_PATH = './test_results/topK_em.csv'
LOG_PATH = 'output.log'
log_file = open(LOG_PATH, 'w+', encoding='utf-8')
# preds ,gts
squad_dev = json.load(open(DEV_PATH))
nbest = json.load(open(NBEST_PATH))
# em results
top_k = [1, 2, 3, 5, 10, 20]
max_k = 20
ground_truth = {}
exact_match = []


# log to file instead of printing in console
def log(s='', end='\n', printable=False):
    log_file.write(s + end)
    if printable:
        print(s, end=end)


# get ground_truth dict {'qas_id': [ans_texts]} from squad_v2 dataset
def transfer(dev):
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
    res = {}
    for examples in dev['data']:
        for paragraphs in examples['paragraphs']:
            for qas in paragraphs['qas']:
                texts = []
                for answer in qas['answers']:
                    texts.append(answer['text'])
                texts = list(set(texts))
                if len(texts) == 0 and qas['is_impossible']:
                    texts = ['']
                texts = [normalize_answer(text) for text in texts]
                res[qas['id']] = texts
    return res


# check if list1 & list2 has overlap
def has_overlap(list1, list2) -> bool:
    return len(set(list1).intersection(set(list2))) > 0


# write em to csv
def write_em_to_csv(dic):
    with open(CSV_PATH, 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['K'] + top_k)
        writer.writerow(['topK-EM'] + ['{:.3f}'.format(t) for t in dic.values()])


# compute top-N EM on squad_v2-dev
def compute_em(gt, preds) -> dict:
    em = dict([['top{}_em'.format(k), 0.0] for k in top_k])
    for id, gt_texts in tqdm(gt.items()):
        pred_text = [normalize_answer(p['text']) for p in preds[id][:max_k]]
        for k in top_k:
            pred_topk = list(set(pred_text[:k]))
            em['top{}_em'.format(k)] += int(has_overlap(pred_topk, gt_texts))
    for k in top_k:
        em['top{}_em'.format(k)] /= len(gt)
    return em


# analyze top-K error rate
def display_mistakes(gt, preds):
    total_err, no_ans_err = 0, 0
    for id, gt_texts in tqdm(gt.items()):
        pred = preds[id][:max_k]
        pred_text = [normalize_answer(p['text']) for p in pred]
        pred_topk = [int(has_overlap(list(set(pred_text[:k + 1])), gt_texts)) for k in range(max_k)]
        first_correct_k = len(pred_topk) - sum(pred_topk) + 1  # first correct pos, range from int 1 ~ max_k + 1
        if first_correct_k == 1:
            continue
        total_err += 1
        log('Error Case #{} '.format(total_err), end='')
        if '' in gt_texts + pred_text[:first_correct_k - 1]:
            no_ans_err += 1
            log('[No Ans Error]', end='')
        log()
        log('qas_id: {}'.format(id))
        sample = json.load(open(op.join(DEV_DIR_PATH, id + '.json')))
        log('context: {}'.format(sample['context']))
        log('question: {}'.format(sample['question']))
        log('gt: {}'.format(gt_texts))
        log('correct_k: {}'.format(first_correct_k) if first_correct_k <= max_k else 'No Correct Ans in top{}'.format(
            max_k))
        log('preds: ')
        for i, p in enumerate(pred[:first_correct_k]):
            log('{}) {}'.format(i + 1, p))
        log()
    # No_Ans situation make up 61% of top1~20 error, but no help because a concern about wrong-kill [误杀]
    log('No ans error rate: {}/{} = {}'.format(no_ans_err, total_err, no_ans_err / total_err), printable=True)


def main():
    # Compute EM
    global ground_truth, exact_match
    ground_truth = transfer(squad_dev)
    # exact_match = compute_em(ground_truth, nbest)
    # write_em_to_csv(exact_match)
    # print(exact_match)
    display_mistakes(ground_truth, nbest)


if __name__ == '__main__':
    main()
