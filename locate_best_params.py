# coding: utf-8
import os, os.path as op
import json
import torch
import numpy as np
from transformers import AlbertTokenizer, AlbertConfig, AlbertForQuestionAnswering
from lib import get_all_combinations, ensemble_logits, eval_benchmark, run_prediction

# config
data_dir = 'squad_v2'
model_type = "albert"
num_folds = 4
model_name_or_path = "./albert-xxlarge-v2/pretrained_model"
# nbest_file = './albert-xxlarge-v2/results-ckpt-44066/nbest_predictions.json'
features_file = op.join(data_dir, 'features_AlbertTokenizer_dev_384')
# logits_file = op.join(data_dir, 'logits_dev_384_albert-xxlarge-v2')
n_best_size = 10
max_answer_length = 30
null_score_diff_threshold = 0.0

# setup model
MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
}
config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
# config = config_class.from_pretrained(model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
# model = model_class.from_pretrained(model_name_or_path, config=config)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# get examples, features, logits
features_and_dataset = torch.load(features_file)
examples, features = (
    features_and_dataset["examples"],
    features_and_dataset["features"],
)
print('loaded {} examples, {} features'.format(len(examples), len(features)))
# results = torch.load(logits_file)
# print('loaded {} results'.format(len(results)))

# EM score on devset for each model:
# "1": 80.39248715573149,
# "2": 86.08607765518403,
# "3": 87.06308430893624,
# "4": 86.34717426092816,
logits_filelist = [op.join(data_dir, 'logits_AlbertTokenizer_dev_384_{}folds_{}'.format(num_folds, fold_id))
                   for fold_id in range(1, num_folds + 1)]


def main():
    # QA Demo from raw text
    # context = "New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean. " \
    #           "It has a total land area of 268,000 square kilometres (103,500 sq mi), and a population of 4.9 million. " \
    #           "New Zealand's capital city is Wellington, and its most populous city is Auckland."
    # questions = ["How many people live in New Zealand?",
    #              "What's the largest city?"]
    # for question in questions:
    #     prediction = run_prediction(question, context, model, tokenizer)
    #     print(prediction)
    # QA Demo from dataset
    # start_idx, end_idx = 3347, 3355
    # questions = [examples[idx].question_text for idx in range(start_idx, end_idx + 1)]
    # contexts = [examples[idx].context_text for idx in range(start_idx, end_idx + 1)]
    # for question, context in zip(questions, contexts):
    #     prediction = run_prediction(question, context, model, tokenizer)
    #     print(prediction)

    # Benchmark Test to locate best ensemble model combinations => (2,3,4) == 87.7116
    # all_combinations = get_all_combinations(range(1, num_folds + 1))
    # print("Testing {} ensemble combinations:\n{}".format(len(all_combinations), all_combinations))
    # for index_list in all_combinations:
    #     logits_files = [logits_filelist[file_id - 1] for file_id in index_list]
    #     results = ensemble_logits(logits_files)
    #     benchmark, time_cost = eval_benchmark(examples, features, results, n_best_size, null_score_diff_threshold, tokenizer)
    #     print(index_list, "{:.4f}".format(benchmark['exact']), time_cost)
    #     print(json.dumps(dict(benchmark), indent=4))

    # Benchmark Test to locate best null_score_diff_thresh
    thresh_list = np.arange(-6, 3, 0.5)
    # thresh_list = [0]
    print("Testing {} values for param null_score_diff_threshold:\n{}".format(len(thresh_list), thresh_list))
    index_list = (2, 3, 4)  # best solution: fold2, fold3, fold4
    logits_files = [logits_filelist[file_id - 1] for file_id in index_list]
    results = ensemble_logits(logits_files)
    for t in thresh_list:
        benchmark, time_cost = eval_benchmark(examples, features, results, n_best_size, t, tokenizer)
        print(t, "{:.4f}".format(benchmark['exact']), time_cost)
    #     print(json.dumps(dict(benchmark), indent=4))


if __name__ == '__main__':
    main()
