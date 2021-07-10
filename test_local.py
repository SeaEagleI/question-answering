# coding: utf-8
import os.path as op
import time
import torch
import transformers
from transformers import AlbertTokenizer, AlbertConfig, AlbertForQuestionAnswering
from lib import run_prediction

# config
data_dir = 'squad_v2'
features_file = op.join(data_dir, 'features_dev_384')
# logits_file = op.join(data_dir, 'logits_dev_384')
# nbest_file = './albert-xxlarge-v2/results-ckpt-44066/nbest_predictions.json'
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
# models = [model_class.from_pretrained(model_path, config=config) for model_path in model_paths]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# models = [model.to(device) for model in models]
# count model parameters
model = model_class.from_pretrained(model_paths[0], config=config).to(device)
for param in model.named_parameters():
    # print(param[0], param[1].size())
    print(param[1].numel())
# num_params = sum(p.numel() for p in model.parameters())
# print("Total params: %.2fM" % (num_params / 1e6))


# get examples, features, logits
# features_and_dataset = torch.load(features_file)
# all_examples, all_features = (
#     features_and_dataset["examples"],
#     features_and_dataset["features"],
# )
# all_results = torch.load(logits_file)
# print('loaded {} examples, {} features and {} results'.format(len(all_examples), len(all_features), len(all_results)))


# test locally
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
    # for i, question in enumerate(questions):
    #     # input('press any key to run a sample...')
    #     start_t = time.time()
    #     answer_text, score = run_prediction(question, context, models, tokenizer)
    #     result = {
    #         'answer_text': answer_text,
    #         # 'gold_text': golds[i],
    #         'score': '{:.4f}'.format(score),
    #         'inference time': '{:.0f}ms'.format((time.time() - start_t) * 1e3),
    #     }
    #     print(result)

    # *******************************Test Results*********************************
    #                           | GTX 1060M 6G (local) |  RTX 2080Ti 12G (remote)
    # Avg_Infer_Time_Single_Doc |       750ms          |         250ms
    # Avg_Infer_Time_Multi_Doc  |   3000ms~3500ms      |        1000ms


if __name__ == '__main__':
    test()
