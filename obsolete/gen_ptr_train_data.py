# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AlbertConfig, AlbertTokenizer
from transformers.models.albert.modeling_albert import AlbertModel

# paras
model_name_or_path = "./albert-xxlarge-v2/finetuned-ckpt-44066"
data_dir = 'squad_v2'
max_seq_length = 384
max_question_len = 64
target_train_size = 10000000  # INF
train_batch_size = 8 * 4
# save_interval = 200
save_dir = os.path.join(data_dir, 'ptr_train')
save_path_tplt = os.path.join(save_dir, 'train_data_{}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setup model
config_class, tokenizer_class, model_class = (
    AlbertConfig,
    AlbertTokenizer,
    AlbertModel,
)
config = config_class.from_pretrained(model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
model = AlbertModel(config, add_pooling_layer=False)
model = model.from_pretrained(model_name_or_path, config=config, add_pooling_layer=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(model).to(device)


# convert torch.tensor to list
def to_cpu(tensor, dtype=torch.int):
    return tensor.to(dtype).detach().cpu()


# load examples, features, dataset
def load_features_and_dataset():
    cached_features_file = os.path.join(data_dir, "features_{}_{}".format("train", max_seq_length))
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file):
        print("Loading features from cached file", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        examples, features, dataset = (
            features_and_dataset["examples"],
            features_and_dataset["features"],
            features_and_dataset["dataset"],
        )
        return examples, features, dataset


# get train logits from AlbertModel sequence_outputs[0] as ptr-net's training input
def gen_ptr_train_data(model):
    global target_train_size
    train_examples, train_features, train_dataset = load_features_and_dataset()
    target_train_size = min(target_train_size, len(train_dataset))
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cur_id = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Generation")):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            example_indices = batch[5]
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            for i in range(len(example_indices)):
                # example.qas_id & feature.unique_id are all fixed here (0, 1e9)
                input_ids = to_cpu(batch[0][i])
                albert_output = to_cpu(outputs[0][i], torch.float16)
                start_position = to_cpu(batch[3][i])
                end_position = to_cpu(batch[4][i])

                # convert AlbertModel output => PointerNetwork input (question_emb, passage_att) ## require deep copy
                question_emb, passage_att = albert_output[:max_question_len + 2].clone(), albert_output.clone()
                # padding for each sample
                first_sep_pos = list(input_ids).index(3)  # locate first [SEP]
                question_emb[first_sep_pos + 1:] = 0      # [CLS], [Question], [SEP], ##[PAD]##
                passage_att[1:first_sep_pos] = 0          # [CLS], ##[PAD]##, [SEP], [Context], [SEP], ##[PAD]##

                cur_id += 1
                ptr_train_data = {
                    "albert_output": albert_output,
                    "question_embedding": question_emb,
                    "passage_attention": passage_att,
                    "start_position": start_position,
                    "end_position": end_position,
                }
                torch.save(ptr_train_data, save_path_tplt.format(cur_id))
    print("{} train files generated.".format(len(os.listdir(save_dir))))


def main():
    gen_ptr_train_data(model)
    pass


if __name__ == '__main__':
    main()
