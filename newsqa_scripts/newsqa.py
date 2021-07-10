# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from tqdm import tqdm
from transformers.data.processors.squad import SquadExample, SquadProcessor


class NewsqaV1Processor(SquadProcessor):
    train_file = "train-v1.0.json"
    dev_file = "dev-v1.0.json"

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for paragraph in tqdm(input_data):
            title = paragraph["storyId"]
            paragraph_id = re.findall(r'./cnn/stories/(.*?).story', title)[0]
            context_text = paragraph["text"]
            for qa_index, qa in enumerate(paragraph["questions"]):
                qas_id = "{}_{}".format(paragraph_id, qa_index + 1)
                question_text = qa["q"]
                start_position_character = None
                answer_text = None
                answers = []

                is_bad_question = qa["consensus"].get("badQuestion", False)
                if is_bad_question:
                    continue
                is_impossible = qa["consensus"].get("noAnswer", False)
                if not is_impossible:
                    # get train only answer from field "consensus"
                    if is_training:
                        answer = qa["consensus"]
                        start_position_character = answer["s"]
                        answer_text = context_text[answer["s"]:answer["e"] - 1]
                    # get dev golden answers from field "validatedAnswers" & "consensus"
                    elif "validatedAnswers" in qa:
                        for answer in qa["validatedAnswers"]:
                            if "s" not in answer:
                                continue
                            answers.append({
                                "answer_start": answer["s"],
                                "text": context_text[answer["s"]:answer["e"] - 1],
                            })
                    else:
                        answer = qa["consensus"]
                        answers.append({
                            "answer_start": answer["s"],
                            "text": context_text[answer["s"]:answer["e"] - 1],
                        })

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    title=title,
                    is_impossible=is_impossible,
                    answers=answers,
                )
                examples.append(example)

        return examples


# if __name__ == "__main__":
#     data_dir = '../newsqa_v1'
#     train_file = 'train-v1.0.json'
#     predict_file = 'dev-v1.0.json'
#
#     processor = NewsqaV1Processor()
#     train_examples = processor.get_train_examples(data_dir, filename=train_file)
#     print("loaded {} train examples from {}".format(len(train_examples), train_file))
#     eval_examples = processor.get_dev_examples(data_dir, filename=predict_file)
#     print("loaded {} eval examples from {}".format(len(eval_examples), predict_file))
