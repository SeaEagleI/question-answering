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

import json
import os
from tqdm import tqdm

from transformers.utils import logging
from transformers.data.processors.squad import SquadExample, SquadFeatures, SquadResult, SquadProcessor

logger = logging.get_logger(__name__)


class SquadV2Processor(SquadProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"
    trainset_dir = "trainset"

    def get_train_examples(self, data_dir, datalist_file=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            datalist_file: A text file specify the train datalist in N-folds training.
        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(datalist_file, "r", encoding="utf-8") as reader:
            input_datalist = reader.readlines()
            examples = []
            for line in tqdm(input_datalist):
                json_file = os.path.join(data_dir, self.trainset_dir, line.replace('\n', ''))
                sample_data = json.load(open(json_file))

                is_impossible = sample_data["is_impossible"]
                answer_text = None
                start_position_character = None
                if not is_impossible:
                    answer_text = sample_data["answer_text"][0]
                    start_position_character = sample_data["start_character"][0]

                example = SquadExample(
                    qas_id=sample_data["qas_id"],
                    question_text=sample_data["question"],
                    context_text=sample_data["context"],
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    title=sample_data["title"],
                    is_impossible=is_impossible,
                    answers=[],
                )
                examples.append(example)
            return examples
