# -*- coding: utf-8 -*-
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from tqdm import tqdm


# shuffle & gen trainval list
def shuffle_files(src_dir, txt_path):
    file_list = os.listdir(src_dir)
    shuffle(file_list)
    write_files_to_txt(txt_path, file_list)
    return file_list


# load txt to list
def load_files_from_txt(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(txt_path)
    return [f.replace('\n', '') for f in open(txt_path, encoding='utf-8').readlines() if len(f)]


# write list to txt
def write_files_to_txt(txt_path, file_list):
    with open(txt_path, 'w+') as f:
        for file in tqdm(file_list):
            f.write(file + '\n')
    if os.path.exists(txt_path):
        print('written {} filepaths to {}'.format(len(file_list), txt_path))
    else:
        print('failed to write filepaths to {}'.format(txt_path))


class SquadDataset(Dataset):
    """ A customized :class:`Dataset` class for AlbertModel squad_v2 outputs."""

    def __init__(self, data_folder, set_type, set_size):
        """ Init function for :class:`SquadDataset`
        :param data_folder: specify the dir_path contains data samples, each sample is a dict in format {x,start_pos,end_pos}
        :param set_type: 'train' or 'val'
        :param set_size: if set_type is 'train', load data in range [:set_size]; otherwise load data in range [-set_size:]
        """
        assert set_type in ['train', 'val'], "ValueError: Expected set_type in ['train', 'val'], but got {}".format(
            set_type)
        super(Dataset, self).__init__()
        self.data_folder = data_folder
        self.set_type = set_type
        self.trainval_txt = 'trainval.txt'

        # get file list
        # self.filename_format = r'train_data_(.*?)$'
        # all_filenames = sorted(os.listdir(self.data_folder), key=lambda x: int(re.findall(self.filename_format, x)[0]))
        self.all_filenames = load_files_from_txt(self.trainval_txt) if os.path.exists(self.trainval_txt) \
            else shuffle_files(self.data_folder, self.trainval_txt)
        self.filenames = self.all_filenames[:set_size] if self.set_type == 'train' else self.all_filenames[-set_size:]
        self.set_size = len(self.filenames)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filepath = os.path.join(self.data_folder, self.filenames[index])
        data = torch.load(filepath)
        return data['albert_output'].to(torch.float), \
               data['question_embedding'].to(torch.float), data['passage_attention'].to(torch.float), \
               data['start_position'].to(torch.long), data['end_position'].to(torch.long)
