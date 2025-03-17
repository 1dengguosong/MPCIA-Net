import torch
import pickle
import ipdb
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import seaborn as sns
from config import args


class REDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, samples):
        batch = {}
        for key in ['input_ids', 'token_type_ids', 'label',\
                    'attention_mask', 'has_trigger', 'trigger_len','utt_local_mask',
                    'utt_global_mask','sa_cross_mask','sa_self_mask']:
            batch[key] = [torch.tensor(sample[key], dtype=torch.long) for sample in samples]
            batch[key] = torch.stack(batch[key])
        t_s = []
        t_e = []
        x_s = []
        x_e = []
        y_s = []
        y_e = []

        for sample in samples:
            t_s.append(sample['t_start'])
            t_e.append(sample['t_end'])
            x_s.append(sample['x_st'])
            x_e.append(sample['x_nd'])
            y_s.append(sample['y_st'])
            y_e.append(sample['y_nd'])

        t_idx = [t_s, t_e]
        x_idx = [x_s, x_e]
        y_idx = [y_s, y_e]
        batch['t_idx'] = torch.tensor(t_idx, dtype=torch.long).T
        batch['x_idx'] = torch.tensor(x_idx, dtype=torch.long).T
        batch['y_idx'] = torch.tensor(y_idx, dtype=torch.long).T
        return batch


""" token_speaker_ids = sample['tokens_a_speaker_ids']  #用作说话者间和说话者内注意力掩饰
           token_mention_ids = sample['tokens_a_mention_ids'] #用作码间和话语内注意力掩饰
           att_mask = torch.ones(512, 512)
           utt_local_mask = torch.zeros_like(att_mask)
           utt_global_mask = torch.zeros_like(att_mask)
           sa_self_mask = torch.zeros_like(att_mask)
           sa_cross_mask = torch.zeros_like(att_mask)

           #语句内和语句间注意力掩码
           current_speaker = token_speaker_ids[0]
           length = len(token_speaker_ids)
           start_index = 0
           for i in range(1, length):
               if token_speaker_ids[i] != current_speaker:
                   # Set within utterance attention for the last segment
                   utt_local_mask[start_index:i, start_index:i] = 1
                   # Update the current speaker
                   current_speaker = token_speaker_ids[i]
                   start_index = i

           # Set for the last segment
           utt_local_mask[start_index:, start_index:] = 1

           # Set between utterance attention
           for i in range(length):
               for j in range(length):
                   if token_speaker_ids[i] != token_speaker_ids[j]:
                       utt_global_mask[i, j] = 1  #话语间


           length = len(token_mention_ids)

           for i in range(length):
               for j in range(length):
                   if token_mention_ids[i] != token_mention_ids[j]:
                       sa_cross_mask[i, j] = 1  # Allow attending between different speakers
                   if token_mention_ids[i] == token_mention_ids[j]:
                       sa_self_mask[i, j] = 1 #话语内
           # 添加到批量级别的列表
           utt_local_masks.append(utt_local_mask)
           utt_global_masks.append(utt_global_mask)
           sa_self_masks.append(sa_self_mask)
           sa_cross_masks.append(sa_cross_mask)"""