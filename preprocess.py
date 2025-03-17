from transformers import BertTokenizer
import torch
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import ipdb
import spacy
import ipdb
from contraction import contractions_dict
from config import args

import re
contractions_re = re.compile('(%s)'%'|'.join(contractions_dict.keys()))


NUM_SPEAKER_DICT = {'DialogRE':10, 'DailyDialog':2, 'EmoryNLP':8, 'MELD':8, 'MRDA':26 }


def tokenize1(text, tokenizer, start_mention_id, head, tail, dataset):
    SPEAKER_IDS = ["speaker {}".format(x) for x in range(1, NUM_SPEAKER_DICT[dataset] + 1)]  # max_num_speaker
    speaker_dict = dict(zip(SPEAKER_IDS, range(1, NUM_SPEAKER_DICT[dataset] + 1)))
    speaker_dict['[s1]'] = NUM_SPEAKER_DICT[dataset] + 1
    speaker_dict['[s2]'] = NUM_SPEAKER_DICT[dataset] + 2

    speaker2id = speaker_dict
    D = ['[s1]', '[s2]'] + SPEAKER_IDS
    text_tokens = []
    # textraw = [text]
    textraw = text.split('\n')

    ntextraw = []
    for i, turn in enumerate(textraw):
        first_colon = turn.find(':')
        speakers = turn[:first_colon]
        dialog = turn[first_colon:]
        speakers = [speakers]
        for delimiter in D:
            tmp_text = []
            for k in range(len(speakers)):
                tt = speakers[k].split(delimiter)
                for j, t in enumerate(tt):
                    tmp_text.append(t)
                    if j != len(tt) - 1:
                        tmp_text.append(delimiter)
            speakers = tmp_text
        ntextraw.extend(speakers)
        ntextraw.append(dialog)
    textraw = ntextraw

    text = []
    # speaker_ids, mention_ids may relate to speaker embedding in Fig 2
    # mention_id is the order of apperance for a specific speaker_id
    # both speaker_id and mention_id are assigned to each token in current turn
    speaker_ids = []  # same length as tokens
    mention_ids = []  # same length as tokens
    mention_id = start_mention_id
    speaker_id = 0  # number of mentioned speakers?
    mentioned_h = set()
    mentioned_t = set()

    for t in textraw:

        if t in SPEAKER_IDS:
            speaker_id = speaker2id[t]
            mention_id += 1

            # add [CLS] for each dialog
            text += ['[turn]']
            speaker_ids.append(speaker_id)
            mention_ids.append(mention_id)

            tokens = tokenizer.tokenize(t + " ")
            for tok in tokens:
                text += [tok]
                speaker_ids.append(speaker_id)
                mention_ids.append(mention_id)

        elif t in ['[s1]', '[s2]']:
            speaker_id = speaker2id[t]
            mention_id += 1

            # add [CLS] for each dialog
            text += ['[turn]']
            speaker_ids.append(speaker_id)
            mention_ids.append(mention_id)

            text += [t]
            speaker_ids.append(speaker_id)
            mention_ids.append(mention_id)
        else:
            tokens = tokenizer.tokenize(t)

            for tok in tokens:
                text += [tok]
                speaker_ids.append(speaker_id)
                mention_ids.append(mention_id)

        # establish an edge between an argument and a turn that mentioned it
        if head in t:
            mentioned_h.add(mention_id)
        if tail in t:
            mentioned_t.add(mention_id)

    return text, speaker_ids, mention_ids, mentioned_h, mentioned_t

def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)


MAX_LEN = 512
nlp = spacy.load("en_core_web_sm")

def lemmatize(text):
    text = nlp(text)
    return " ".join([(t.text) for t in text if t.pos_ not in ['PUNCT', 'SPACE'] or t.lemma_ == ':']).replace(" '", "'").replace(" n't", "n't")

def preprocess(PATH: str, mode: str) -> None:
    def is_speaker(a):
        a = a.split()
        return (len(a) == 2 and a[0] == "speaker" and a[1].isdigit())
    def rename(d: str, x:str, y: str) -> (str, str, str):
        unused = ["[s1]", "[s2]"]
        a = []
        if is_speaker(x):
            a += [x]
        else:
            a += [None]
        if x != y and is_speaker(y):
            a += [y]
        else:
            a += [None]
        for i in range(len(a)):
            if a[i] is None:
                continue
            d = d.replace(a[i] + ":", unused[i] + " :")
            if x == a[i]:
                x = unused[i]
            if y == a[i]:
                y = unused[i]
        return d, x, y

    tokenizer = BertTokenizer.from_pretrained(args.model)
    tokenizer.add_tokens(["[s1]", "[s2]"])
    data = json.load(open(PATH, 'r',encoding='utf-8'))
    duplicate = 0
    map = {}
    examples = []
    map1 = {'under500':0, 'over500': 0}
    trigger_location = {500: 0, 1000: 0, 1500: 0}
    qq = 0

    with open(PATH[:-5] + '.pkl', 'ab') as fp:
        for i in tqdm(range(len(data))):
            text_a = "\n".join(data[i][0])
            text_a = expand_contractions(text_a)

            for j in range(len(data[i][1])):
                text_b = data[i][1][j]['x']
                text_b = expand_contractions(text_b)
                text_c = data[i][1][j]['y']
                text_c = expand_contractions(text_c)
                map[len(data[i][1][j]['rid'])] = \
                    map.get(len(data[i][1][j]['rid']), 0) + 1
                for l in range(len(data[i][1][j]['rid'])):
                    if l > 0: duplicate += 1
                    if mode == 'train' and l > 0:
                        break
                    label = data[i][1][j]['rid'][l] - 1
                    d, x, y = rename(text_a.lower(), text_b.lower(), text_c.lower())

                    tokens_a, tokens_a_speaker_ids, tokens_a_mention_ids, mentioned_x, mentioned_y = tokenize1(
                        d, tokenizer, 0, x, y, 'DialogRE')

                    turn_indexes = [index for index, token in enumerate(tokens_a) if token == '[turn]']
                    ii = 0
                    for turn_index in turn_indexes:
                        del tokens_a_speaker_ids[turn_index - ii]
                        del tokens_a_mention_ids[turn_index - ii]
                        del tokens_a[turn_index - ii]
                        ii = ii + 1
                    d = tokenizer.convert_tokens_to_string(tokens_a)

                    if len(tokens_a) < MAX_LEN - 2 - (len(tokenizer(x).input_ids) + len(tokenizer(y).input_ids) - 2):
                        # 这里 tokens_a_speaker_ids和tokens_a_mention_ids 填充后面到512原有的在后面元素0
                        tokens_a_speaker_ids.extend([0] * (MAX_LEN - len(tokens_a_speaker_ids)))
                        tokens_a_mention_ids.extend([0] * (MAX_LEN - len(tokens_a_mention_ids)))
                    else:
                        # 这里 tokens_a_speaker_ids和tokens_a_mention_ids 保留0 -- MAX_LEN - 2 - (len(tokenizer(x).input_ids) + len(tokenizer(y).input_ids) - 2):的的元素
                        available_len = MAX_LEN - 2 - (len(tokenizer(x).input_ids) + len(tokenizer(y).input_ids) - 2)
                        tokens_a_speaker_ids = tokens_a_speaker_ids[:available_len]  # 截取有效长度
                        tokens_a_mention_ids = tokens_a_mention_ids[:available_len]
                        tokens_a_speaker_ids.extend([0] * (MAX_LEN - len(tokens_a_speaker_ids)))
                        tokens_a_mention_ids.extend([0] * (MAX_LEN - len(tokens_a_mention_ids)))

                    t_start = 0
                    t_end = 1
                    trigger = data[i][1][j]['t'][l].lower()
                    trigger = expand_contractions(trigger)
                    if len(trigger) > 0:
                        t_start = d.find(trigger)
                        t_start = len(tokenizer(d[:t_start]).input_ids) - 1
                        trigger_ids = tokenizer(trigger).input_ids[1:-1]
                        d_ids = tokenizer(d).input_ids
                        for k in range(len(d_ids)):
                            if d_ids[k: k + len(trigger_ids)] == trigger_ids:
                                t_start = k
                                break
                        try:
                            t_end = t_start + len(tokenizer(trigger).input_ids) - 2
                        except:
                            ipdb.set_trace()

                    text = d + '[SEP]' + x + '[CLS]' + y
                    attn_mask = torch.ones(512)

                    tmp_ids = tokenizer(text).input_ids[1:-1]

                    if len(tmp_ids) <= MAX_LEN:
                        map1['under500'] += 1
                    else:
                        map1['over500'] += 1
                        if t_end <= 500:
                            trigger_location[500] += 1
                        elif t_end <= 1000:
                            trigger_location[1000] += 1
                        else:
                            trigger_location[1500] += 1

                    x_st = len(tokenizer(d).input_ids)
                    if len(tmp_ids) < MAX_LEN - 2:
                        text += '[PAD] ' * (MAX_LEN - 2 - len(tmp_ids))
                        attn_mask[-1 - (MAX_LEN - 2 - len(tmp_ids)): -1] = 0
                    elif len(tmp_ids) > MAX_LEN - 2:
                        last_available = MAX_LEN - 2 - (len(tokenizer(x).input_ids) + len(tokenizer(y).input_ids) - 2)
                        if t_start and t_end:
                            if t_start >= last_available + 1 or t_end >= last_available:
                                t_start = 0
                                t_end = 1
                        text = tokenizer.decode(tmp_ids[:last_available]) + '[SEP]' + x + '[CLS]' + y
                        x_st = len(tmp_ids[:last_available]) + 2

                    token_type_ids = torch.zeros(512)
                    second_seq_len = len(tokenizer(x).input_ids) + len(tokenizer(y).input_ids) - 2
                    token_type_ids[-second_seq_len:] = 1

                    x_nd = x_st + len(tokenizer(x).input_ids) - 2
                    y_st = x_nd + 1
                    y_nd = y_st + len(tokenizer(y).input_ids) - 2

                    ############# Assertion #############
                    tmp_ids = tokenizer(text).input_ids
                    if t_start != 0 and t_end != 1:
                        try:
                            if not tokenizer.decode(tmp_ids[t_start:t_end]).replace(' ', '') == tokenizer.decode(
                                    tokenizer(trigger).input_ids[1:-1]).replace(' ', '') and \
                                    not tokenizer.decode(tokenizer(trigger).input_ids[1:-1]).replace(' ',
                                                                                                     '') in tokenizer.decode(
                                        tmp_ids[t_start:t_end]).replace(' ', ''):
                                raise NameError('Error from TRIGGER!')
                        except:
                            ipdb.set_trace()
                    try:
                        assert tokenizer.decode(tmp_ids[x_st:x_nd]).replace(' ', '') == tokenizer.decode(
                            tokenizer(x).input_ids[1:-1]).replace(' ', '')
                    except:
                        ipdb.set_trace()
                    try:
                        assert tokenizer.decode(tmp_ids[y_st:y_nd]).replace(' ', '') == tokenizer.decode(
                            tokenizer(y).input_ids[1:-1]).replace(' ', '')
                    except:
                        ipdb.set_trace()

                    try:
                        assert len(tmp_ids) == MAX_LEN
                    except:
                        ipdb.set_trace()
                    #####################################

                    t_len = t_end - t_start if len(trigger) > 0 else 0
                    distributions = [-100] * 15
                    if len(trigger) > 0:
                        distributions[:t_end - t_start] = tmp_ids[t_start:t_end]
                    att_mask = torch.ones(512, 512)
                    utt_local_mask = torch.zeros_like(att_mask)
                    utt_global_mask = torch.zeros_like(att_mask)
                    sa_self_mask = torch.zeros_like(att_mask)
                    sa_cross_mask = torch.zeros_like(att_mask)

                    # 语句内和语句间注意力掩码
                    length = len(tokens_a_speaker_ids)
                    """ start_index = 0
                    for t in range(1, length):
                        if tokens_a_speaker_ids[t] != current_speaker:
                            # Set within utterance attention for the last segment
                            utt_local_mask[start_index:t, start_index:t] = 1
                            # Update the current speaker
                            current_speaker = tokens_a_speaker_ids[t]
                            start_index = t

                    # Set for the last segment
                    utt_local_mask[start_index:, start_index:] = 1"""

                    # Set between utterance attention
                    for t in range(length):
                        for m in range(length):
                            if tokens_a_speaker_ids[t] != tokens_a_speaker_ids[m]:
                                utt_global_mask[t, m] = 1  # 话语间
                            else:
                                utt_local_mask[t,m] = 1


                    length = len(tokens_a_mention_ids)

                    for m in range(length):
                        for n in range(length):
                            if tokens_a_mention_ids[m] != tokens_a_mention_ids[n]:
                                sa_cross_mask[m, n] = 1  # Allow attending between different speakers
                            if tokens_a_mention_ids[m] == tokens_a_mention_ids[n]:
                                sa_self_mask[m, n] = 1  # 话语内

                    example = {
                        'has_trigger': 1 if len(trigger) > 0 else 0,
                        'trigger_len': t_len,
                        'd': d,
                        'x': x,
                        'y': y,
                        'text': text,
                        'input_ids': tmp_ids,
                        'attention_mask': attn_mask,
                        'token_type_ids': token_type_ids,
                        'label': label,
                        't_start': t_start,
                        't_end': t_end,
                        'trigger': trigger,
                        'x_st': x_st,
                        'x_nd': x_nd,
                        'y_st': y_st,
                        'y_nd': y_nd,
                        'distributions': distributions,
                        'tokens_a_speaker_ids': tokens_a_speaker_ids,
                        'tokens_a_mention_ids': tokens_a_mention_ids,
                        'mentioned_x': mentioned_x,
                        'mentioned_y': mentioned_y,
                        'utt_local_mask': utt_local_mask,
                        'utt_global_mask': utt_global_mask,
                        'sa_cross_mask': sa_cross_mask,
                        'sa_self_mask': sa_self_mask
                    }
                    pickle.dump(example, fp)
                    qq = qq +1

    print('cached!')
    print(f"duplicate: {duplicate}")
    print(f"stats: {map}")
    print(map1)
    print(f"trigger_location: {trigger_location}")
    print(f"QQ= {qq}")

if __name__ == '__main__':
    #preprocess('data_dre/dev1.json', 'dev')
    #preprocess('data_dre/dev.json', 'dev')
    preprocess('data_dre/train.json', 'train')
    preprocess('data_dre/test.json', 'test')