from model import REModel
import numpy as np
from pathlib import Path
from dataset import REDataset
import argparse
import pickle
from config import args, device
import json
import sys
import torch
import torch.nn as nn
import ipdb
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.optim as optim
from tqdm.auto import tqdm, trange
from sklearn.metrics import f1_score
from cal import cal_seen_unseen_stats

def load_model(path: Path) -> REModel:
    model = REModel()
    #model.reset(13)
    model.to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    del ckpt
    return model

def test(test_loader,model=None, epoch=2):
    
    if model is None:
        model = load_model(args.load_path)
    #model.reset(13)
    model.to(device)

    y_true, y_pred = [], []
    model.eval()

    preds, gts = [], []
    bin_pred, bin_gt = [], []

    has_trig_preds, has_trig_gt = [], []
    no_trig_preds, no_trig_gt = [], []


    trigger_length = {}
    start_pred, end_pred = [], []
    start_gt, end_gt = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            y_true.extend([label.item() for label in batch['label']])
            bin_gt.extend([label.item() for label in batch['has_trigger']])
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            pred, has_trigger, ids, p_trigs, gt_trigs  = model.infer(batch)

            for i in range(len(batch['label'])):
                cur_trig_len = (ids[i][1] - ids[i][0])
                if not type(cur_trig_len) == int: cur_trig_len = cur_trig_len.item()
                trigger_length[cur_trig_len] = trigger_length.get(cur_trig_len, 0) + 1
                start_pred.append(ids[i][0])
                end_pred.append(ids[i][1])
                start_gt.append(batch['t_idx'][i][0])
                end_gt.append(batch['t_idx'][i][1])


            preds.extend(p_trigs)
            gts.extend(gt_trigs)

            y_pred.extend([label.item() for label in pred])
            bin_pred.extend([label.item() for label in has_trigger])
            for i in range(len(batch["label"])):
                if batch["has_trigger"][i] == 1:
                    has_trig_gt.append(batch['label'][i].item())
                    has_trig_preds.append(pred[i].item())
                else:
                    no_trig_gt.append(batch['label'][i].item())
                    no_trig_preds.append(pred[i].item())

        cal_metric(preds, gts)
        #宏观 F1 分数是通过将所有类别的 F1 分数计算后取平均
        print("f1_macro ",f1_score(y_true, y_pred, average='macro'))
        all_f1 = f1_score(y_true, y_pred, average='micro')
        #微观 F1 分数是通过将所有类别的 的 TP（真正例）、FP（假正例）和 FN（假负例）合并后计算 F1 分数
        print(f"f1_micro  {all_f1}")
        no_trigger_f1 = f1_score(no_trig_gt, no_trig_preds, average='micro')
        print(f"no trig f1: {no_trigger_f1}")
        has_trigger_f1 = f1_score(has_trig_gt, has_trig_preds, average='micro')
        print(f"has trig f1: {has_trigger_f1}")
        bin_acc = f1_score(bin_gt, bin_pred, average='micro')
        #评估模型在二元分类（有触发器与没有触发器）任务上的表现
        print(f"bin_acc_f1: {bin_acc}")

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        print("acc ",(y_pred==y_true).sum() / len(np.array(y_true)))

        #差异性数据
        with open(f"save/{epoch}_diff.csv", 'w') as f:
            diff = list(zip(gts, preds))
            diff = [','.join([pair[0], pair[1]]) for pair in diff]
            print("\n".join(diff), file=f)
        #预测结果
        with open(f"save/{epoch}_preds.txt", 'w') as f1:
            print("\n".join(preds), file=f1)
        #真实结果
        with open(f"save/{epoch}_gts.txt", 'w') as f2:
            print("\n".join(gts), file=f2)
        np.savetxt(f'save/{epoch}_y_pred.txt', np.array(y_pred), fmt='%d')
        np.savetxt(f'save/{epoch}_y_true.txt', np.array(y_true), fmt='%d')

        print("(un)seen stats:")
        cal_seen_unseen_stats(y_pred, y_true)
        return all_f1

def cal_metric(preds, gts):
    assert len(preds) == len(gts)
    false_pos = 0
    false_neg = 0
    exact_match = 0
    cls_cnt = 0
    f = open('diff.csv', 'w')
    res = []
    for i in range(len(preds)):
        res.append(preds[i] + "," + gts[i])
        if gts[i] == '[CLS]':
            cls_cnt += 1
        if gts[i] == '[CLS]' and preds[i] != '[CLS]':
            false_pos += 1
        if gts[i] != '[CLS]' and preds[i] == '[CLS]':
            false_neg += 1
        if gts[i] != '[CLS]' and preds[i] == gts[i]:
            exact_match += 1
    print(f"false_pos (假正例):: {false_pos}")
    print(f"false_neg (假负例): {false_neg}")
    print(f"exact_match (完全匹配): {exact_match}")
    print(f"cls_cnt (类别计数): {cls_cnt}")
    print(f"total amount (总样本数量):: {len(preds)}")
    print("\n".join(res), file=f)


if __name__ == '__main__':
    def load_from_pkl(pkl_file_path):
        data_list = []
        with open(pkl_file_path, 'rb') as f:
            while True:
                try:
                    # Load each object one by one
                    data_list.append(pickle.load(f))
                except EOFError:
                    break  # Stop when we reach the end of the file
        return data_list

    test_data = load_from_pkl(args.test_path)

    test_set = REDataset(test_data)
    test_loader = DataLoader(
        test_set,
        args.batch_size,
        shuffle=False,
        collate_fn=test_set.collate_fn
    )
    test(test_loader)
