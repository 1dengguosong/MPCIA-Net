import os

from model import REModel
import numpy as np
from dataset import REDataset
import argparse
import pickle
from config import args, device
import torch
import torch.nn as nn
import ipdb
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm.auto import tqdm, trange
from sklearn.metrics import f1_score
from test import test, load_model

def save_model(SAVED_MDL_PATH, epoch, model, optimizer, acc):
    if not os.path.exists(SAVED_MDL_PATH):
        with open(SAVED_MDL_PATH, 'w') as f:
            f.write('')  # 写入空内容，创建文件

    torch.save(obj={
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'acc': acc
    },
        f=SAVED_MDL_PATH,
        _use_new_zipfile_serialization=False
    )

def main():
    print(device)
    seed = 123
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import pickle

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

    dev_data = load_from_pkl(args.dev_path)
    dev_set = REDataset(dev_data)

    train_data = load_from_pkl(args.train_path)
    train_set = REDataset(train_data)

    model = REModel()


    #model = load_model(args.load_path)
    #model.reset(13)

    model.to(device)
    train_loader = DataLoader(
        train_set,
        args.batch_size,
        shuffle=True,
        collate_fn=train_set.collate_fn
    )

    dev_loader = DataLoader(
        dev_set,
        args.batch_size,
        shuffle=False,
        collate_fn=dev_set.collate_fn
    )


    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(),lr=args.lr)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_acc = float('-inf')
    for epoch in epoch_pbar:
        print(f"Epoch: {epoch}")
        pbar = tqdm(
            enumerate(train_loader),
            desc=f"Epoch: {epoch}",
            total=len(train_loader),
            ncols=0,
        )

        cum_loss = cum_binary_loss = cum_trigger_loss = cum_relation_loss = 0

        model.train()
        for local_step, batch in pbar:
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            optimizer.zero_grad()
            # [batch_size,37] - [batch_size,512,2] - [batch_size,2]
            relation_logit, start_end_logit, binary_logit = model(batch)
            relation_loss = loss_fn(relation_logit, batch['label'])

            trigger_loss = loss_fn(start_end_logit, batch['t_idx'])

            binary_loss = loss_fn(binary_logit, batch['has_trigger'])
            #binary_loss = 0

            total_loss = 1.0 * trigger_loss + 1.0 * relation_loss + 1.0 * binary_loss
            total_loss.backward()
            optimizer.step()

            cum_trigger_loss += float(trigger_loss)
            cum_relation_loss += float(relation_loss)
            cum_binary_loss += float(binary_loss)
            cum_loss += float(total_loss)
            run_loss = float(cum_loss) / (local_step + 1)
            pbar.set_postfix_str(f"Loss: {run_loss:.6f} | "
                                 f"trigger_loss: {cum_trigger_loss / (local_step+1):.10f} | "
                                 f"relation_loss: {cum_relation_loss / (local_step+1):.10f} | "
                                 f"binary_loss: {cum_binary_loss / (local_step+1):.10f}")

        print('=======val=======')
        test_f1 = test(dev_loader,model, epoch)
        print("test_f1: ", test_f1," best_acc: ", best_acc)
        if test_f1 > best_acc:
            best_acc = test_f1
            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
            ckpt_path = args.ckpt_dir / f"{test_f1}_epoch_{epoch}.pt"
            save_model(ckpt_path, epoch, model, optimizer, test_f1)
            print(f"Saving model at {str(ckpt_path)} with acc: {test_f1}")
        print('=======end========\n')

if __name__ == '__main__':
    main()
