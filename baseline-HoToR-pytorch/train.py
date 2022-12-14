#!/usr/bin/env python
# coding=utf-8
import math
from datetime import datetime
import socket
import os
import random
import pickle
import argparse
import time
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from multiprocessing import cpu_count
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.utils.tensorboard import SummaryWriter

from model import HoToR


class TripletUniformPair(IterableDataset):
    def __init__(self, num_item, user_list, click_pair, buy_pair, shuffle, num_epochs, lambda_mod):
        self.num_item = num_item
        self.user_list = user_list
        self.click_pair = click_pair
        self.buy_pair = buy_pair
        self.total_pair = click_pair[:] + buy_pair[:]
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.lambda_mod = lambda_mod

    def __iter__(self):
        return self.get_batch()

    def _example(self, idx):
        u = self.total_pair[idx][0]
        i = self.total_pair[idx][1]
        r_ui = self.total_pair[idx][2]
        j = np.random.randint(self.num_item)
        while j in self.user_list[u]:
            j = np.random.randint(self.num_item)
        return u, i, r_ui, j

    def get_batch(self):
        for i in range(self.num_epochs * (len(self.total_pair) // args.batch_size)):
            randomNum = random.Random().random()
            # print(randomNum)
            if randomNum < args.lambda_mod:
                isClick = True
            else:
                isClick = False
            if isClick:
                idx = np.random.choice(list(range(len(self.click_pair))), size=args.batch_size)
                for index in idx:
                    yield self._example(index)
            else:
                idx = np.random.choice(list(range(len(self.buy_pair))), size=args.batch_size)
                for index in idx:
                    yield self._example(index + len(self.click_pair))


def precision_and_recall_k(user_emb, item_emb, bias_emb, train_user_list, test_user_list, klist, batch=512):
    """Compute precision at k using GPU.

    Args:
        user_emb (torch.Tensor): embedding for user [user_num, dim]
        item_emb (torch.Tensor): embedding for item [item_num, dim]
        train_user_list (list(set)):
        test_user_list (list(set)):
        k (list(int)):
    Returns:
        (torch.Tensor, torch.Tensor) Precision and recall at k
    """
    # Calculate max k value
    max_k = max(klist)
    train_user_list = [[x[0] for x in l] for l in train_user_list]

    # Compute all pair of training and test record
    result = None
    for i in range(0, user_emb.shape[0], batch):
        # Create already observed mask
        mask = user_emb.new_ones([min([batch, user_emb.shape[0] - i]), item_emb.shape[0]])
        for j in range(batch):
            if i + j >= user_emb.shape[0]:
                break
            mask[j].scatter_(dim=0, index=torch.tensor(list(train_user_list[i + j]), device=args.device),
                             value=torch.tensor(0.0, device=args.device))
        # Calculate prediction value
        cur_result = torch.mm(user_emb[i:i + min(batch, user_emb.shape[0] - i), :], item_emb.t()) + bias_emb.t()
        cur_result = torch.sigmoid(cur_result)
        assert not torch.any(torch.isnan(cur_result))
        # Make zero for already observed item
        cur_result = torch.mul(mask, cur_result)
        _, cur_result = torch.topk(cur_result, k=max_k, dim=1)
        result = cur_result if result is None else torch.cat((result, cur_result), dim=0)

    result = result.cpu()
    # Sort indice and get test_pred_topk
    warmup_user_count = 0
    precisions, recalls = [], []
    for k in klist:
        precision, recall = 0, 0
        for i in range(user_emb.shape[0]):
            if len(train_user_list[i]) == 0 or len(test_user_list[i]) == 0:
                continue
            warmup_user_count += 1
            test = set(test_user_list[i])
            pred = set(result[i, :k].numpy().tolist())
            val = len(test & pred)
            precision += val / k
            recall += val / len(test)
        precisions.append(precision / warmup_user_count)
        recalls.append(recall / warmup_user_count)
    return precisions, recalls


def main(args):
    # Initialize seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load preprocess data
    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
        click_pair, buy_pair = dataset['click_train_pair'], dataset['buy_train_pair']
    print('Load complete')

    # Create dataset, model, optimizer
    dataset = TripletUniformPair(item_size, train_user_list, click_pair, buy_pair, True, args.n_epochs, args.lambda_mod)
    loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)
    model = HoToR(user_size, item_size, args.dim, args.weight_decay).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    writer = SummaryWriter(
        log_dir=os.path.join("runs", f'{args.dataset}',
                             datetime.now().strftime("%b%d_%H-%M-%S") + "_" + socket.gethostname())
    )

    # Training
    idx = 0
    epoch_start_time = time.time()
    start_time = time.time()
    batch_sum = (len(dataset.click_pair) + len(dataset.buy_pair)) // args.batch_size
    for u, i, r_ui, j in loader:
        optimizer.zero_grad()
        loss = model(u, i, r_ui, j)
        loss.backward()
        optimizer.step()

        writer.add_scalar('train/loss', loss, idx)
        epoch = idx // ((len(dataset.click_pair) + len(dataset.buy_pair)) // args.batch_size) + 1
        batch_num = idx % batch_sum + 1
        if batch_num == 1:
            epoch_start_time = time.time()
        # ????????????epoch??? ??????????????????
        # ??? args.print_every_batch ??? batchsize ??? ??????????????????
        if batch_num % args.print_every_batch == 0:
            print('| epoch {:3d} | {:4d}/{:4d} batches | {:4.4f} ms/batch | loss {:4.4f}'
                  .format(epoch, batch_num, batch_sum,
                          (time.time() - epoch_start_time) * 1000 / batch_num,
                          loss))
        if batch_num == batch_sum:
            if epoch % args.eval_every_epoch == 0 or epoch == 1:
                eval(epoch, idx, loss, model, start_time, test_user_list, train_user_list, writer)
            print('-' * 89)
        if epoch % args.save_every_epoch == 1 and batch_num == 1:
            dirname = os.path.dirname(os.path.abspath(args.model))
            os.makedirs(dirname, exist_ok=True)
            torch.save(model.state_dict(), args.model)
        idx += 1
    torch.save(model.state_dict(), args.model)
    eval(epoch, idx, loss, model, start_time, test_user_list, train_user_list, writer)


def eval(epoch, idx, loss, model, start_time, test_user_list, train_user_list, writer):
    plist, rlist = precision_and_recall_k(model.U.detach(),
                                          model.V.detach(),
                                          model.biasV.detach(),
                                          train_user_list,
                                          test_user_list,
                                          klist=[1, 5, 10])

    print('=' * 89)
    print(
        '| end of epoch {:4d} | time: {:4.4f}s | valid loss: {:4.4f} | P@1: {:4.4f} | P@5: {:4.4f} | '
        'P@10: {:4.4f} | R@1: {:4.4f} | R@5: {:4.4f} | R@10: {:4.4f} |'
        .format(
            epoch, time.time() - start_time, loss,
            plist[0], plist[1], plist[2],
            rlist[0], rlist[1], rlist[2]
        )
    )
    print('=' * 89)

    writer.add_scalars('eval', {'P@1': plist[0],
                                'P@5': plist[1],
                                'P@10': plist[2]}, idx)
    writer.add_scalars('eval', {'R@1': rlist[0],
                                'R@5': rlist[1],
                                'R@10': rlist[2]}, idx)


if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default="BookCrossing",
                        help="Proceeding Dataset")
    parser.add_argument('--data',
                        type=str,
                        default=os.path.join('preprocessed', 'BookCrossing', 'BookCrossing.pickle'),
                        help="File path for data")
    # Seed
    parser.add_argument('--seed',
                        type=int,
                        default=random.randint(0, 1000),
                        help="Seed (For reproducability)")
    # Model
    parser.add_argument('--dim',
                        type=int,
                        default=20,
                        help="Dimension for embedding")
    # Optimizer
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.01,
                        help="Weight decay factor")
    # Training
    parser.add_argument('--n_epochs',
                        type=int,
                        default=800,
                        help="Number of epoch during training")
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help="Batch size in one iteration")
    parser.add_argument('--lambda_mod',
                        type=float,
                        default=0.5,
                        help="Probability for selecting from click_pair")
    parser.add_argument('--print_every_batch',
                        type=int,
                        default=50,
                        help="Period for printing per batch")
    parser.add_argument('--print_every_epoch',
                        type=int,
                        default=20,
                        help="Period for printing per epoch")
    parser.add_argument('--save_every_epoch',
                        type=int,
                        default=50,
                        help="Period for saving model during training")
    parser.add_argument('--eval_every_epoch',
                        type=int,
                        default=20,
                        help="Period for evaluating precision and recall during training")
    parser.add_argument('--device',
                        type=str,
                        default="cpu",
                        help="model running on X device")
    parser.add_argument('--model',
                        type=str,
                        default=os.path.join('output', 'BookCrossing', 'bpr.pt'),
                        help="File path for model")
    parser.add_argument('--worker_count',
                        type=int,
                        default=8,
                        help="Number of CPU worker")
    args = parser.parse_args()

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        if args.device == "cpu":
            print("WARNING: You have a GPU device, so you should probably run with --device cuda/mps ")

    main(args)
