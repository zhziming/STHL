import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from model import Model
from sklearn import metrics
from tqdm import tqdm
from dataloading import read_data


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings

    parser.add_argument('--window_size', type=int, default=2, help='window size of inter block')
    parser.add_argument('--hid_dim', type=int, default=256, help='the number of hidden embedding')
    parser.add_argument('--n_cats', type=int, default=5, help='the number of categories')
    parser.add_argument('--raw_path', type=str, default="data.npz", help='the path of the raw data')
    parser.add_argument('--input_path', type=str, default="input_data.npz", help='the path of the input data')
    parser.add_argument('--epochs', type=int, default=10, help='the number of epoch')
    parser.add_argument('--batch_size', type=int, default=2, help='the number of epoch')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--frequency_of_the_test', type=int, default=100, help='How frequently to run eval')
    parser.add_argument('--cmp', type=str, default=None, help='comparison experiment')
    parser.add_argument('--att', type=bool, default=True, help='whether attention?')



    args = parser.parse_args(args=[])

    return args


def acc_f1(model, test_data,test_label):
    model.eval()
    test_num = len(test_data)
    with torch.no_grad():
        y_pred = []
        y_true = []
        for i in range(test_num):
            test = torch.Tensor(test_data[i])
            logits,reocn_loss = model(test)
            label = test_label[i]
            y_pred.append(logits.numpy())
            y_true.append(label.numpy())
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    pred = np.argmax(y_pred, axis=1)
    true = np.argmax(y_true, axis=1)

    return metrics.accuracy_score(true, pred), metrics.f1_score(true, pred, average='weighted')

def calculate_loss(model, data_list,label_list,batch_size, criterion):
    label_loss = 0
    recons_loss = 0
    for idx in range(batch_size):
        train_data = next(data_list)
        label = next(label_list)
        logits,r_loss = model(torch.Tensor(train_data))
        loss = criterion(logits, torch.Tensor(label))
        loss = loss.sum()
        label_loss += loss
        recons_loss += r_loss
    batch_loss = label_loss + recons_loss
    # print("label_loss:{},recons_loss:{}".format(label_loss,recons_loss))
    return batch_loss

def run_model(args):
    train_feature, train_label, val_feature, val_label, test_feature, test_label = read_data(args.raw_path, args.input_path, args.window_size-1)
    writer = SummaryWriter('./experiment_result')
    num_train  = train_feature.shape[0]
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    model = Model(args.window_size)
    weight_decay = 0.01
    if args.cmp == "simple":
        model = Model_simple(args.window_size)
        args.lr = 0.02
        weight_decay = 0
    if args.att == False:
        model = Model(args.window_size,att= args.att)
        args.lr = 0.01
        weight_decay = 0
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay = weight_decay)

    val_best_rlt = -1
    val_best_acc = -1
    print("win size = {}".format(args.window_size))
    global_step = 0
    
    for i in tqdm(range(args.epochs)):
        print("epoch:{}".format(i))
        tot_loss = 0
        for b_i in range(int(num_train / args.batch_size)):
            global_step += 1
            optimizer = opt
            optimizer.zero_grad()
            x_train = iter(train_feature[b_i*args.batch_size:(b_i+1)*args.batch_size])
            y_train = iter(train_label[b_i*args.batch_size:(b_i+1)*args.batch_size])
            batch_loss = calculate_loss(model, x_train, y_train, args.batch_size, criterion)
            tot_loss += batch_loss.detach().item()
            batch_loss.backward()
            optimizer.step()
            # print("batch:{} loss:{}".format(b_i,batch_loss))
            if b_i % args.frequency_of_the_test == 0 or b_i == int(num_train / args.batch_size) - 1:
                acc, f1 = acc_f1(model, val_feature, val_label)
                print("epoch:{},batch:{},acc:{},f1:{}".format(i,b_i,acc,f1))
                writer.add_scalar('val: acc', acc, global_step=global_step)
                writer.add_scalar('val: f1', f1, global_step=global_step)
                if f1 > val_best_rlt:
                    val_best_rlt = f1
                    state_dict = deepcopy(model.state_dict())
                if acc > val_best_acc:
                    val_best_acc = acc
                    state_dict = deepcopy(model.state_dict())
        print("epoch:{} completed".format(i))
        print(f"loss: {tot_loss}")
    
    model = Model(args.window_size)
    model.load_state_dict(state_dict)
    test_best_acc,test_best_rlt = acc_f1(model, test_feature, test_label)
    print("acc on test set:{},f1 on test set:{}".format(test_best_acc,test_best_rlt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    # % args.cmp = "simple"
    # % # args.att = False
    run_model(args)


