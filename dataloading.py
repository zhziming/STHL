import os
import numpy as np
import torch

def get_raw_data(path):
    # raw data
    middle_data = np.load(path)
    train_feature = middle_data["train_feature"]
    val_feature = middle_data["val_feature"]
    train_targets = middle_data["train_targets"]
    val_targets = middle_data["val_targets"]
    return train_feature,val_feature,train_targets,val_targets

def get_input_data(path):
    # data with temporal window sliced
    middle_data = np.load(path)
    train_feature = middle_data["train_feature"]
    val_feature = middle_data["val_feature"]
    train_label = middle_data["train_label"]
    val_label = middle_data["val_label"]
    test_feature  = middle_data["test_feature"]
    test_label  = middle_data['test_label']
    return train_feature, train_label, val_feature, val_label, test_feature, test_label

def window_split(feature,label,num,lag = 0):
    feature = [feature[i-lag:i+1] for i in range(lag+1,num)]
    label = [label[i] for i in range(lag+1,num)]
    return np.array(feature),np.array(label)

def shuffle(feature,label):
    num = feature.shape[0]
    idx = [i for i in range(num)]
    np.random.shuffle(idx)
    return feature[idx],label[idx]
    
def preprocess_data(train_feature,val_feature,train_targets,val_targets,lag = 1):
    train_num = train_feature.shape[0]
    test_num = val_feature.shape[0]
    train_feature,train_label = window_split(train_feature,train_targets,train_num,lag = lag)

    test_feature,test_label = window_split(val_feature,val_targets,test_num,lag = lag)
    val_feature,val_label = train_feature[train_num-test_num:],train_label[train_num-test_num:]
    train_feature, train_label = train_feature[:train_num-test_num],train_label[:train_num-test_num]

    test_feature, test_label = shuffle(test_feature,test_label)
    val_feature, val_label = shuffle(val_feature,val_label)
    train_feature, train_label = shuffle(train_feature, train_label)
    np.savez("input_data.npz",
             train_feature=train_feature,
             val_feature=val_feature,
             train_label= train_label,
             val_label=val_label,
             test_feature = test_feature,
             test_label = test_label
             )
    return train_feature, train_label, val_feature, val_label, test_feature, test_label
    
def read_data(raw_path,output_path,lag):
    if os.path.exists(output_path):
        train_feature, train_label, val_feature, val_label, test_feature, test_label = get_input_data(output_path)
    else:
        train_feature, val_feature, train_targets, val_targets = get_raw_data(raw_path)
        train_feature, train_label, val_feature, val_label, test_feature, test_label  = \
            preprocess_data(train_feature, val_feature, train_targets, val_targets,lag = lag)
    return torch.Tensor(train_feature), torch.Tensor(train_label), torch.Tensor(val_feature), torch.Tensor(val_label), torch.Tensor(test_feature), torch.Tensor(test_label)