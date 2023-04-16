import numpy as np
from sklearn.model_selection import train_test_split
from torch import manual_seed
def semi_one_class_processing(data,anormal_class:int,args=None):
    labels,orilabel=semi_one_class_labeling(data.labels,anormal_class)
    return semi_one_class_masking(labels,orilabel)


def semi_one_class_labeling(labels,anormal_class:int):
    normal_idx=np.where(labels!=anormal_class)[0]
    abnormal_idx=np.where(labels==anormal_class)[0]
    orilabel=np.zeros_like(labels)
    labels[normal_idx]=0
    labels[abnormal_idx]=1
    orilabel[normal_idx]=0
    orilabel[abnormal_idx]=1
    return labels,orilabel


def semi_one_class_masking(labels,orilabel):
    train_mask = np.zeros(labels.shape,dtype='bool')
    val_mask = np.zeros(labels.shape,dtype='bool')
    test_mask = np.zeros(labels.shape,dtype='bool')
    indices_normal  = np.where(orilabel==0)[0]
    indices_abnormal  = np.where(orilabel==1)[0]
    train_idx,test_idx = train_test_split(indices_normal,test_size=0.2,random_state=0)
    train_idx,val_idx = train_test_split(train_idx,test_size=0.2,random_state=0)

    train_abidx,test_abidx =  train_test_split(indices_abnormal,test_size=0.8,random_state=0)
    train_abidx,val_abidx = train_test_split(train_abidx,test_size=0.8,random_state=0)
    unlabeled_idx,_ =  train_test_split(train_idx,test_size=0.05,random_state=0)
    np.random.shuffle(train_abidx)
    unlabeled_abidx = train_abidx[20:]
    labels[unlabeled_idx] = -1
    labels[unlabeled_abidx]=-1
    train_mask[train_idx]=1
    test_mask[test_idx]=1
    val_mask[val_idx]=1
    train_mask[train_abidx]=1
    test_mask[test_abidx]=1
    val_mask[val_abidx]=1
    return labels,orilabel,train_mask,val_mask,test_mask




