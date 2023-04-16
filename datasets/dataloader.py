from dgl.data import load_data, tu
from dgl import DGLGraph
import torch
import torch.utils.data
import numpy as np
from scipy import sparse
import torch
import dgl
import networkx as nx
from scipy.io import savemat
from datasets.semi_prepocessing import semi_one_class_processing
def loader(args):
    # load and preprocess dataset
    
    data = load_data(args)

    print(f'anormal_class is {args.anormal_class}')

    labels,orilabel,train_mask,val_mask,test_mask=semi_one_class_processing(data,args.anormal_class,args)
    index_normal_train =  np.where(labels[train_mask]==0)[0]
    index_abnormal_train = np.where(labels[train_mask]==1)[0]
    index_normal_test = np.where(labels[test_mask]==0)[0]
    index_abnormal_test = np.where(labels[test_mask]==1)[0]
    index_normal_val = np.where(labels[val_mask]==0)[0]
    index_abnormal_val = np.where(labels[val_mask]==1)[0]

    index_abnormal_train=torch.LongTensor(index_abnormal_train)
    index_normal_train=torch.LongTensor(index_normal_train)
    index_abnormal_val=torch.LongTensor(index_abnormal_val)
    index_normal_val=torch.LongTensor(index_normal_val)
    index_abnormal_test=torch.LongTensor(index_abnormal_test)
    index_normal_test=torch.LongTensor(index_normal_test)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(labels)
    orilabel = torch.LongTensor(orilabel)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.sum().item(),
              val_mask.sum().item(),
              test_mask.sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        orilabel = orilabel.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        index_abnormal_train=index_abnormal_train.cuda()
        index_normal_train=index_normal_train.cuda()
        index_abnormal_val=index_abnormal_val.cuda()
        index_normal_val=index_normal_val.cuda()
        index_abnormal_test=index_abnormal_test.cuda()
        index_normal_test=index_normal_test.cuda()
    # graph preprocess and calculate normalization factor
    g = data.graph
    
        
    # add self loop
    if args.self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        #g=transform.remove_self_loop(g)
        #if args.module!='GraphSAGE':
        g.add_edges_from(zip(g.nodes(), g.nodes()))

    g = DGLGraph(g)
    g = g.to(torch.device('cuda:0'))
    n_edges = g.number_of_edges()
    if args.norm:
        
        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        if cuda:
            norm = norm.cuda()
        g.ndata['norm'] = norm.unsqueeze(1)

    datadict={'g':g,'features':features,'labels':labels,'orilabel':orilabel,'train_mask':train_mask,
        'val_mask':val_mask,'test_mask': test_mask,'input_dim':in_feats,'n_classes':n_classes,'n_edges':n_edges,
        'index_normal_test':index_normal_test,'index_abnormal_test':index_abnormal_test,
        'index_normal_train':index_normal_train,'index_abnormal_train':index_abnormal_train,
        'index_normal_val':index_normal_val,'index_abnormal_val':index_abnormal_val}

    return datadict

def emb_dataloader(args):
    # load and preprocess dataset
    data = load_data(args)
    anormal_class=args.anormal_class
    labels,train_mask,val_mask,test_mask=semi_one_class_processing(data,anormal_class,args)

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.sum().item(),
              val_mask.sum().item(),
              test_mask.sum().item()))

    g = data.graph


    datadict={'g':g,'features':features,'labels':labels,'train_mask':train_mask,
        'val_mask':val_mask,'test_mask': test_mask,'in_feats':in_feats,'n_classes':n_classes,'n_edges':n_edges}

    return datadict