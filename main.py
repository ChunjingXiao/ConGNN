import argparse
import os
import torch
from dgl.data import register_data_args
import logging
#import fire
from train import trainer, TUtrainer, AEtrainer
from train.loss import loss_function,init_center
from datasets import dataloader,TUloader
from networks.init import init_model
import numpy as np
import torch
from dgl import random as dr

def main(args):
    if args.seed!=-1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        #torch.backends.cudnn.deterministic=True
        dr.seed(args.seed)

    checkpoints_path=f'./checkpoints/{args.dataset}+OC-{args.module}+bestcheckpoint.pt'
    logging.basicConfig(filename=f"./train_log/{args.dataset}+OC-{args.module}.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
    logger=logging.getLogger('OCGNN')


#     print('model:',args.module)
#     print('seed:',args.seed)

    if args.dataset in 'PROTEINS_full'+'ENZYMES'+'FRANKENSTEIN':
        train_loader, val_loader, test_loader, input_dim, label_dim=TUloader.loader(args)
        model=init_model(args,input_dim)
        model=TUtrainer.train(args,logger,train_loader,model,val_dataset=val_loader,path=checkpoints_path)
        # auc,ap,f1,acc,precision,recall,_= multi_graph_evaluate(args,checkpoints_path,
        #     model, data_center,test_loader,radius,mode='test')

        # torch.cuda.empty_cache()
        # print("Test AUROC {:.4f} | Test AUPRC {:.4f}".format(auc,ap))
        # print(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
        # logger.info("Current epoch: {:d} Test AUROC {:.4f} | Test AUPRC {:.4f}".format(epoch,auc,ap))
        # logger.info(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
        # logger.info('\n')
    else:
        data=dataloader.loader(args)
        model=init_model(args,data['input_dim'])
        if args.module != 'GAE':
            model=trainer.train(args,logger,data,model,checkpoints_path)
        else:
            model=AEtrainer.train(args,logger,data,model,checkpoints_path)

def parse_arguments(default_config="train.yaml"):
    parser = argparse.ArgumentParser(description="Running Experiments")
    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        default=os.path.join('config', default_config),
        # required=True,
        help="Path of config file")
    parser.add_argument(
        '-l',
        '--log_level',
        type=str,
        default='INFO',
        help="Logging Level, one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument('-m', '--comment', type=str,
                        default="", help="A single line comment for the experiment")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ConDA')
#     register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--nu", type=float, default=0.001 ,
            help="hyperparameter nu (must be 0 < nu <= 1)")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--seed", type=int, default=52,
            help="random seed, -1 means dont fix seed")
    parser.add_argument("--module", type=str, default='GCN',
            help="GCN/GAT/GIN/GraphSAGE/GAE")
    parser.add_argument('--n-worker', type=int,default=1,
            help='number of workers when dataloading')
    parser.add_argument('--batch-size', type=int,default=128,
            help='batch size')
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--normal-class", type=int, default=0,
            help="normal class")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=64,
            help="number of hidden gnn units")
    parser.add_argument("--n-layers", type=int, default=3,
            help="number of hidden gnn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--norm", action='store_true',
            help="graph normalization (default=False)")
    parser.add_argument("--dataset", type=str, default='pubmed',
            help="dataset")
    parser.add_argument("--noise_path", type=str, default="data/pubmedgenerate.npy",
            help="path of DDPM generate noise") 
    parser.add_argument("--noise_d", type=int, default=2,
            help="layer of noise inserted")   
    parser.add_argument("--beta_1", type=float, default=1e-4,
            help="start beta value")
    parser.add_argument("--beta_T", type=float, default=0.02,
            help="end beta value")    
    parser.add_argument("--mean_type", type=str, default='epsilon',
            help="start beta value")
    parser.add_argument("--var_type", type=str, default='fixedlarge',
            help="end beta value")   
    parser.add_argument("--ema_decay", type=float, default=0.9999,
            help="ema decay rate")  
    parser.set_defaults(self_loop=True)
    parser.set_defaults(norm=False)
    args = parser.parse_args()
    if args.module=='GCN':
        #args.self_loop=True
        args.norm=True
    if args.module=='GAE':
        args.lr=0.002
        args.dropout=0.
        args.weight_decay=0.
        # args.n_hidden=32
    #     args.self_loop=True
    # if args.module=='GraphSAGE':
    #     args.self_loop=True


    if args.dataset in ('citeseer' + 'reddit'):
        args.anormal_class=5
    if args.dataset in ('pubmed'):
        args.anormal_class=1
    if args.dataset in ('cora'):
        args.anormal_class=2
    if args.dataset in 'TU_PROTEINS_full':
        args.anormal_class=0
    
    main(args)
