import time
import numpy as np
import torch
import logging
#from dgl.contrib.sampling.sampler import NeighborSampler
# import torch.nn as nn
# import torch.nn.functional as F



from train.loss import loss_function,init_center,get_radius,EarlyStopping

from utils import fixed_graph_evaluate

def train(args,logger,data,model,path):
    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.gpu)
    checkpoints_path=path

    # logging.basicConfig(filename=f"./log/{args.dataset}+OC-{args.module}.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
    # logger=logging.getLogger('OCGNN')
    #loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer AdamW
    logger.info('Start training')
    logger.info(f'dropout:{args.dropout}, nu:{args.nu},seed:{args.seed},lr:{args.lr},self-loop:{args.self_loop},norm:{args.norm}')

    logger.info(f'n-epochs:{args.n_epochs}, n-hidden:{args.n_hidden},n-layers:{args.n_layers},weight-decay:{args.weight_decay}')

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    # initialize data center

    input_feat=data['features']
    input_g=data['g']

    noise = np.load(args.noise_path,allow_pickle=True)
    # noise = noise.tolist()
    # for indexi,i in enumerate(noise):
    #     for indexj,j in enumerate(i):
    #         noise[indexi,indexj]= float(j)
    noise = torch.tensor(noise*0.01,device=device)
    data_center= init_center(args,input_g,input_feat, model,noise)
    radius=torch.tensor(0, device=device)# radius R initialized with 0 by default.


    arr_epoch=np.arange(args.n_epochs)
    arr_loss=np.zeros(args.n_epochs)
    arr_valauc=np.zeros(args.n_epochs)
    arr_testauc=np.zeros(args.n_epochs)
    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        #model.train()
        #if epoch %5 == 0:
        t0 = time.time()
        # forward

        outputs= model(input_g,input_feat,noise,args.noise_d)
        #print('model:',args.module)
        #print('output size:',outputs.size())

        loss,dist,_=loss_function(args.nu, data_center,outputs,data['index_normal_train'],data['index_abnormal_train'],radius,data['train_mask'])
        arr_loss[epoch]=loss.item()
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch>= 3:
            dur.append(time.time() - t0)

        radius.data=torch.tensor(get_radius(dist, args.nu), device=device)


        auc,ap,f1,acc,precision,recall,val_loss,precision1,precision2,precision3 = fixed_graph_evaluate(args,checkpoints_path, model, data_center,data,radius,noise,data['val_mask'],data['index_normal_val'],data['index_abnormal_val'])
        arr_valauc[epoch]=auc
        print("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} | Val AUROC {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item()*100000,
                                            val_loss.item()*100000, auc, data['n_edges'] / np.mean(dur) / 1000))
        print(f'Val f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
        if args.early_stop:
            if stopper.step(auc,val_loss.item(), model,epoch,checkpoints_path):
                break

    if args.early_stop:
        print('loading model before testing.')
        model.load_state_dict(torch.load(checkpoints_path))


    auc,ap,f1,acc,precision,recall,loss,precision1,precision2,precision3 = fixed_graph_evaluate(args,checkpoints_path,model, data_center,data,radius,noise,data['test_mask'],data['index_normal_test'],data['index_abnormal_test'])
    test_dur = 0
    arr_testauc[epoch]=auc
    print("Test Time {:.4f} | Test AUROC {:.4f} | Test AUPRC {:.4f}".format(test_dur,auc,ap))
    print(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    logger.info("Current epoch: {:d} Test AUROC {:.4f} | Test AUPRC {:.4f}".format(epoch,auc,ap))
    logger.info(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)},pre@50:{round(precision1,4)},pre@100:{round(precision2,4)},pre@200:{round(precision3,4)},')
    logger.info('\n')

    #np.savez('SAGE-2.npz',epoch=arr_epoch,loss=arr_loss,valauc=arr_valauc,testauc=arr_testauc)

    return model


