import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import *
import os
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/XiAn_City',help='data path')
parser.add_argument('--adjdata',type=str,default='data/XiAn_City/adj_mat.pkl',help='adj data path')
parser.add_argument('--adjdatacluster',type=str,default='data/XiAn_City/adj_mat_cluster.pkl',help='adj data path')
parser.add_argument('--transmit',type=str,default='data/XiAn_City/transmit.csv',help='data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--in_dim_cluster',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=792,help='number of nodes')
parser.add_argument('--cluster_nodes',type=int,default=40,help='number of cluster')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0000,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=50,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument("--force", type=str, default=False,help="remove params dir", required=False)
parser.add_argument('--save',type=str,default='./garage/XiAn_City',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--model',type=str,default='gwnet',help='adj type')
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')


args = parser.parse_args()
seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main():
    #load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    sensor_ids_cluster, sensor_id_to_ind_cluster, adj_mx_cluster = util.load_adj(args.adjdatacluster,args.adjtype)
    dataloader = util.load_dataset_cluster(args.data, args.batch_size, args.batch_size, args.batch_size)
    #scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    supports_cluster = [torch.tensor(i).to(device) for i in adj_mx_cluster]
    transmit_np=np.float32(np.loadtxt(args.transmit,delimiter=','))
    transmit=torch.tensor(transmit_np).to(device)
    
    
    print(args)
    
    if args.model=='H_GCN':
        engine = trainer7( args.in_dim,args.in_dim_cluster, args.seq_length, args.num_nodes,args.cluster_nodes,args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports,supports_cluster,transmit,args.decay
                         )
    elif args.model=='H_GCN_wdf':
        engine = trainer6( args.in_dim,args.in_dim_cluster, args.seq_length, args.num_nodes,args.cluster_nodes,args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports,supports_cluster,transmit,args.decay
                         )
    # check parameters file
    params_path=args.save+"/"+args.model
    if os.path.exists(params_path) and not args.force:
        raise SystemExit("Params folder exists! Select a new params path please!")
    else:
        if os.path.exists(params_path):
            shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Create params directory %s' % (params_path))

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        
        dataloader['train_loader_cluster'].shuffle()
        
        for iter,(x,y,x_cluster,y_cluster) in enumerate(dataloader['train_loader_cluster'].get_iterator()):
            
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            trainx_cluster = torch.Tensor(x_cluster).to(device)
            trainx_cluster= trainx_cluster.transpose(1, 3)
            trainy_cluster = torch.Tensor(y_cluster).to(device)
            trainy_cluster = trainy_cluster.transpose(1, 3)
            metrics = engine.train(trainx,trainx_cluster,trainy[:,0,:,:],trainy_cluster)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
            
        #engine.scheduler.step()
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        
        for iter,(x,y,x_cluster,y_cluster) in enumerate(dataloader['val_loader_cluster'].get_iterator()):          
            validx = torch.Tensor(x).to(device)
            validx = validx.transpose(1, 3)
            validy = torch.Tensor(y).to(device)
            validy = validy.transpose(1, 3)
            validx_cluster = torch.Tensor(x_cluster).to(device)
            validx_cluster = validx_cluster.transpose(1, 3)
            validy_cluster = torch.Tensor(y_cluster).to(device)
            validy_cluster = validy_cluster.transpose(1, 3)
            metrics = engine.eval(validx,validx_cluster,validy[:,0,:,:],validy_cluster)
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])
        
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), params_path+"/"+args.model+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(params_path+"/"+args.model+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
    engine.model.eval()
    
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    
    realy = realy.transpose(1,3)[:,0,:,:]
    #print(realy.shape)
    for iter, (x,y,x_cluster,y_cluster) in enumerate(dataloader['test_loader_cluster'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        testx_cluster = torch.Tensor(x_cluster).to(device)
        testx_cluster = testx_cluster.transpose(1, 3)
        with torch.no_grad():
            preds,_,_ = engine.model(testx,testx_cluster)
            preds=preds.transpose(1,3)
        outputs.append(preds.squeeze())
    for iter, (x,y,x_cluster,y_cluster) in enumerate(dataloader['test_loader_cluster'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        testx_cluster = torch.Tensor(x_cluster).to(device)
        testx_cluster = testx_cluster.transpose(1, 3)
        with torch.no_grad():
            _,spatial_at,parameter_adj = engine.model(testx,testx_cluster)
        break
            
        
    
    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    #print(yhat.shape)
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    amae = []
    amape = []
    armse = []
    prediction=yhat
    for i in range(12):
        pred = prediction[:,:,i]
        #pred = scaler.inverse_transform(yhat[:,:,i])
        #prediction.append(pred)
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
    
    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(),params_path+"/"+args.model+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
    prediction_path=params_path+"/"+args.model+"_prediction_results"
    ground_truth=realy.cpu().detach().numpy()
    prediction=prediction.cpu().detach().numpy()
    spatial_at=spatial_at.cpu().detach().numpy()
    parameter_adj=parameter_adj.cpu().detach().numpy()
    np.savez_compressed(
            os.path.normpath(prediction_path),
            prediction=prediction,
            spatial_at=spatial_at,
            parameter_adj=parameter_adj,
            ground_truth=ground_truth
        )


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
