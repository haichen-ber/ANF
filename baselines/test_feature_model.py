"""Targeted point perturbation attack."""

import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import pickle
import sys
sys.path.append('../')
sys.path.append('./')

from baselines.config import BEST_WEIGHTS
from baselines.config import MAX_PERTURB_BATCH as BATCH_SIZE
from baselines.dataset import ModelNet40Attack, ModelNetDataLoader
from baselines.model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg, PointNetCls_feature,\
    PointNet2ClsSsg_feature, PointConvDensityClsSsg_feature, DGCNN_feature, Pct_feature, Pct
from baselines.util.utils import str2bool, set_seed
from baselines.attack import CWPerturb
from baselines.attack import CrossEntropyAdvLoss, UntargetedLogitsAdvLoss
from baselines.attack import L2Dist
from baselines.attack import ClipPointsLinf


def predict_fn(mdoel, inputs):
    logits = mdoel(inputs)
    return logits
    


def attack(args):
    model.eval()
    model_feature.eval()
    num=0
    num1=0
    num2=0
    num3=0
    denom = 0
    num_sharp = 0
    num_pos = 0
    for i, (pc, _, label,sharply) in enumerate(test_loader, 1):
        pc, label = pc.float().cuda(non_blocking=True), label.long().cuda(non_blocking=True)
        pc = pc.transpose(1, 2).contiguous()
        logits_ = model(pc)  # [B, num_classes]
        pred = torch.argmax(logits_[0], dim=1)
        if isinstance(logits_, tuple):  # PointNet
            # logits = logits_[0]
            feature = logits_[-1][0] #(B,256)

        N = feature.shape[-1] // args.num_players
        mask_book_list = []
        for i in range(pc.shape[0]):
            mask_book = {}
            for code in range(args.num_players):
                x = code % args.num_players
                mask_i = torch.zeros_like(feature[0])
                if feature.shape[-1] - (x+1)*N < N:
                    mask_i[x*N:] = 1
                else:
                    mask_i[x*N:(x+1)*N] = 1
                mask_book[code] = mask_i
            mask_book_list.append(mask_book)
        mask_pos_code_list = []
        for i in range(pc.shape[0]):
            shapley_values = sharply[i]
            num_sharp += shapley_values.shape[0]
            index_pos = torch.where(shapley_values>0)
            if index_pos[0].shape[0]==0:
                _,index = torch.topk(shapley_values, args.k_sharp)
            else:
                if index_pos[0].shape[0]< args.k_sharp:
                    index = index_pos[0]
                else:
                    _,index_top = torch.topk(shapley_values[index_pos[0]], args.k_sharp)
                    index = index_pos[0][index_top]

            
            # index_thr = torch.where(shapley_values[index_pos[0]] > args.threshold)
            # if index_thr[0].shape[0] == 0:
            #     index = index_pos[0]
            # else:
            #     index = index_pos[0][index_thr]

            # index = torch.where(shapley_values>args.threshold)
            num_pos += index.shape[0]
            # _,index = torch.topk(shapley_values, args.k_sharp)
            # mean_sharply = shapley_values.mean()
            # mask_sharply = shapley_values > mean_sharply
            # index = torch.where(mask_sharply)
            mask_book = mask_book_list[i]
            mask_pos = []
            for idx in index:
                mask_pos.append(mask_book[int(idx)])
            mask_pos_code = torch.stack(mask_pos).sum(dim=0)
            mask_pos_code_list.append(mask_pos_code)
        mask_pos_code = torch.stack(mask_pos_code_list)
        mask_neg_code = torch.logical_not(mask_pos_code).float()
        # normVal = torch.norm(feature.view(feature.shape[0], -1), 2, 1)
        # feature = feature/normVal.view(feature.shape[0], 1)
        feature_neg_ori = feature * mask_neg_code
        feature_pos_ori = feature * mask_pos_code

        logits_com = predict_fn(model_feature, feature)
        pred_com = torch.argmax(logits_com, dim=1)
        logits_neg = predict_fn(model_feature, feature_neg_ori)
        pred_neg = torch.argmax(logits_neg, dim=1)
        logits_pos = predict_fn(model_feature, feature_pos_ori)
        pred_pos = torch.argmax(logits_pos, dim=1)
        mask_ori = (pred == label)
        mask_com = (pred_com == label)
        mask_neg = (pred_neg == label)
        mask_pos = (pred_pos==label)
        num += mask_ori.sum().float().item() #分类成功
        num1 += mask_com.sum().float().item() #分类成功
        num2 += mask_neg.sum().float().item() #分类成功
        num3 += mask_pos.sum().float().item() #分类成功
        denom += float(pc.shape[0])

    print('ori Overall accuracy: {:.4f}'.format(num / (denom + 1e-9))) #模型本身的分类成功率
    print('com Overall accuracy: {:.4f}'.format(num1 / (denom + 1e-9))) #模型本身的分类成功率
    print('neg Overall accuracy: {:.4f}'.format(num2 / (denom + 1e-9))) #模型本身的分类成功率
    print('pos Overall accuracy: {:.4f}'.format(num3 / (denom + 1e-9))) #模型本身的分类成功率
    print('all_num: {:.4f}'.format(num_sharp)) #模型本身的分类成功率
    print('pos_num: {:.4f}'.format(num_pos)) #模型本身的分类成功率
    print('neg_num: {:.4f}'.format(num_sharp-num_pos)) #模型本身的分类成功率
    
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='baselines/official_data/modelnet40_normal_resampled')
    parser.add_argument('--model', type=str, default='pointconv', metavar='N',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40',
                                 'opt_mn40', 'conv_opt_mn40', 'aug_mn40'])
    #PCT参数
    parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate')
    parser.add_argument('--k_sharp', type=int, default=5) #25-18 64-54  44-34 15-10
    parser.add_argument('--batch_size', type=int, default=16, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--kappa', type=float, default=30.,
                        help='min margin in logits adv loss')
    parser.add_argument('--budget', type=float, default=0.18,
                        help='clip budget')
    parser.add_argument('--attack_lr', type=float, default=1e-2,
                        help='lr in CW optimization')
    parser.add_argument('--binary_step', type=int, default=10, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter', type=int, default=500, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--sharply_root', type=str,
                        default='sharply_value/pointconv/pointconv_64players-concat.npz')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    #### sharply value #### threshold
    parser.add_argument('--num_players', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--M', type=int, default=5)
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    args = parser.parse_args()
    BATCH_SIZE = BATCH_SIZE[args.num_point]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_point]
    if args.batch_size == -1:
        args.batch_size = BATCH_SIZE[args.model]
    set_seed(1)
    print(args)

    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)
    cudnn.benchmark = True

    # build model
    if args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)
        model_feature = DGCNN_feature(40)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
        model_feature = PointNetCls_feature(k=40)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
        model_feature = PointNet2ClsSsg_feature(40)
    elif args.model.lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40)
        model_feature = PointConvDensityClsSsg_feature(40)
    elif args.model.lower() == 'pct':
        model = Pct(args)
        model_feature = Pct_feature(40)
    else:
        print('Model not recognized')
        exit(-1)

    # load model weight
    state_dict = torch.load(
        BEST_WEIGHTS[args.model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    # distributed mode on multiple GPUs!
    # much faster than nn.DataParallel
    # model = DistributedDataParallel(
    #     model.cuda(), device_ids=[args.local_rank])
    model = model.cuda()
    # for name,parameter in model_feature.state_dict().items():
    #     print(name)
    #     print(parameter)
    new_state_dict = model_feature.state_dict()
    pretrainde_dict = {k: v for k, v in state_dict.items() if k in new_state_dict}
    new_state_dict.update(pretrainde_dict)
    model_feature.load_state_dict(new_state_dict)
    model_feature = model_feature.cuda()
    # attack
    test_set = ModelNetDataLoader(root=args.data_root, args=args, split='test', process_data=args.process_data)
    # test_sampler = DistributedSampler(test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, 
                    shuffle=False, num_workers=4, 
                    drop_last=False, sampler=None)
    attack(args)
    
    ### concat all pickle ###
    