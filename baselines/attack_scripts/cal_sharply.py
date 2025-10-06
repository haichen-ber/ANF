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
from baselines.model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg, PointNetCls_feature, PointNet2ClsSsg_feature, PointConvDensityClsSsg_feature, DGCNN_feature, Pct_feature, PointTransformerCls_feature,\
                            Pct, PointTransformerCls
from baselines.util.utils import str2bool, set_seed
from baselines.attack import CrossEntropyAdvLoss, UntargetedLogitsAdvLoss
from baselines.attack import L2Dist
from baselines.attack import ClipPointsLinf


def predict_fn(mdoel, inputs, lbl):
    logits = mdoel(inputs.unsqueeze(0))
    v = logits[:, lbl] - torch.logsumexp(logits[:, np.arange(40) != lbl.item()], dim=1)
    return v
        

def compute_shapley_value(model, input_batches, mask_book, player_index, labels, num_samples=1000):
    num_batches = len(input_batches) #6
    num_players = len(player_index) #
    batched_shapley_values = torch.zeros(num_batches, num_players)

    for batch in range(num_batches):
        input_data = input_batches[batch] ##一个特征，256维
        label = labels[batch]
        shapley_values = torch.zeros(num_players) #每一个区域表示的shapley

        for _ in range(num_samples):
            permutation = torch.randperm(num_players).long() #随机生成1-15的值，打乱的
            player_present = torch.zeros(num_players).bool() #初始都是false

            prev_inputs = torch.zeros_like(input_data)
            prev_prediction = predict_fn(model, prev_inputs, label) #得到反馈

            for i in range(num_players):
                player_present[permutation[i]] = True #将该玩家加入
                new_inputs = torch.zeros_like(input_data)

                for idx, present in enumerate(player_present):
                    if present:
                        new_inputs += mask_book[idx] * input_data ###加入该部分表示的特征

                new_prediction = predict_fn(model, new_inputs, label)
                marginal_contribution = new_prediction - prev_prediction #反馈的差异

                shapley_values[permutation[i]] += marginal_contribution.item() #该玩家获取的收益计入

                prev_inputs = new_inputs
                prev_prediction = new_prediction#记录上次玩家加入得到收益

        shapley_values /= num_samples
        batched_shapley_values[batch] = shapley_values

    return batched_shapley_values


def save_pickle(data, file_name):
	f = open(file_name, "wb")
	pickle.dump(data, f)
	f.close()


def attack(args):
    model.eval()
    model_feature.eval()
    for i, (pc, _,label) in enumerate(test_loader, 1):
        scattered_result_path = os.path.join('sharply_value/%s_feature1' %(args.model.lower()), str(i)+'.pkl')
        if os.path.exists(scattered_result_path):
            print(scattered_result_path, 'exists! pass!')
            continue
        else:
            with open(scattered_result_path, "wb") as f:
                pickle.dump(' ', f)
        pc, label = pc.float().cuda(non_blocking=True), label.long().cuda(non_blocking=True)
        pc = pc.transpose(1, 2).contiguous()
        if args.model.lower() == 'pointtransformer':
            pc = pc.transpose(1, 2).contiguous()
        logits_ = model(pc)  # [B, num_classes]
        if isinstance(logits_, tuple):  # PointNet
            # logits = logits_[0]
            feature_first = logits_[-1][0] #(B,1024) 获取特征
        N = feature_first.shape[-1] // args.num_players
        mask_book = {}
        for code in range(args.num_players):
            x = code % args.num_players
            mask_i = torch.zeros_like(feature_first[0])
            if feature_first.shape[-1] - (x+1)*N < N:
                mask_i[x*N:] = 1
            else:
                mask_i[x*N:(x+1)*N] = 1
            mask_book[code] = mask_i ###里面都是01的值，判断该区域是否选取
        batched_shapley_values = compute_shapley_value(model_feature, feature_first, mask_book, list(range(args.num_players)), label)
        save_pickle(batched_shapley_values, 'sharply_value/%s/%s_feature1_%splayers_%s.pkl' %(args.model.lower(), args.model, args.num_players, i))

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='baselines/official_data/modelnet40_normal_resampled')
    parser.add_argument('--model', type=str, default='pointnet', metavar='MODEL',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv', 'pointtransformer', 'pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]. '
                             'If not specified, judge from data_root')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40',
                                 'opt_mn40', 'conv_opt_mn40', 'aug_mn40'])
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
                        default=None)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    #### sharply value ####
    parser.add_argument('--num_players', type=int, default=64)
    parser.add_argument('--M', type=int, default=5)
      
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    #PT参数
    parser.add_argument('--model_name', default='Hengshuang', help='model name')
    parser.add_argument('--nneighbor', type=int, default=16)
    parser.add_argument('--nblocks', type=int, default=4)
    parser.add_argument('--transformer_dim', type=int, default=512)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--num_class', type=int, default=40)
    #
    #PCT参数
    parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate')
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
    if args.model.lower() == 'pointtransformer':
        model = PointTransformerCls(args) 
        model_feature = PointTransformerCls_feature(args)
    elif args.model.lower() == 'pct':
        model = Pct(args) # 模型输入[B,3,N]
        model_feature = Pct_feature(output_channels=40)
    elif args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)
        model_feature = DGCNN_feature(40)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
        model_feature = PointNetCls_feature(k=40, dim=1024)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
        model_feature = PointNet2ClsSsg_feature(40)
    elif args.model.lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40)
        model_feature = PointConvDensityClsSsg_feature(40)
    else:
        print('Model not recognized')
        exit(-1)

    # load model weight
    state_dict = torch.load(
        BEST_WEIGHTS[args.model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))
    if args.model.lower() == 'pointtransformer':
        state_dict = state_dict['model_state_dict']
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
    