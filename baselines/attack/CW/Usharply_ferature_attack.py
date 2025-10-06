"""Implementation of optimization based attack,
    CW Attack for point perturbation.
Based on CVPR'19: Generating 3D Adversarial Point Clouds.
"""

import pdb
import time
from pytorch3d.ops import knn_points, knn_gather
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn


def feature_distance_loss(feature1, feature2, alpha):
    t_feature1 = torch.sign(feature1) * torch.pow(torch.abs(feature1+1e-4), alpha)
    t_feature2 = torch.sign(feature2) * torch.pow(torch.abs(feature2+1e-4), alpha)
    dis = torch.norm(t_feature1-t_feature2, 2, dim=1)
    return dis


def offset_proj(offset, ori_pc, ori_normal, project='dir'):
    # offset: shape [b, 3, n], perturbation offset of each point
    # normal: shape [b, 3, n], normal vector of the object
    ori_offset = offset.clone()
    if offset.shape[1] !=3:
        offset = offset.transpose(1, 2).contiguous()
        ori_pc = ori_pc.transpose(1, 2).contiguous()
        ori_normal = ori_normal.transpose(1, 2).contiguous()
    condition_inner = torch.zeros(offset.shape).cuda().byte()

    intra_KNN = knn_points(offset.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    normal = knn_gather(ori_normal.permute(0,2,1), intra_KNN.idx).permute(0,3,1,2).squeeze(3).contiguous() # [b, 3, n]

    normal_len = (normal**2).sum(1, keepdim=True).sqrt()
    normal_len_expand = normal_len.expand_as(offset) #[b, 3, n]

    # add 1e-6 to avoid dividing by zero
    offset_projected = (offset * normal / (normal_len_expand + 1e-6)).sum(1,keepdim=True) * normal / (normal_len_expand + 1e-6)

    # let perturb be the projected ones
    offset = torch.where(condition_inner, offset, offset_projected)
    if offset.shape[1]!=ori_offset.shape[1]:
        offset = offset.transpose(1, 2).contiguous()
    return offset



def _compare(output, target, gt, targeted):
    if targeted:
        return output == target
    else:
        return output != gt
    

def sample(delta, pp):
    b, s, n = delta.size()
    only_add_one_mask = torch.from_numpy(np.random.choice([0, 1], size=(b,s,n), p=[1 - pp, pp])).cuda()


    leave_one_mask = 1 - only_add_one_mask

    only_add_one_perturbation = delta * only_add_one_mask
    leave_one_out_perturbation = delta * leave_one_mask

    return only_add_one_perturbation, leave_one_out_perturbation



def get_features(
    model,
    x,
    perturbation,
    leave_one_out_perturbation,
    only_add_one_perturbation,
):

    outputs = model(x + perturbation)
    leave_one_outputs = model(x + leave_one_out_perturbation)
    only_add_one_outputs = model(x + only_add_one_perturbation)

    return (outputs, leave_one_outputs, only_add_one_outputs)



class CWsharply_feature:
    """Class for CW attack.
    """

    def __init__(self, model_name, model, com_noise,pos_noise,neg_noise,adv_func, dist_func, players, k_sharp, initial_const, pp, trans_weight, attack_lr=1e-2,
                 binary_step=2, num_iter=200, clip_func=None):
        """CW attack by perturbing points.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            dist_func (function): distance metric
            attack_lr (float, optional): lr for optimization. Defaults to 1e-2.
            binary_step (int, optional): binary search step. Defaults to 10.
            num_iter (int, optional): max iter num in every search step. Defaults to 500.
        """

        self.model = model.cuda()
        self.model.eval()
        self.model_name = model_name
        self.com_noise = com_noise
        self.pos_noise=pos_noise
        self.neg_noise=neg_noise
        self.adv_func = adv_func
        self.dist_func = dist_func
        self.attack_lr = attack_lr
        self.pp = pp
        self.binary_step = binary_step
        self.num_iter = num_iter
        self.clip_func = clip_func
        self.players = players
        self.k_sharp = k_sharp
        self.initial_const = initial_const
        self.trans_weight = trans_weight

    def attack(self, data, normal, target, sharply):
        """Attack on given data to target.

        Args:
            data (torch.FloatTensor): victim data, [B, num_points, 3]
            target (torch.LongTensor): target output, [B]
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        if self.model_name !=  'pointtransformer':
            data = data.transpose(1, 2).contiguous()
        ori_data = data.clone().detach()
        ori_data.requires_grad = False

        normal = normal.float().cuda().detach()
        if self.model_name !=  'pointtransformer':
            normal = normal.transpose(1, 2).contiguous()
        normal_data = normal.clone().detach()
        normal_data.requires_grad = False

        target = target.long().cuda().detach()
        label_val = target.detach().cpu().numpy()  # [B]

        
        lower_bound = torch.ones(B) * 0
        scale_const = torch.ones(B) * self.initial_const
        upper_bound = torch.ones(B) * 1e10

        best_loss = [1e10] * B
        if self.model_name !=  'pointtransformer':
            best_attack = torch.ones(B, 3, K).to(data.device)
        else:
            best_attack = torch.ones(B, K, 3).to(data.device)
        best_attack_step = [-1] * B
        bestscore = np.array([-1] * B)
        best_attack_BS_idx = [-1] * B
        all_loss_list = [[-1] * B] * self.num_iter
        

        with torch.no_grad():
            logits = self.model(data)  # [B, num_classes]
        if isinstance(logits, tuple):  # PointNet
            feature = logits[-1][0]  ###默认是feature = logits[-1][-1]
        N = feature.shape[-1] // self.players
        mask_book_list = []
        for i in range(B):
            mask_book = {}
            for code in range(self.players):
                x = code % self.players
                mask_i = torch.zeros_like(feature[0])
                if feature.shape[-1] - (x+1)*N < N:
                    mask_i[x*N:] = 1
                else:
                    mask_i[x*N:(x+1)*N] = 1
                mask_book[code] = mask_i
            mask_book_list.append(mask_book)
        
        mask_pos_code_list = []
        for i in range(B):
            shapley_values = sharply[i]
            index_pos = torch.where(shapley_values>0)
            if index_pos[0].shape[0]==0:
                _,index = torch.topk(shapley_values, self.k_sharp)
            else:
                if index_pos[0].shape[0]< self.k_sharp:
                    index = index_pos[0]
                else:
                    _,index_top = torch.topk(shapley_values[index_pos[0]], self.k_sharp)
                    index = index_pos[0][index_top]

            mask_book = mask_book_list[i]
            mask_pos = []
            for idx in index:
                mask_pos.append(mask_book[int(idx)])
            mask_pos_code = torch.stack(mask_pos).sum(dim=0)
            mask_pos_code_list.append(mask_pos_code)
        mask_pos_code = torch.stack(mask_pos_code_list)
        mask_neg_code = torch.logical_not(mask_pos_code).float()
        feature_pos_ori = feature * mask_pos_code
        feature_neg_ori = feature * mask_neg_code

        # perform binary search
        per_batch_time = time.time()
        for binary_step in range(self.binary_step):
            
            iter_best_loss = [1e10] * B
            iter_best_score = [-1] * B
            constrain_loss = torch.ones(B) * 1e10
            attack_success = torch.zeros(B).cuda()

            input_all = None

            total_time = 0.
            optimize_time = 0.
            clip_time = 0.
            update_time = 0.

            # one step in binary search
            for iteration in range(self.num_iter):
                if iteration == 0:
                    if self.model_name !=  'pointtransformer':
                        offset = torch.zeros(B, 3, K).cuda()
                    else:
                        offset = torch.zeros(B, K, 3).cuda()
                    nn.init.normal_(offset, mean=0, std=1e-3)
                    offset.requires_grad_()

                    optimizer = optim.Adam([offset], lr=self.attack_lr)
                    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9990, last_epoch=-1)
                    periodical_pc = ori_data.clone()
                
                input_all = periodical_pc + offset
                input_curr_iter = input_all

                with torch.no_grad():
                    adv_output = self.model(input_curr_iter)
                    output_label = torch.argmax(adv_output[0], dim=1)
                    pred_val = output_label.detach().cpu().numpy()  # [B]
                    attack_success = _compare(output_label, target, target.cuda(), False)
                    metric = constrain_loss.detach().clone().cpu()

                    for e, (attack, pred, metric_, input_, output_label_) in enumerate(zip(attack_success, pred_val, metric, input_curr_iter, output_label)):
                        if attack and metric_ < best_loss[e]:
                            best_loss[e] = metric_
                            bestscore[e] = pred
                            best_attack[e] = input_.clone()
                            best_attack_BS_idx[e] = binary_step
                            best_attack_step[e] = iteration
                        if attack and (metric_ <iter_best_loss[e]):
                            iter_best_loss[e] = metric_
                            iter_best_score[e] = output_label_

                t1 = time.time()
                scale_const = scale_const.float().cuda()
                logits = self.model(input_curr_iter)
                if isinstance(logits, tuple):  # PointNet
                    logits = logits[0]
                optimizer.zero_grad()
                adv_loss = self.adv_func(logits, target)
                if self.model_name !=  'pointtransformer':
                    dist_loss = self.dist_func(input_curr_iter.transpose(1, 2).contiguous(), ori_data.transpose(1, 2).contiguous(), weights=None, batch_avg=False)
                else:
                    dist_loss = self.dist_func(input_curr_iter, ori_data, weights=None, batch_avg=False)
                dist_loss_ = scale_const * dist_loss
                loss_0 = adv_loss + dist_loss_
                loss_0.mean().backward()
                only_add_one_perturbation, leave_one_out_perturbation = sample(offset, self.pp)

                outputs = self.model(ori_data + offset)
                trans_loss = torch.zeros_like(loss_0).cuda()
                if self.com_noise:
                    complete_feature_loss = -feature_distance_loss(outputs[-1][0], feature_pos_ori, 0.5)
                    loss1 = scale_const * self.trans_weight* complete_feature_loss
                    trans_loss += loss1.detach().clone()
                    loss1.mean().backward()

                leave_one_outputs = self.model(ori_data + leave_one_out_perturbation)
                if self.neg_noise:
                    pos_feature_loss = -feature_distance_loss(leave_one_outputs[-1][0], feature_pos_ori, 0.5)
                    loss2 = scale_const * self.trans_weight* pos_feature_loss
                    trans_loss += loss2.detach().clone()
                    loss2.mean().backward()
                
                if self.pos_noise:
                    only_add_one_outputs = self.model(ori_data + only_add_one_perturbation)
                    neg_feature_loss = feature_distance_loss(only_add_one_outputs[-1][0], feature_neg_ori, 0.5)
                    loss3 = scale_const * self.trans_weight* neg_feature_loss
                    trans_loss += loss3.detach().clone()
                    loss3.mean().backward()
                
                constrain_loss = dist_loss.clone()
                if self.com_noise:
                    constrain_loss = constrain_loss + self.trans_weight*complete_feature_loss
                if self.neg_noise:
                    constrain_loss = constrain_loss + self.trans_weight*pos_feature_loss 
                if self.pos_noise:
                    constrain_loss = constrain_loss + self.trans_weight*neg_feature_loss
                loss_n = adv_loss + scale_const * constrain_loss
                all_loss_list[iteration] = loss_n.detach().tolist()
             
                optimizer.step()

                t2 = time.time()
                optimize_time += t2 - t1

                #clip
                with torch.no_grad():
                    proj_offset = offset_proj(offset, ori_data, normal_data)
                    offset.data = proj_offset.data
                offset.data = self.clip_func(offset.data)
                t3 = time.time()
                clip_time += t3 - t2

                # print
                with torch.no_grad():
                    adv_data = ori_data + offset
                    flogits = self.model(adv_data)  # [B, num_classes]
                    if isinstance(flogits, tuple):  # PointNet
                        flogits = flogits[0]
                    pred = torch.argmax(flogits, dim=1)  # [B]
                success_num = (pred != target).sum().item()
                if iteration % (self.num_iter // 5) == 0:
                    print('Step {}, iteration {}, success {}/{}\n'
                          'adv_loss: {:.4f}, trans_loss: {:.4f}, dist_loss: {:.4f}'.
                          format(binary_step, iteration, success_num, B,
                                 (adv_loss.mean().item()), trans_loss.mean().item(), dist_loss.mean().item()))

            # adjust the scale constants
            for k in range(B):
                if _compare(output_label[k], target[k], target[k].cuda(), False).item() and iter_best_score[k] != -1:
                    lower_bound[k] = max(lower_bound[k], scale_const[k])
                    if upper_bound[k] < 1e9:
                        scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5
                    else:
                        scale_const[k] *= 2
                else:
                    upper_bound[k] = min(upper_bound[k], scale_const[k])
                    if upper_bound[k] < 1e9:
                        scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5

            torch.cuda.empty_cache()
        t_batch = time.time()
        total_batch_time = t_batch - per_batch_time
        print('ANF total time: {:.2f}'.format(total_batch_time))
        # end of CW attack
        # fail to attack some examples
        fail_idx = (bestscore < 0)
        best_attack[fail_idx] = input_curr_iter[fail_idx]

        adv_pc = best_attack
        logits = self.model(adv_pc)
        if isinstance(logits, tuple):  # PointNet
            logits = logits[0]
        preds = torch.argmax(logits, dim=-1)
        # return final results
        print(preds.shape)
        success_num = (preds != target).sum().item()
        print('Successfully attack {}/{}'.format(success_num, B))
        return adv_pc, success_num
