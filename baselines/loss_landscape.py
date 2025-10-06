"""Test the victim models"""
import argparse
import numpy as np
import os
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
sys.path.append('../')
sys.path.append('./')
from baselines.dataset import ModelNet40Attack, ModelNet40Transfer, load_data
from baselines.model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg, Pct, PointTransformerCls
from baselines.util.utils import AverageMeter, str2bool, set_seed
from baselines.config import BEST_WEIGHTS
from baselines.config import MAX_TEST_BATCH as BATCH_SIZE
from baselines.config import MAX_DUP_TEST_BATCH as DUP_BATCH_SIZE


loss = nn.CrossEntropyLoss(reduction='none')

def merge(data_root, prefix):
    ori_data_lst = []
    adv_data_lst = []
    label_lst = []
    save_name = prefix+"-concat.npz"
    if os.path.exists(os.path.join(data_root, save_name)):
        return os.path.join(data_root, save_name)
    for file in os.listdir(data_root):
        if file.startswith(prefix):
            file_path = os.path.join(data_root, file)
            ori_data, adv_data, label = load_data(file_path, partition='transfer')
            ori_data_lst.append(ori_data)
            adv_data_lst.append(adv_data)
            label_lst.append(label)
    if len(ori_data_lst)==1:
        all_ori_pc = ori_data_lst[0]  # [num_data, K, 3]
        all_adv_pc = adv_data_lst[0]  # [num_data, K, 3]
        all_real_lbl = label_lst[0]
    else:
        all_ori_pc = np.concatenate(ori_data_lst, axis=0)  # [num_data, K, 3]
        all_adv_pc = np.concatenate(adv_data_lst, axis=0)  # [num_data, K, 3]
        all_real_lbl = np.concatenate(label_lst, axis=0)  # [num_data]

    np.savez(os.path.join(data_root, save_name),
             ori_pc=all_ori_pc.astype(np.float32),
             test_pc=all_adv_pc.astype(np.float32),
             test_label=all_real_lbl.astype(np.uint8))
    return os.path.join(data_root, save_name)

def merge_attack(data_root, prefix):
    target_label_lst = []
    adv_data_lst = []
    label_lst = []
    save_name = prefix+"-concat.npz"
    if os.path.exists(os.path.join(data_root, save_name)):
        return os.path.join(data_root, save_name)
    for file in os.listdir(data_root):
        if file.startswith(prefix):
            file_path = os.path.join(data_root, file)
            adv_data, label, target_label = \
                load_data(file_path, partition='attack')
            adv_data_lst.append(adv_data)
            label_lst.append(label)
            target_label_lst.append(target_label)
    all_adv_pc = np.concatenate(adv_data_lst, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(label_lst, axis=0)  # [num_data]
    all_target_lbl = np.concatenate(target_label_lst, axis=0)  # [num_data]

    np.savez(os.path.join(data_root, save_name),
             test_pc=all_adv_pc.astype(np.float32),
             test_label=all_real_lbl.astype(np.uint8),
             target_label=all_target_lbl.astype(np.uint8))
    return os.path.join(data_root, save_name)


def get_model_name(npz_path):
    """Get the victim model name from npz file path."""
    if 'dgcnn' in npz_path.lower():
        return 'dgcnn'
    if 'pointconv' in npz_path.lower():
        return 'pointconv'
    if 'pointnet2' in npz_path.lower():
        return 'pointnet2'
    if 'pointnet' in npz_path.lower():
        return 'pointnet'
    print('Victim model not recognized!')
    exit(-1)

def compute_perturb(model, image, label, vec_x, vec_y, range_x, range_y,
                 grid_size=50, loss=nn.CrossEntropyLoss(reduction='none'),
                 batch_size=128):
    rx = np.linspace(*range_x, grid_size)
    ry = np.linspace(*range_y, grid_size)

    images = []
    loss_list = []

    image = image.to('cuda')
    label = label.to('cuda')
    vec_x = vec_x.to('cuda')
    vec_y = vec_y.to('cuda')

    for j in ry :
        for i in rx :
            images.append(image + i*vec_x + j*vec_y)

            if len(images) == batch_size :
                images = torch.stack(images)
                labels = torch.stack([label]*batch_size)
                outputs = model(images.transpose(1, 2).contiguous().cuda())[0]
                loss_list.append(loss(outputs, labels).data.cpu().numpy())
                images = []

    images = torch.stack(images)
    labels = torch.stack([label]*len(images))
    outputs = model(images.transpose(1, 2).contiguous().cuda())[0]
    loss_list.append(loss(outputs, labels).data.cpu().numpy())
    loss_list = np.concatenate(loss_list).reshape(len(rx), len(ry))

    return rx, ry, loss_list

def plot_perturb_plt(rx, ry, zs, save_path, eps,
                     title=None, width=8, height=7, linewidth = 0.1,
                     pane_color=(0.0, 0.0, 0.0, 0.01),
                     tick_pad_x=0, tick_pad_y=0, tick_pad_z=1.5,
                     xlabel=None, ylabel=None, zlabel=None,
                     xlabel_rotation=0, ylabel_rotation=0, zlabel_rotation=0,
                     view_azimuth=230, view_altitude=30,
                     light_azimuth=315, light_altitude=45, light_exag=0,
                     random=False) :

    xs, ys = np.meshgrid(rx, ry)

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111, projection='3d')

    if title is not None :
        ax.set_title(title)

    # The azimuth (0-360, degrees clockwise from North) of the light source. Defaults to 315 degrees (from the northwest).
    # The altitude (0-90, degrees up from horizontal) of the light source. Defaults to 45 degrees from horizontal.

    ls = LightSource(azdeg=light_azimuth, altdeg=light_altitude)
    cmap = plt.get_cmap('coolwarm')
    fcolors = ls.shade(zs, cmap=cmap, vert_exag=light_exag, blend_mode='soft')
    surf = ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, facecolors=fcolors,
                           linewidth=linewidth, antialiased=True, shade=False, alpha=0.7)
    contour = ax.contourf(xs, ys, zs, zdir='z', offset=np.min(zs), cmap=cmap)

    #surf.set_edgecolor(edge_color)
    ax.view_init(azim=view_azimuth, elev=view_altitude)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    if xlabel is not None :
        ax.set_xlabel(xlabel, rotation=xlabel_rotation)
    if ylabel is not None :
        ax.set_ylabel(ylabel, rotation=ylabel_rotation)
    if zlabel is not None :
        ax.set_zlabel(zlabel, rotation=zlabel_rotation)

    x_min, x_max = xs[0][0], xs[0][-1]
    xtick_step = np.linspace(x_min, x_max, 5)
    y_min, y_max = ys[0][0], ys[-1][-1]
    ytick_step = np.linspace(y_min, y_max, 5)

    ax.set_xticks(xtick_step)
    ax.set_xticklabels(['{}'.format(int(eps*i)) for i in xtick_step])
    ax.set_yticks(ytick_step)
    ax.set_yticklabels(['{}'.format(int(eps*i)) for i in ytick_step])
    #ax.set_zticks(None)

    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)

    ax.tick_params(axis='x', pad=tick_pad_x)
    ax.tick_params(axis='y', pad=tick_pad_y)
    ax.tick_params(axis='z', pad=tick_pad_z)

    if not random:
        plt.savefig('loss_landscape.png', format='png', dpi=300)
    else:
        plt.savefig(save_path('loss_landscape_rademacher'), format='png', dpi=300)
    print('saved loss landscape')



# 生成扰动
def perturb_cloud(cloud, rx, ry, rz, magnitudes, label, num):
    perturbed_clouds = []
    label_ = []
    for j, val0 in enumerate(rx) :
        for i, val1 in enumerate(ry) :
            for k, val2 in enumerate(rz):
                perturbed_cloud = cloud.cpu().numpy() + magnitudes[j*(num**2)+i*num+k] * np.array([val0, val1, val2])
                perturbed_clouds.append(torch.from_numpy(perturbed_cloud).float())
                label_.append(label)
    return torch.stack(perturbed_clouds, axis=0), torch.stack(label_, dim=0)


def minibatch_input(data, batch_size, model, label):
    loss_ = []
    
    num_batches = len(data) // batch_size  # 计算总共有多少个批次

    for i in range(num_batches):
        batch = data[i*batch_size : (i+1)*batch_size]  # 提取当前批次的数据
        batch_label = label[i*batch_size : (i+1)*batch_size]
        # 将 batch 输入模型进行处理
        adv_logits = model(batch)[0]
        losses = loss(adv_logits, batch_label)
        loss_.append(losses)
    # 如果数据总量不是批次大小的整数倍，则处理剩下的数据
    if len(data) % batch_size != 0:
        batch = data[num_batches*batch_size : ]  # 剩余的数据作为一个批次
        batch_label = label[num_batches*batch_size : ]
       # 将 batch 输入模型进行处理
        adv_logits = model(batch)[0]
        losses = loss(adv_logits, batch_label)
        loss_.append(losses)
    return torch.cat(loss_)



def test_normal():
    """Normal test mode.
    Test on all data.
    """
    model.eval()
    num = 0
    rx = np.linspace(-0.18, 0.18, 10)
    ry = np.linspace(-0.18, 0.18, 10)
    rz = np.linspace(-0.18, 0.18, 10)
    magnitudes = np.linspace(-1, 1, 10*10*10)  # 生成10000个不同的扰动幅度

    with torch.no_grad():
        for ori_data, adv_data, label in test_loader:
            ori_data, adv_data, label = \
                ori_data.float().cuda(), adv_data.float().cuda(), label.long().cuda()
            
            batch_size = label.size(0)
            
            if num == 0:
                adv_vec = adv_data[0]
                rademacher_vec = 2.*(torch.randint(2, size=adv_vec.shape)-1.) * 0.18
                x_ = ori_data[0]
                y_ = label[0]

                rx, ry, zs = compute_perturb(model=model,
                                    image=x_, label=y_,
                                    vec_x=adv_vec, vec_y=rademacher_vec,
                                    range_x=(-1,1), range_y=(-1,1),
                                    grid_size=50,
                                    loss=nn.CrossEntropyLoss(reduction='none'))
                print('computed adversarial loss landscape')
                plot_perturb_plt(rx, ry, zs, 'loss_lanscape', 18,
                                xlabel='Adv', ylabel='Rad',)
                
                
                # # 扰动点云
                # perturbed_clouds, labels = perturb_cloud(adv_data, rx, ry, rz, magnitudes, label, 10)
                # perturbed_clouds_tensor = perturbed_clouds.transpose(1, 2).contiguous().cuda()
                # losses = minibatch_input(perturbed_clouds_tensor, 100, model, labels)
                # import matplotlib.pyplot as plt
                # # 将损失值随着扰动幅度的变化进行曲线图可视化
                # plt.plot(magnitudes, losses.cpu().numpy())
                # plt.title('Loss Curve')  # 添加标题
                # plt.xlabel('Perturbation Magnitude')  # 添加x轴标签
                # plt.ylabel('Loss')  # 添加y轴标签
                # plt.grid(True)  # 显示网格
                # plt.savefig('loss_curve.png')  # 保存图像为文件
                # plt.show()
                break

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='attack/results/finetune_3d-adv')
    parser.add_argument('--prefix', type=str,
                        default='UFTAN-pointnet-0.18-success')
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['normal', 'target'],
                        help='Testing mode')
    parser.add_argument('--batch_size', type=int, default=100, metavar='BS',
                        help='Size of batch, use config if not specified')
    parser.add_argument('--model', type=str, default='pointnet', metavar='MODEL',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv', 'pointtransformer', 'pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]. '
                             'If not specified, judge from data_root')
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
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40',
                                 'opt_mn40', 'conv_opt_mn40', 'aug_mn40','ori_mn40'])
    parser.add_argument('--normalize_pc', type=str2bool, default=False,
                        help='normalize in dataloader')

    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='',
                        help='Model weight to load, use config if not specified')


    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    args = parser.parse_args()

    # victim model
    if not args.model:
        args.model = get_model_name(args.data_root)

    # random seed
    set_seed(1)

    # in case adding attack
    if 'add' in args.data_root.lower():
        # we add 512 points in adding attacks
        if args.num_points == 1024:
            num_points = 1024
            args.num_points = 1024 + 512
        elif args.num_points == 1024 + 512:
            num_points = 1024
    elif 'cluster' in args.data_root.lower():
        # we add 3*32=96 points in adding cluster attack
        if args.num_points == 1024:
            num_points = 1024
            args.num_points = 1024 + 3 * 32
        elif args.num_points == 1024 + 3 * 32:
            num_points = 1024
    elif 'object' in args.data_root.lower():
        # we add 3*64=192 points in adding object attack
        if args.num_points == 1024:
            num_points = 1024
            args.num_points = 1024 + 3 * 64
        elif args.num_points == 1024 + 3 * 64:
            num_points = 1024
    else:
        num_points = args.num_points

    # determine the weight to use
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][num_points]
    BATCH_SIZE = BATCH_SIZE[num_points]
    DUP_BATCH_SIZE = DUP_BATCH_SIZE[num_points]
    if args.batch_size == -1:  # automatic assign
        args.batch_size = BATCH_SIZE[args.model]
    # add point attack has more points in each point cloud
    if 'ADD' in args.data_root:
        args.batch_size = int(args.batch_size / 1.5)
    # sor processed point cloud has different points in each
    # so batch size only can be 1
    if 'sor' in args.data_root:
        args.batch_size = 1

    # enable cudnn benchmark
    cudnn.benchmark = True

    # build model
    if args.model.lower() == 'pointtransformer':
        model = PointTransformerCls(args) 
    elif args.model.lower() == 'pct':
        model = Pct(args) # 模型输入[B,3,N]
    elif args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
    elif args.model.lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40)
    else:
        print('Model not recognized')
        exit(-1)

    model = nn.DataParallel(model).cuda()

    # load model weight
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
    else:
        if args.model.lower() == 'pointtransformer':
            state_dict = torch.load(
            BEST_WEIGHTS[args.model], map_location='cpu')
            # concat 'module.' in keys
            state_dict = {'module.'+k: v for k, v in state_dict['model_state_dict'].items()}
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(torch.load(BEST_WEIGHTS[args.model]))

    # prepare data
    if args.mode == 'target':
        data_path = merge_attack(args.data_root, args.prefix)
        test_set = ModelNet40Attack(data_path, num_points=args.num_points,
                                    normalize=args.normalize_pc)
    else:
        data_path = merge(args.data_root, args.prefix)
        test_set = ModelNet40Transfer(data_path, num_points=args.num_point)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=8,
                             pin_memory=True, drop_last=False)

    # test
    if args.mode == 'normal':
        test_normal()
    else:
        test_target()