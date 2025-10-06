"""Test the victim models"""
import argparse
import numpy as np
import os
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.spatial import cKDTree
import open3d as o3d
from scipy.spatial import Delaunay
import trimesh
from sklearn.neighbors import NearestNeighbors
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
sys.path.append('./')
from baselines.dataset import ModelNet40Attack, ModelNet40Transfer, load_data
from baselines.model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg, Pct, PointTransformerCls
from baselines.util.utils import AverageMeter, str2bool, set_seed
from baselines.config import BEST_WEIGHTS
from baselines.config import MAX_TEST_BATCH as BATCH_SIZE
from baselines.config import MAX_DUP_TEST_BATCH as DUP_BATCH_SIZE
from baselines.attack import L2Dist, ChamferDist, HausdorffDist
import torch
from pytorch3d.structures import Meshes


def adjacency_matrix(num_vertices, faces, sparse=True):
    r"""Calculates a adjacency matrix of a mesh.

    Args:
        num_vertices (int): Number of vertices of the mesh.
        faces (torch.LongTensor):
            Faces of shape :math:`(\text{num_faces}, \text{face_size})` of the mesh.
        sparse (bool): Whether to return a sparse tensor or not. Default: True.

    Returns:
        (torch.FloatTensor or torch.sparse.FloatTensor): adjacency matrix

    Example:
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> adjacency_matrix(3, faces)
        tensor(indices=tensor([[0, 0, 1, 1, 2, 2],
                               [1, 2, 0, 2, 0, 1]]),
               values=tensor([1., 1., 1., 1., 1., 1.]),
               size=(3, 3), nnz=6, layout=torch.sparse_coo)
    """
    device = faces.device

    forward_i = torch.stack([faces, torch.roll(faces, 1, dims=-1)], dim=-1)
    backward_i = torch.stack([torch.roll(faces, 1, dims=-1), faces], dim=-1)
    indices = torch.cat([forward_i, backward_i], dim=1).reshape(-1, 2)
    indices = indices.unique(dim=0)

    if sparse:
        indices = indices.t()
        # If vertex i and j have an edge connect to it, A[i, j] = 1
        values = torch.ones(indices.shape[1], device=device)
        adjacency = torch.sparse.FloatTensor(indices, values, (num_vertices, num_vertices))
    else:
        adjacency = torch.zeros((num_vertices, num_vertices), device=device, dtype=torch.float)
        adjacency[indices[:, 0], indices[:, 1]] = 1

    return adjacency

def uniform_laplacian(num_vertices, faces):
    r"""Calculates the uniform laplacian of a mesh.
    :math:`L[i, j] = \frac{1}{num\_neighbours(i)}` if i, j are neighbours.
    :math:`L[i, j] = -1` if i == j. 
    :math:`L[i, j] = 0` otherwise.

    Args:
        num_vertices (int): Number of vertices for the mesh.
        faces (torch.LongTensor):
            Faces of shape :math:`(\text{num_faces}, \text{face_size})` of the mesh.

    Returns:
        (torch.Tensor):
            Uniform laplacian of the mesh of size :math:`(\text{num_vertices}, \text{num_vertices})`
    Example:
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> uniform_laplacian(3, faces)
        tensor([[-1.0000,  0.5000,  0.5000],
                [ 0.5000, -1.0000,  0.5000],
                [ 0.5000,  0.5000, -1.0000]])
    """
    batch_size = faces.shape[0]

    dense_adjacency = adjacency_matrix(num_vertices, faces).to_dense()

    # Compute the number of neighbours of each vertex
    num_neighbour = torch.sum(dense_adjacency, dim=1).view(-1, 1)

    L = torch.div(dense_adjacency, num_neighbour)

    torch.diagonal(L)[:] = -1

    # Fill NaN value with 0
    L[torch.isnan(L)] = 0

    return L


def uniform_laplacian_smoothing(vertices, faces):
    r"""Calculates the uniform laplacian smoothing of meshes.
    The position of updated vertices is defined as :math:`V_i = \frac{1}{N} * \sum^{N}_{j=1}V_j`,
    where :math:`N` is the number of neighbours of :math:`V_i`, :math:`V_j` is the position of the
    j-th adjacent vertex.

    Args:
        vertices (torch.Tensor):
            Vertices of the meshes, of shape :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        faces (torch.LongTensor):
            Faces of the meshes, of shape :math:`(\text{num_faces}, \text{face_size})`.

    Returns:
        (torch.FloatTensor):
            smoothed vertices, of shape :math:`(\text{batch_size}, \text{num_vertices}, 3)`.

    Example:
        >>> vertices = torch.tensor([[[1, 0, 0],
        ...                           [0, 1, 0],
        ...                           [0, 0, 1]]], dtype=torch.float)
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> uniform_laplacian_smoothing(vertices, faces)
        tensor([[[0.0000, 0.5000, 0.5000],
                 [0.5000, 0.0000, 0.5000],
                 [0.5000, 0.5000, 0.0000]]])
    """
    dtype = vertices.dtype
    num_vertices = vertices.shape[1]

    laplacian_matrix = uniform_laplacian(num_vertices, faces).to(dtype)
    smoothed_vertices = torch.matmul(laplacian_matrix, vertices)
    loss = smoothed_vertices.norm(dim=2)
    #loss2 = loss.view(B, 1002)
    loss2 = loss.view(vertices.shape[0], num_vertices)
    loss3 = torch.sum(loss2, dim=1)
    return loss3



def batched_cdist(a, b):
    """
    Compute pairwise distance between each pair of points in a and b in a batched manner.
    """
    diff = a.unsqueeze(2) - b.unsqueeze(1)
    return torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-9)

def batched_laplacian_loss(faces, vertices):
    """
    Calculate Laplacian loss for each mesh in the batch.
    """
    B, N = faces.shape[0], faces.shape[1]

    # Reshaping vertices and faces to fit embedding requirements
    vertices_flat = vertices.view(B * vertices.shape[1], -1)
    faces_flat = faces.view(-1, faces.shape[-1])

    # Embedding and reshaping to get mesh
    meshes = F.embedding(faces_flat, vertices_flat).view(B, N, -1, 3)

    # Calculating edges
    edge_1 = meshes[:, :, 1] - meshes[:, :, 0]
    edge_2 = meshes[:, :, 2] - meshes[:, :, 0]
    edge_3 = meshes[:, :, 1] - meshes[:, :, 2]

    # Calculating distances
    dis = torch.stack([torch.norm(edge_1, dim=-1), torch.norm(edge_2, dim=-1), torch.norm(edge_3, dim=-1)], dim=2)
    return dis



def batched_triangle_indices(distances, k):
    """
    Generate triangle indices for each point cloud in the batch.
    """
    _, indices = torch.topk(distances, k=k+1, largest=False)
    B, N, _ = indices.shape
    triangles = []
    for j in range(1, k + 1):
        triangles.append(torch.stack([torch.arange(N).unsqueeze(0).repeat(B, 1).cuda(), indices[:, :, j], indices[:, :, (j % k) + 1]], dim=-1))
    return torch.cat(triangles, dim=1)



def batched_angles(faces, vertices):
    B, N = faces.shape[0], faces.shape[1]

    # Reshaping vertices and faces to fit embedding requirements
    vertices_flat = vertices.view(B * vertices.shape[1], -1)
    faces_flat = faces.view(-1, faces.shape[-1])

    # Embedding and reshaping to get mesh
    meshes = torch.nn.functional.embedding(faces_flat, vertices_flat).view(B, N, -1, 3)

    # Calculating edges
    edge_1 = meshes[:, :, 1] - meshes[:, :, 0]
    edge_2 = meshes[:, :, 2] - meshes[:, :, 0]
    edge_3 = meshes[:, :, 2] - meshes[:, :, 1]

    # Calculating angle results
    result = torch.zeros((B, N, 3), dtype=vertices.dtype)
    result[:, :, 0] = torch.acos(torch.clamp((edge_1.mul(edge_2)).sum(dim=2), -1, 1))
    result[:, :, 1] = torch.acos(torch.clamp((edge_3.mul(-edge_1)).sum(dim=2), -1, 1))
    result[:, :, 2] = math.pi - result[:, :, 0] - result[:, :, 1]

    return result


def batched_vertex_defects(vertices, faces):
    B, N = vertices.shape[0], vertices.shape[1]
    B1, N1 = faces.shape[0], faces.shape[1]
    face_angles = batched_angles(faces, vertices).cuda()

    columns = len(vertices[0])
    row = faces[0].view(-1) #3072*3=9216
    col = torch.arange(len(faces[0])).view(-1, 1).repeat(1, faces.shape[2]).view(-1).int().cuda() #9216

    i = torch.cat((row, col)).view(2, -1).repeat(B, 1, 1) #(2, 9216)
    spar_matrix = []
    for j in range(B):
         matrix = torch.sparse_coo_tensor(indices=i[j], values=face_angles[j].flatten(), size=(columns, len(faces[0])))
         angle_sum = matrix.to_dense().sum(dim=1).flatten()
         defect = (2 * math.pi) - angle_sum
         spar_matrix.append(defect)
    defect = torch.stack(spar_matrix, dim=0)
    return defect


def batched_discrete_gaussian_curvature_measure(vertices, faces, points, radius):
    # 计算顶点缺陷
    defects = batched_vertex_defects(vertices, faces)
    
    # 将顶点数据转为 numpy 数组
    vertices_np = [v.cpu().numpy() for v in vertices]

    # 构建 KD 树
    kd_trees = [cKDTree(verts) for verts in vertices_np]

    # 查询最近邻
    nearest_lists = [tree.query_ball_point(pts, radius) for tree, pts in zip(kd_trees, points)]

    # 计算高斯曲率
    gauss_curv = [[defects[i][neigh].sum() for neigh in nearest] for i, nearest in enumerate(nearest_lists)]

    return torch.tensor(gauss_curv).to(vertices.device)




def write_ply(save_path, points,text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]  #[batchsize,1024,3]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)


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
            ori_data_lst.append(ori_data.cpu().numpy())
            adv_data_lst.append(adv_data.detach().cpu().numpy())
            label_lst.append(label.cpu().numpy())
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


def test_target():
    """Target test mode.
    Show both classification accuracy and target success rate.
    """
    model.eval()
    acc_save = AverageMeter()
    success_save = AverageMeter()
    with torch.no_grad():
        for data, label, target in tqdm(test_loader):
            data, label, target = \
                data.float().cuda(), label.long().cuda(), target.long().cuda()
            # to [B, 3, N] point cloud
            data = data.transpose(1, 2).contiguous()
            batch_size = label.size(0)
            # batch in
            if args.model.lower() == 'pointnet':
                logits = model(data)[0]
            else:
                logits = model(data)[0]
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == label).sum().float() / float(batch_size)
            acc_save.update(acc.item(), batch_size)
            success = (preds == target).sum().float() / float(batch_size)
            success_save.update(success.item(), batch_size)

    print('Overall accuracy: {:.4f}, '
          'attack success rate: {:.4f}'.
          format(acc_save.avg, success_save.avg))


def test_normal():
    """Normal test mode.
    Test on all data.
    """
    model.eval()
    denom=0
    l2_dist = 0
    chamfer_dist=0
    hausdorf_dist=0
    with torch.no_grad():
        for ori_data, adv_data, label in test_loader:
            ori_data, adv_data, label = \
                ori_data.float().cuda(), adv_data.float().cuda(), label.long().cuda()
            double_points = torch.cat((ori_data, ori_data), dim=0)
            double_adv_points = torch.cat((adv_data, adv_data), dim=0)
            # # Batched distance calculation
            # batch_distances = batched_cdist(double_points, double_points)
            # k = 3

            # # Batched triangle generation
            # batch_triangles = batched_triangle_indices(batch_distances, k)
            # #  Convert the point cloud data to a numpy array for visualization
            # point_cloud_np = double_points[0].cpu().numpy()
            # point_cloud_adv_np = double_adv_points[0].cpu().numpy()
            # # Create a mesh using the generated triangles and point cloud
            # mesh = trimesh.Trimesh(vertices=point_cloud_np, faces=batch_triangles[0].cpu())
            # mesh_adv = trimesh.Trimesh(vertices=point_cloud_adv_np, faces=batch_triangles[0].cpu())
            # # Save the mesh to a PLY file
            # mesh.export('point_cloud_mesh.ply')
            # mesh_adv.export('point_cloud_adv_mesh.ply')
            # # Batched Laplacian loss calculation
            # dist = batched_laplacian_loss(batch_triangles, double_points)
            # dis_adv = batched_laplacian_loss(batch_triangles, double_adv_points)
            # loss_distance_2 = torch.sum((dis_adv - dist) ** 2, dim=(1, 2))

            # # Calculate final loss
            # final_loss = torch.mean(loss_distance_2)

            batch_distances = batched_cdist(double_points, double_points)
            k = 3

            # Batched triangle generation
            batch_triangles = batched_triangle_indices(batch_distances, k)
            loss_laplacian = uniform_laplacian_smoothing(double_points, batch_triangles)
            
            gaussian_curvature = batched_discrete_gaussian_curvature_measure(double_points, batch_triangles, double_points.cpu().numpy(), 0.05)

            def angles(face, vertex):
                meshes = torch.nn.functional.embedding(face, vertex)
                edge_1 = meshes[:, 1] - meshes[:, 0]
                edge_2 = meshes[:, 2] - meshes[:, 0]
                edge_3 = meshes[:, 2] - meshes[:, 1]
                result = torch.zeros((len(face), 3), dtype=torch.float64)
                result[:, 0] = torch.acos(torch.clamp((edge_1.mul(edge_2)).sum(dim=1), -1, 1))
                result[:, 1] = torch.acos(torch.clamp((edge_3.mul(-edge_1)).sum(dim=1), -1, 1))
                result[:, 2] = math.pi - result[:, 0] - result[:, 1]
                return result
            
            def vertex_defects(vertices, faces):
                face_angles = angles(faces, vertices).cuda()

                columns = len(vertices)

                row = faces.view(-1) #3072*3=9216
                col = torch.arange(len(faces)).view(-1, 1).repeat(1, faces.shape[1]).view(-1).int().cuda() #9216

                i = torch.cat((row, col)).view(2, -1) #(2, 9216)

                matrix = torch.sparse_coo_tensor(indices=i, values=face_angles.flatten(), size=(columns, len(faces)))

                angle_sum = matrix.to_dense().sum(dim=1).flatten()
                defect = (2 * math.pi) - angle_sum
                return defect
            
            def discrete_gaussian_curvature_measure(vertices, faces, points, radius):

                defects = vertex_defects(vertices, faces)
                kdtree = cKDTree(vertices.detach().cpu().numpy())

                nearest = kdtree.query_ball_point(points, radius)
                gauss_curv = [defects[neigh].sum() for neigh in nearest]

                return gauss_curv

            all_loss_distance_2 = []
            k=3
            for point_cloud_data, point_cloud_adv_data in zip(double_points, double_adv_points):
                distances = torch.cdist(point_cloud_data, point_cloud_data, p=2)
                _, indices = torch.topk(distances, k=k+1, largest=False)
                triangles = []
                for i in range(len(point_cloud_data)):
                    for j in range(1, k + 1):
                        triangles.append((i, indices[i, j], indices[i, (j % k) + 1]))
                triangles = torch.tensor(triangles).cuda()
                gaussian_curvature_ = discrete_gaussian_curvature_measure(point_cloud_data, triangles, point_cloud_data.cpu().numpy(), 0.05)
               

                #  Convert the point cloud data to a numpy array for visualization
                point_cloud_np = point_cloud_data.cpu().numpy()
                point_cloud_adv_np = point_cloud_adv_data.cpu().numpy()
                # Create a mesh using the generated triangles and point cloud
                mesh = trimesh.Trimesh(vertices=point_cloud_np, faces=triangles.cpu())
                mesh_adv = trimesh.Trimesh(vertices=point_cloud_adv_np, faces=triangles.cpu())
                # Save the mesh to a PLY file
                mesh.export('point_cloud_mesh.ply')
                mesh_adv.export('point_cloud_adv_mesh.ply')

                all_loss_distance_2.append(loss_distance_2)
            final_loss = torch.mean(torch.stack(all_loss_distance_2))
            #
            batch_size = label.size(0)
            l2dist = L2Dist()(adv_data, ori_data, batch_avg=False)
            chdist = ChamferDist()(adv_data, ori_data, batch_avg=False)
            hsdist = HausdorffDist()(adv_data, ori_data, batch_avg=False)
            
            denom += float(batch_size)
            l2_dist += l2dist.sum()
            chamfer_dist += chdist.sum()
            hausdorf_dist += hsdist.sum()
 

    print('Overall L2 dist: {:.4f}'.format(l2_dist))  
    # print('Overall accuracy: {:.4f}'.format(num / (denom + 1e-9)))
    print('Overall CHamfer dist: {:.4f}'.format(chamfer_dist)) #模型本身的分类成功率
    print('Overall Hausdorf dist: {:.4f}'.format(hausdorf_dist) )


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='attack/results/SP/player_15_1024/sharply_feature_attack')
    parser.add_argument('--prefix', type=str,
                        default='Usharply_feature_attack-pointnet2-0.18-success')
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['normal', 'target'],
                        help='Testing mode')
    parser.add_argument('--batch_size', type=int, default=10, metavar='BS',
                        help='Size of batch, use config if not specified')
    parser.add_argument('--model', type=str, default='pointconv', metavar='MODEL',
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
