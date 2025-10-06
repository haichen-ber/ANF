import pickle
import os
import numpy as np
import torch

def load_pickle(file_name):
	f = open(file_name, "rb+")
	data = pickle.load(f)
	f.close()
	return data

save_name = 'pointnet_feature1_64players'+"-concat.npz"

sharply_list = []
for file in os.listdir('sharply_value/pointnet'):
    if file.startswith('pointnet_feature1_64players'):
        file_path = os.path.join('sharply_value/pointnet', file)
        sharply = load_pickle(file_path)
        sharply_list.append(sharply)
sharply_value = torch.concat(sharply_list, dim=0)
print(sharply_value.shape)
# with open(os.path.join('sharply_value', save_name), "wb") as f:
#                 pickle.dump(sharply_value, f)
np.savez(os.path.join('sharply_value/pointnet', save_name), sharply_value.cpu().numpy().astype(np.float32))