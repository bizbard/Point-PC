import sys

import torch
from utils.dataset import MemoryCore
from models.pointnet import PointNetEncoder
from tqdm.auto import tqdm
import h5py
import numpy as np
from models.pct import PCTransformer

train_dset = MemoryCore(
    path='./data/ShapeNet55_mem3d.hdf5',
    split='train',
)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

ptn_ckpt = torch.load('./dvae_encoder.pt')
print(ptn_ckpt.keys())
encoder = PointNetEncoder(encoder_channel=256).cuda()
encoder.load_state_dict(ptn_ckpt['state_dict'])
# encoder = PCTransformer(in_chans=3, embed_dim=384, depth=[6, 8], drop_rate=0., num_query=96, knn_layer=1).cuda()

feature_list, gt_list = [], []
for batch_idx, batch in enumerate(tqdm(train_loader, desc='Train')):
    partial = batch['partial'].cuda() # b, 2048, 3
    gt = batch['pointcloud'].cuda() # b, 2048, 3

    global_feature = encoder(partial)
    feature_list.append(global_feature.cpu().detach().numpy())
    gt_list.append(gt.cpu().detach().numpy())

feature_list = np.concatenate(feature_list, axis=0)
gt_list = np.concatenate(gt_list, axis=0)
print(gt_list.shape[0])

dst_name = 'memory_init.hdf5'
with h5py.File(dst_name, 'a') as f:
    keys = f.create_dataset('keys', data=feature_list)
    values = f.create_dataset('values', data=gt_list)