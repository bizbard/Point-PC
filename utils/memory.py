import os
import h5py
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from utils.misc import seprate_point_cloud
from scipy.spatial.distance import cdist
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from models.pct import PCTransformer

npoints = 8192
class init_memory():
    def __init__(self, train_loader, args, config):
        super().__init__()
        self.train_loader = train_loader
        self.args = args
        self.config = config
        self.get_memory()

    def get_memory(self):
        save_path = os.path.join(self.args.log_root, 'all-cates' + '.hdf5')
        if os.path.exists(save_path):
            print('loading memory from existing file...')
            with h5py.File(save_path, 'r') as f:
                self.feature_list = f['key'][...]
                self.gt_list = f['value'][...]
            return self.feature_list, self.gt_list

        if self.args.ve_ckpt != None:
            value_encoder = PCTransformer(config=self.config.value_encoder)
            ckpt = torch.load(self.args.ve_ckpt)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['ve_encoder'].items()}
            value_encoder.load_state_dict(base_ckpt, strict=True)
            value_encoder = nn.DataParallel(value_encoder).cuda()

        feature_list, gt_list = [], []
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(self.train_loader, desc='Train')):
            gt = data[1].cuda()

            _, _, global_feature = value_encoder(gt)
            feature_list.append(global_feature.cpu().detach().numpy())
            gt_list.append(gt.cpu().detach().numpy())

        self.feature_list = np.concatenate(feature_list, axis=0)
        self.gt_list = np.concatenate(gt_list, axis=0)
        with h5py.File(save_path, 'a') as f:
            keys = f.create_dataset('key', data=self.feature_list)
            values = f.create_dataset('value', data=self.gt_list)
        return self.feature_list, self.gt_list


# memory network logit
def update_feature(F, K):
    F, K = F.unsqueeze(0), K.unsqueeze(0)
    K = F + K
    m = torch.norm(K)
    K = K / m
    return K.squeeze(0).cpu().detach().numpy()

def update_age(ages, id):
    new_ages = np.array([i + 1 for i in ages])
    new_ages[id] = 0
    return new_ages

def search_value(args, feature, gt, memory_keys, memory_values, memory_ages):
    """ memory network """
    """ it seems the memory is fixed length, and should be ready before a epoch """
    # compute cosine similarity
    shapes = []
    loss_func_cdl2 = ChamferDistanceL2()
    s_k = cdist(feature.cpu().detach().numpy(), memory_keys, metric='cosine')
    for i in range(feature.size(0)):
        # find the topk key
        near_ids = np.argsort(s_k[i]).tolist()[:args.topk]
        # find the topk value
        near_shapes = []
        for i in range(len(near_ids)):
            near_shape = torch.from_numpy(memory_values[near_ids[i]]).cuda()
            near_shapes.append(near_shape)
        near_shapes = torch.stack(near_shapes, dim=0)
        shapes.append(near_shapes)
        # find the nearest value
        nearest_id = np.argsort(s_k[i]).tolist()[0]
        nearest_gt = torch.from_numpy(memory_values[nearest_id]).cuda()
        # compute the chamfer distance between the in and out shape
        s_v = loss_func_cdl2(gt[i].unsqueeze(0), nearest_gt.unsqueeze(0))
        if s_v <= args.thresh:
            # do not change the shape, update the feature
            memory_keys[nearest_id] = update_feature(feature[i], torch.from_numpy(memory_keys[nearest_id]).cuda())
            # update the age
            memory_ages = update_age(memory_ages, nearest_id)

        else:
            oldest_id = memory_ages.argmax()
            memory_keys[oldest_id] = feature[i].cpu().detach().numpy()
            memory_values[oldest_id] = gt[i].cpu().detach().numpy()
            memory_ages = update_age(memory_ages, oldest_id)

    shapes = torch.stack(shapes, dim=0) # b, k, 8192, 3
    return shapes


def search_value_v2(args, feature, memory_keys, memory_values):
    """ memory network """
    """ it seems the memory is fixed length, and should be ready before a epoch """
    # compute cosine similarity
    shapes = []
    s_k = cdist(feature.cpu().detach().numpy(), memory_keys, metric='cosine')
    for i in range(feature.size(0)):
        # find the topk key
        near_ids = np.argsort(s_k[i]).tolist()[:args.topk]
        # find the topk value
        near_shapes = []
        for i in range(len(near_ids)):
            near_shape = torch.from_numpy(memory_values[near_ids[i]]).cuda()
            near_shapes.append(near_shape)
        near_shapes = torch.stack(near_shapes, dim=0)
        shapes.append(near_shapes)

    shapes = torch.stack(shapes, dim=0) # b, k, 8192, 3
    return shapes