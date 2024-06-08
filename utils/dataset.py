import os
import random
import sys
from copy import copy
import torch
from einops import rearrange
from torch.utils.data import Dataset
import numpy as np
import json
import h5py
from tqdm.auto import tqdm
# from pointnet2_ops import pointnet2_utils
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
import pointnet2_utils
from models.pointnet import PointNetEncoder
from utils import data_transforms
import open3d


synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}
CLASS_NUM = [i for i in range(0, 55)]
CLASS_DICT = dict(zip([k for k, v in synsetid_to_cate.items()], CLASS_NUM))

class ShapeNet55Core(Dataset):

    def __init__(self, data_path, split_path, split, cates):
        super().__init__()
        self.data_path = data_path
        self.split_path = split_path
        self.split = split
        self.data_list_file = os.path.join(self.split_path, f'{self.split}.txt')
        if 'all' in cates:
            cates = cate_to_synsetid.keys()
        self.cate_synsetids = [cate_to_synsetid[s] for s in cates]

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            if taxonomy_id in self.cate_synsetids:
                # model_id = line.split('-')[1].split('.')[0]
                self.file_list.append({
                    'taxonomy_id': taxonomy_id,
                    'label': CLASS_DICT[taxonomy_id],
                    # 'model_id': model_id,
                    'file_path': line,
                })
        print(f'[DATASET] {len(self.file_list)} instances of {len(self.cate_synsetids)} categories were loaded')


    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc


    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = np.load(os.path.join(self.data_path, sample['file_path'])).astype(np.float32)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()

        return sample['taxonomy_id'], data, sample['label']


    def __len__(self):
        return len(self.file_list)


class PCNCore(Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config, subset, cate):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        # self.subset = config.subset
        self.subset = subset
        self.cate = cate
        self.cars = config.CARS

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] in self.cate]
        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in samples:
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_path': [
                        self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path':
                    self.complete_points_path % (subset, dc['taxonomy_id'], s),
                })

        print('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            # data[ri] = IO.get(file_path).astype(np.float32)
            pc = open3d.io.read_point_cloud(file_path)
            data[ri] = np.array(pc.points).astype(np.float32)

        assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)


class KITTICore(Dataset):
    def __init__(self, config):
        self.cloud_path = config.CLOUD_PATH
        self.bbox_path = config.BBOX_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        assert self.subset == 'test'

        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
        self.transforms = data_transforms.Compose([{
            'callback': 'NormalizeObjectPose',
            'parameters': {
                'input_keys': {
                    'ptcloud': 'partial_cloud',
                    'bbox': 'bounding_box'
                }
            },
            'objects': ['partial_cloud', 'bounding_box']
        }, {
            'callback': 'RandomSamplePoints',
            'parameters': {
                'n_points': 2048
            },
            'objects': ['partial_cloud']
        }, {
            'callback': 'ToTensor',
            'objects': ['partial_cloud', 'bounding_box']
        }])
        self.file_list = self._get_file_list(self.subset)

    def _get_file_list(self, subset):
        """Prepare file list for the dataset"""
        file_list = []
        for dc in self.dataset_categories:
            samples = dc[subset]
            for s in samples:
                file_list.append({
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_cloud_path': self.cloud_path % s,
                    'bounding_box_path': self.bbox_path % s,
                })
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        for ri in ['partial_cloud', 'bounding_box']:
            file_path = sample['%s_path' % ri]
            # data[ri] = IO.get(file_path).astype(np.float32)
            pc = open3d.io.read_point_cloud(file_path)
            data[ri] = np.array(pc.points).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data['partial_cloud']