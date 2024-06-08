import os
import sys
import time

import argparse
import json
import yaml
from easydict import EasyDict
import torch.utils.tensorboard

from utils.dataset import *
from utils.misc import *
from utils.optim import *
from utils.memory import *
from utils.metrics import *
from models.memptc_bak import MemPtc
from models.pct import PCTransformer
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from utils.nt_xent import NTXentLoss


# Arguments
parser = argparse.ArgumentParser()
# Configs and ckpts
parser.add_argument('--resume', type=str, default='./logs_ae/AE_2022_12_24__02_45_00_/checkpoint-best-190-0.016787.pth')
parser.add_argument('--config', type=str, default='./memptc_pcn.yaml')
parser.add_argument('--ke_ckpt', type=str, default=None)
parser.add_argument('--ve_ckpt', type=str, default=None)
parser.add_argument('--log_root', type=str, default='./logs_ae')

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/ShapeNet55-34/shapenet_pc')
parser.add_argument('--split_path', type=str, default='./data/ShapeNet55-34/ShapeNet-55')
parser.add_argument('--categories', type=str_list, default=['table'])
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=1)

# Training
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--use_causal', type=str, default=False)
parser.add_argument('--mode', type=str, default='easy')
parser.add_argument('--thresh', type=float, default=0.0015)
parser.add_argument('--topk', type=int, default=3)
parser.add_argument('--val_freq', type=float, default=10)
parser.add_argument('--epoches', type=int, default=300)
parser.add_argument('--num_val_batches', type=int, default=10)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=4)
args = parser.parse_args()
seed_all(args.seed)

# Logging
log_dir = get_new_log_dir(args.log_root, prefix='AE_', postfix='_')
logger = get_logger('train', log_dir)
writer = torch.utils.tensorboard.SummaryWriter(log_dir)
logger.info(args)

# Datasets and loaders
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = EasyDict(config)
logger.info('Loading datasets...')
train_dset = PCNCore(config.dataset, subset='train')
test_dset = PCNCore(config.dataset, subset='test')

train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.train_batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.test_batch_size, num_workers=0)


# Model
logger.info('Building model...')
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = EasyDict(config)
ckpt = torch.load(args.resume)
if args.resume != None:
    key_encoder = PCTransformer(config=config.value_encoder)
    base_model = MemPtc(args, config)
    ke_ckpt = {k.replace("module.", ""): v for k, v in ckpt['ke_encoder'].items()}
    key_encoder.load_state_dict(ke_ckpt, strict=True)
    key_encoder = nn.DataParallel(key_encoder).cuda()
    base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
    base_model.load_state_dict(base_ckpt, strict=True)
    base_model = nn.DataParallel(base_model).cuda()
logger.info(repr(key_encoder))
logger.info(repr(base_model))


# optimizer and schedular
# ke_solver, ke_scheduler = build_opti_sche(key_encoder, config.ke_opti_sche)
# model_solver, model_scheduler = build_opti_sche(base_model, config.ve_opti_sche)


# init memory network
logger.info('init memory network...')
memory_keys = ckpt['memory_keys']
memory_values = ckpt['memory_values']
memory_ages = ckpt['memory_ages']
# mem = init_memory(train_loader, args, config)
# with torch.no_grad():
#     feature_list, gt_list = init_memory(train_loader, value_encoder)
# memory_keys = mem.feature_list
# memory_values = mem.gt_list
# print(memory_keys.shape, memory_values.shape) # (3236, 384) (3236, 8192, 3)
# memory_keys = torch.randn([3236, 384]).detach().numpy()
# memory_values = torch.randn([3236, 8192, 3]).detach().numpy()
# memory_ages = np.array([0]*(memory_keys.shape[0]))


# Training loop
logger.info('Start testing...')
npoints = 8192
crop_ratio = {
    'easy': 1/4,
    'median': 1/2,
    'hard': 3/4
}
best_metric = 100
loss_func_cdl1 = ChamferDistanceL1()
loss_func_cdl2 = ChamferDistanceL2()
# Test Loop
with torch.no_grad():
    n_itrs = len(test_loader)
    key_encoder.eval()
    base_model.eval()
    test_losses = AverageMeter()
    num_crop = int(npoints * crop_ratio[args.mode])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_loader, desc='Test')):
        # if batch_idx == args.num_val_batches:
        #     break
        taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item() #TODO: make sure batch_size to be 1
        # gt = data.cuda()
        # partial, _ = seprate_point_cloud(gt, npoints, num_crop, fixed_points=None)
        partial = data[0].cuda()
        gt = data[1].cuda()

        key_center, key_feature, cdist_feature = key_encoder(partial)

        """ memory network """
        # compute cosine similarity
        shapes = []
        s_k = cdist(cdist_feature.cpu().detach().numpy(), memory_keys, metric='cosine')
        for i in range(cdist_feature.size(0)):
            # find the topk key
            near_ids = np.argsort(s_k[i]).tolist()[:args.topk]
            # find the topk value
            near_shapes = []
            for i in range(len(near_ids)):
                near_shape = torch.from_numpy(memory_values[near_ids[i]]).cuda()
                near_shapes.append(near_shape)
            near_shapes = torch.stack(near_shapes, dim=0)
            shapes.append(near_shapes)
        shapes = torch.stack(shapes, dim=0)

        coarse, recon = base_model(shapes, key_center, key_feature)

        # test loss
        recon = fps(recon, 14336)
        dense = torch.cat([partial, recon], dim=1)  # add the partial points in final results, may decrease the cd distance
        test_loss = loss_func_cdl2(dense, gt)
        test_losses.update(test_loss.item())

        writer.add_mesh('Test/partial', partial, global_step=batch_idx)
        writer.add_mesh('Test/reconstruction', dense, global_step=batch_idx)
        writer.add_mesh('Test/groundtruth', gt, global_step=batch_idx)

        # test metrics: CDl1, CDl2, f1-score
        _metrics = Metrics.get(dense, gt)

        if taxonomy_id not in category_metrics:
            category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
        category_metrics[taxonomy_id].update(_metrics)

        # input_pc = partial.squeeze().cpu().detach().numpy()
        # input_pc = get_ptcloud_img(input_pc)
        # writer.add_image('Model%02d/Input' % batch_idx, input_pc, epoch_idx, dataformats='HWC')
        #
        # dense = recon.squeeze().cpu().detach().numpy()
        # dense_img = get_ptcloud_img(dense)
        # writer.add_image('Model%02d/Dense' % batch_idx, dense_img, epoch_idx, dataformats='HWC')


        # os.makedirs(os.path.join(log_dir, 'partial'), exist_ok=True)
        # os.makedirs(os.path.join(log_dir, 'recon'), exist_ok=True)
        # os.makedirs(os.path.join(log_dir, 'gt'), exist_ok=True)
        #
        # partial_output_file = os.path.join(log_dir, 'partial', str(batch_idx)+'.txt')
        # np.savetxt(partial_output_file, partial.squeeze(0).detach().cpu().numpy())
        #
        # recon_output_file = os.path.join(log_dir, 'recon', str(batch_idx) + '.txt')
        # np.savetxt(recon_output_file, dense.squeeze(0).detach().cpu().numpy())
        #
        # gt_output_file = os.path.join(log_dir, 'gt', str(batch_idx) + '.txt')
        # np.savetxt(gt_output_file, gt.squeeze(0).detach().cpu().numpy())

        # TODO: make sure batch_size to be 1
        logger.info('[Test] [%d/%d] Taxonomy = %s Losses = %.6f Metrics = %s' % (batch_idx + 1, n_itrs, taxonomy_id, test_loss.item(), ['%.4f' % m for m in _metrics]))

    # three metrics across all categories
    for _, v in category_metrics.items():
        test_metrics.update(v.avg())
    logger.info('Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]))

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    logger.info('============================ TEST RESULTS ============================')
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    logger.info(msg)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        logger.info(msg)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    logger.info(msg)

    writer.add_scalar('val/EpochLoss', test_losses.avg(), 0)
    writer.flush()
    print(test_losses.avg()) # 0.0014324504472933582

writer.close()
logger.info('Terminating...')
