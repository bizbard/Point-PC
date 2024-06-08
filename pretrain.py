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
from models.memptc import MemPtc
from models.autoencoder import AutoEncoder
from models.pct import PCTransformer
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from utils.nt_xent import NTXentLoss


# Arguments
parser = argparse.ArgumentParser()
# Configs and ckpts
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--config', type=str, default='./memptc.yaml')
parser.add_argument('--ke_ckpt', type=str, default=None)
parser.add_argument('--log_root', type=str, default='./logs_ae')

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='/workspace/docker/point-cloud-completion/data/ShapeNet55-34/shapenet_pc')
parser.add_argument('--split_path', type=str, default='/workspace/docker/point-cloud-completion/data/ShapeNet55-34/ShapeNet-55')
parser.add_argument('--categories', type=str_list, default=['all'])
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=32)

# Training
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--use_causal', type=str, default=False)
parser.add_argument('--mode', type=str, default='easy')
parser.add_argument('--thresh', type=float, default=0.0015)
parser.add_argument('--topk', type=int, default=3)
parser.add_argument('--val_freq', type=float, default=1)
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
logger.info('Loading datasets...')
train_dset = ShapeNet55Core(data_path=args.dataset_path, split_path=args.split_path, split='train', cates=args.categories)
test_dset = ShapeNet55Core(data_path=args.dataset_path, split_path=args.split_path, split='test', cates=args.categories)

train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.train_batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.test_batch_size, num_workers=0)


# Model
logger.info('Building model...')
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = EasyDict(config)
model = AutoEncoder(config=config)
model = nn.DataParallel(model).cuda()
logger.info(repr(model))


# optimizer and schedular
model_solver, model_scheduler = build_opti_sche(model, config.ve_opti_sche)


# Test Loop
def test_ae(args, epoch_idx):
    with torch.no_grad():
        n_itrs = len(test_loader)
        model.eval()
        test_losses = AverageMeter()
        for batch_idx, (taxonomy_ids, data) in enumerate(tqdm(test_loader, desc='Test')):
            if batch_idx == args.num_val_batches:
                break
            gt = data.cuda()

            recon = model(gt)

            # test loss
            test_loss = loss_func_cdl2(recon, gt)
            test_losses.update(test_loss.item())

            logger.info('[Test] [%d/%d] Losses = %.6f' % (batch_idx + 1, n_itrs, test_loss.item()))

        writer.add_scalar('val/EpochLoss', test_losses.avg(), epoch_idx)
        writer.flush()
        return test_losses.avg()


# Training loop
logger.info('Start training...')
npoints = 8192
crop_ratio = {
    'easy': 1/4,
    'median': 1/2,
    'hard': 3/4
}
best_metric = 100
loss_func_cdl1 = ChamferDistanceL1()
loss_func_cdl2 = ChamferDistanceL2()
contrast = NTXentLoss(temperature=0.1, use_cosine_similarity=True, alpha_weight=0.25)
for epoch_idx in range(0, args.epoches):
    model.train()
    n_batches = len(train_loader)
    losses = AverageMeter()
    # stopwatch
    epoch_start_time = time.time()
    batch_start_time = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    for batch_idx, (taxonomy_ids, data) in enumerate(tqdm(train_loader, desc='Train')):
        data_time.update(time.time() - batch_start_time)
        gt = data.cuda()

        recon = model(gt)
        loss = loss_func_cdl1(recon, gt)
        loss.backward()
        refine_end = time.time()

        model_solver.step()
        model.zero_grad()

        losses.update(loss.item())
        itr = epoch_idx * n_batches + batch_idx
        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()
        logger.info('[Train] Epoch %04d/%04d | BatchTime = %.3f (s) | DataTime = %.3f (s) | Loss %.6f' % (epoch_idx, args.epoches, batch_time.val(), data_time.val(), loss.item()))
        writer.add_scalar('train/BatchLoss', loss.item(), itr)

        if batch_idx == 10:
            break

    if isinstance(model_scheduler, list):
        for item in model_scheduler:
            item.step(epoch_idx)
    else:
        model_scheduler.step(epoch_idx)
    epoch_end_time = time.time()
    writer.add_scalar('train/EpochLoss', losses.avg(), epoch_idx)
    logger.info('[Train] EPOCH: %d EpochTime = %.3f | EpochLoss %.6f' % (epoch_idx, epoch_end_time - epoch_start_time, losses.avg()))

    # save the checkpoint
    if writer is not None and (epoch_idx + 1) <= 50:
        if (epoch_idx + 1) % args.val_freq == 0:
            metric = test_ae(args, epoch_idx)
            file_name = 'checkpoint-epoch-%03d-%04f.pth' % (epoch_idx + 1, metric)
            if metric < best_metric:
                best_metric = metric
                file_name = 'checkpoint-best-%03d-%04f.pth' % (epoch_idx + 1, best_metric)

            output_path = os.path.join(log_dir, file_name)

            # TODO: should save the memory
            checkpoint = {
                've_encoder': model.module.encoder.state_dict(),
            }

            torch.save(checkpoint, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

writer.close()
logger.info('Terminating...')