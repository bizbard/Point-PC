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
from models.pct import PCTransformer
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


# Arguments
parser = argparse.ArgumentParser()
# Configs and ckpts
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--config', type=str, default='./memptc.yaml')
parser.add_argument('--ke_ckpt', type=str, default='./key_encoder.pth')
parser.add_argument('--log_root', type=str, default='./logs_ae')

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='/workspace/docker/point-cloud-completion/data/ShapeNet55-34/shapenet_pc')
parser.add_argument('--split_path', type=str, default='/workspace/docker/point-cloud-completion/data/ShapeNet55-34/ShapeNet-55')
parser.add_argument('--categories', type=str_list, default=['table', 'chair', 'airplane', 'car', 'sofa', 'birdhouse', 'bag', 'remote', 'keyboard', 'rocket'])
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=1)

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
if args.ke_ckpt != None:
    key_encoder = PCTransformer(config=config.key_encoder)
    ckpt = torch.load(args.ke_ckpt)
    base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['ke_encoder'].items()}
    key_encoder.load_state_dict(base_ckpt, strict=True)
    key_encoder = nn.DataParallel(key_encoder).cuda()
base_model = MemPtc(args, config)
base_model = nn.DataParallel(base_model).cuda()
logger.info(repr(key_encoder))
logger.info(repr(base_model))


# optimizer and schedular
if args.ke_ckpt != None:
    ke_solver, ke_scheduler = build_opti_sche(key_encoder, config.ke_opti_sche)
model_solver, model_scheduler = build_opti_sche(base_model, config.ve_opti_sche)


# init memory network
logger.info('init memory network...')
init_encoder = key_encoder
with torch.no_grad():
    feature_list, gt_list = init_memory(train_loader, init_encoder)
memory_keys = feature_list
memory_values = gt_list
# print(memory_keys.shape, memory_values.shape) # (3236, 384) (3236, 8192, 3)
# memory_keys = torch.randn([3236, 384]).detach().numpy()
# memory_values = torch.randn([3236, 8192, 3]).detach().numpy()
memory_ages = np.array([0]*(memory_keys.shape[0]))


# Test Loop
#TODO: still need to be complete, because of batch_size 1, this should be run within few times
def test_ae(args, epoch_idx):
    n_itrs = len(test_loader)
    key_encoder.eval()
    base_model.eval()
    test_losses = AverageMeter()
    num_crop = int(npoints * crop_ratio[args.mode])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    for batch_idx, (taxonomy_ids, data) in enumerate(tqdm(test_loader, desc='Test')):
        # if batch_idx == args.num_val_batches:
        #     break
        taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item() #TODO: make sure batch_size to be 1
        gt = data.cuda()
        partial, _ = seprate_point_cloud(gt, npoints, num_crop, fixed_points=None)
        partial = fps(partial, 2048)

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

        if args.use_causal:
            coarse, recon, recon_2, ca_2 = base_model(shapes, key_feature)
        else:
            coarse, recon = base_model(shapes, key_center, key_feature)

        # test loss
        ful_recon = torch.cat([partial, recon], dim=1) # add the partial points in final results, may decrease the cd distance
        test_loss = loss_func_cdl2(ful_recon, gt)
        test_losses.update(test_loss.item())

        writer.add_mesh('val/partial', partial, global_step=batch_idx)
        writer.add_mesh('val/reconstruction', recon, global_step=batch_idx)

        # test metrics: CDl1, CDl2, f1-score
        _metrics = Metrics.get(recon, gt)

        if taxonomy_id not in category_metrics:
            category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
        category_metrics[taxonomy_id].update(_metrics)

        input_pc = partial.squeeze().cpu().detach().numpy()
        input_pc = get_ptcloud_img(input_pc)
        writer.add_image('Model%02d/Input' % batch_idx, input_pc, epoch_idx, dataformats='HWC')

        dense = recon.squeeze().cpu().detach().numpy()
        dense_img = get_ptcloud_img(dense)
        writer.add_image('Model%02d/Dense' % batch_idx, dense_img, epoch_idx, dataformats='HWC')

        # TODO: make sure batch_size to be 1
        logger.info('[Test] [%d/%d] Taxonomy = %s Losses = %.6f Metrics = %s' % (batch_idx + 1, n_itrs, taxonomy_id, test_loss.item(), ['%.4f' % m for m in _metrics]))

    # three metrics across all categories
    for _, v in category_metrics.items():
        test_metrics.update(v.avg())
    logger.info('[Test] EPOCH: %d  Metrics = %s' % (epoch_idx, ['%.4f' % m for m in test_metrics.avg()]))

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
for epoch_idx in range(0, args.epoches):
    key_encoder.train()
    base_model.train()
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

        key_start = time.time()
        partial, _ = seprate_point_cloud(gt, npoints, [int(npoints * 1 / 4), int(npoints * 3 / 4)], fixed_points=None)
        key_center, key_feature, cdist_feature = key_encoder(partial)
        key_end = time.time()
        key_time = key_end - key_start

        memory_start = time.time()
        shapes = search_value(args, cdist_feature, gt, memory_keys, memory_values, memory_ages)
        memory_end = time.time()
        memory_time = memory_end - memory_start

        refine_start = time.time()
        if args.use_causal:
            coarse, recon_1, recon_2, ca_2 = base_model(shapes, key_feature)
            loss_1 = loss_func_cdl1(coarse, gt)
            loss_2 = loss_func_cdl1(recon_1, gt)
            loss_3 = loss_func_cdl1(recon_2, gt)
            loss_4 = F.kl_div(F.log_softmax(ca_2, dim=2), torch.ones_like(ca_2, dtype=torch.float).to(ca_2.device), reduction='batchmean')*1e-5
            loss = loss_1 + loss_2 + loss_3 + loss_4
            loss.backward()
        else:
            coarse, recon = base_model(shapes, key_center, key_feature)
            loss_1 = loss_func_cdl1(coarse, gt)
            loss_2 = loss_func_cdl1(recon, gt)
            loss = loss_1 + loss_2
            loss.backward()
        refine_end = time.time()
        refine_time = refine_end - refine_start

        ke_solver.step()
        model_solver.step()
        key_encoder.zero_grad()
        base_model.zero_grad()

        losses.update(loss.item())
        itr = epoch_idx * n_batches + batch_idx
        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()
        logger.info('[Train] Iter %04d | BatchTime = %.3f (s) | DataTime = %.3f (s) | Loss %.6f' % (itr, batch_time.val(), data_time.val(), loss.item()))
        logger.info('[Train] Iter %04d | key_time = %.3f (s) | memory_time = %.3f (s) | refine_time = %.3f (s)' % (itr, key_time, memory_time, refine_time))
        writer.add_scalar('train/BatchLoss', loss.item(), itr)

        # if batch_idx == 10:
        #     break

    if isinstance(ke_scheduler, list):
        for item in ke_scheduler:
            item.step(epoch_idx)
    else:
        ke_scheduler.step(epoch_idx)
    if isinstance(model_scheduler, list):
        for item in model_scheduler:
            item.step(epoch_idx)
    else:
        model_scheduler.step(epoch_idx)
    epoch_end_time = time.time()
    writer.add_scalar('train/EpochLoss', losses.avg(), epoch_idx)
    logger.info('[Train] EPOCH: %d EpochTime = %.3f | EpochLoss %.6f' % (epoch_idx, epoch_end_time - epoch_start_time, losses.avg()))
    # for name, param in encoder.named_parameters():
    #     writer.add_histogram(name + '_grad', param.grad, epoch_idx)
    #     writer.add_histogram(name + '_data', param, epoch_idx)

    # save the checkpoint
    if writer is not None and (epoch_idx + 1) % 50 == 0:
        if (epoch_idx + 1) % args.val_freq == 0:
            metric = test_ae(args, epoch_idx)
            file_name = 'checkpoint-epoch-%03d-%04f.pth' % (epoch_idx + 1, metric)
            if metric < best_metric:
                best_metric = metric
                file_name = 'checkpoint-best-%03d-%04f.pth' % (epoch_idx + 1, best_metric)

            output_path = os.path.join(log_dir, file_name)

            # TODO: should save the memory
            checkpoint = {
                'ke_encoder': key_encoder.state_dict(),
                'base_model': base_model.state_dict(),
                'memory_keys': memory_keys,
                'memory_values': memory_values,
                'memory_ages': memory_ages,
            }

            torch.save(checkpoint, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)


writer.close()
logger.info('Terminating...')
