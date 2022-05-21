from __future__ import print_function, absolute_import
import os
import sys
import time
import random
import datetime
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
from models import init_model
from utils.losses import TripletLoss, InfoNce
from utils.utils import AverageMeter, Logger, save_checkpoint, print_time
from utils.eval_metrics import evaluate
from utils.samplers import RandomIdentitySampler
from utils import data_manager
from utils.video_loader import VideoDataset, VideoDatasetInfer


parser = argparse.ArgumentParser(description='Train video model')
# Datasets
parser.add_argument('--root', type=str, default='/home/guxinqian/data/')
parser.add_argument('-d', '--dataset', type=str, default='lsvid',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
# Augment
parser.add_argument('--sample_stride', type=int, default=8, help="stride of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max_epoch', default=160, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start_epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train_batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test_batch', default=32, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=[40, 80, 120], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight_decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--distance', type=str, default='cosine', help="euclidean or consine")
parser.add_argument('--num_instances', type=int, default=4, help="number of instances per identity")
parser.add_argument('--losses', default=['xent', 'htri'], nargs='+', type=str, help="losses")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='c2resnet50', help="c2resnet50, nonlocalresnet50")
parser.add_argument('--pretrain', action='store_true', help="load params form pretrain model on kinetics")
parser.add_argument('--pretrain_model_path', type=str, default='', metavar='PATH')
# Miscs
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval_step', type=int, default=10,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start_eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save_dir', '--sd', type=str, default='')
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu_devices', default='2,3', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

parser.add_argument('--all_frames', action='store_true', help="evaluate with all frames ?")
parser.add_argument('--seq_len', type=int, default=4,
                    help="number of images to sample in a tracklet")
parser.add_argument('--note', type=str, default='', help='additional description of this command')
args = parser.parse_args()

def specific_params(args):
    if args.arch in ['sinet', 'sbnet']:
        args.losses = ['xent', 'htri', 'infonce']

def main():
    # fix the seed in random operation
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.log'))
    elif args.all_frames:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_eval_all_frames.log'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_eval_sampled_frames.log'))

    print_time("============ Initialized logger ============")
    print("\n".join("\t\t%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    print_time("============ Description ============")
    print_time("\t\t %s\n" % args.note)

    print_time("The experiment will be stored in %s\n" % args.save_dir)

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed) 
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("Currently using CPU (GPU is highly recommended)")


    print_time("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset, root=args.root)

    # Data augmentation
    spatial_transform_train = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),
                ST.RandomHorizontalFlip(),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ST.RandomErasing()])

    spatial_transform_test = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    temporal_transform_train = TT.TemporalRestrictedCrop(size=args.seq_len)
    temporal_transform_test = TT.TemporalRestrictedBeginCrop(size=args.seq_len)

    dataset_train = dataset.train
    dataset_query = dataset.query
    dataset_gallery = dataset.gallery

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        VideoDataset(
            dataset_train,
            spatial_transform=spatial_transform_train,
            temporal_transform=temporal_transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,)

    queryloader_sampled_frames = DataLoader(
        VideoDataset(dataset_query, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False)

    galleryloader_sampled_frames = DataLoader(
        VideoDataset(dataset_gallery, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False)

    queryloader_all_frames = DataLoader(
        VideoDatasetInfer(
            dataset_query, spatial_transform=spatial_transform_test, seq_len=args.seq_len),
        batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False)

    galleryloader_all_frames = DataLoader(
        VideoDatasetInfer(dataset_gallery, spatial_transform=spatial_transform_test, seq_len=args.seq_len),
        batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False)

    print_time("Initializing model: {}".format(args.arch))
    model = init_model(
                name=args.arch,
                num_classes = dataset.num_train_pids,
                losses=args.losses,
                seq_len=args.seq_len)

    print_time("Model Size w/o Classifier: {:.5f}M".format(
        sum(p.numel() for name, p in model.named_parameters() if 'classifier' not in name and 'projection' not in name)/1000000.0))

    criterions = {
        'xent': nn.CrossEntropyLoss(),
        'htri': TripletLoss(margin=args.margin, distance=args.distance),
        'infonce': InfoNce(num_instance=args.num_instances)}

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.pretrain:
        print("Loading pre-trained params from '{}'".format(args.pretrain_model_path))
        pretrain_dict = torch.load(args.pretrain_model_path)
        model_dict = model.state_dict()
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)

    if args.resume:
        print_time("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        with torch.no_grad():
            if args.all_frames:
                print_time('==> Evaluate with [all] frames!')
                test(model, queryloader_all_frames, galleryloader_all_frames, use_gpu)
            else:
                print_time('==> Evaluate with sampled [{}] frames per video!'.format(args.seq_len))
                test(model, queryloader_sampled_frames, galleryloader_sampled_frames, use_gpu)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print_time("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterions, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)
        scheduler.step()

        if (epoch+1) >= args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print_time("==> Test")
            with torch.no_grad():
                rank1 = test(model, queryloader_sampled_frames, galleryloader_sampled_frames, use_gpu)

            is_best = rank1 > best_rank1
            if is_best: 
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu: state_dict = model.module.state_dict()
            else: state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print_time("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print_time('=='*50)
    print_time("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

    # using all frames to evaluate the final performance after training
    args.all_frames = True

    infer_epochs = [150]
    if best_epoch !=150: infer_epochs.append(best_epoch)

    for epoch in infer_epochs:
        best_checkpoint_path = osp.join(args.save_dir, 'checkpoint_ep' + str(epoch) + '.pth.tar')
        checkpoint = torch.load(best_checkpoint_path)
        model.module.load_state_dict(checkpoint['state_dict'])

        print_time('==> Evaluate with all frames!')
        print_time("Loading checkpoint from '{}'".format(best_checkpoint_path))
        with torch.no_grad():
            test(model, queryloader_all_frames, galleryloader_all_frames, use_gpu)
        return

def train(epoch, model, criterions, optimizer, trainloader, use_gpu):
    batch_xent_loss = AverageMeter()
    batch_htri_loss = AverageMeter()
    batch_info_loss = AverageMeter()
    batch_loss = AverageMeter()
    batch_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    pd = tqdm(total=len(trainloader), ncols=120, leave=False)
    for batch_idx, (vids, pids, camid) in enumerate(trainloader):
        pd.set_postfix({'Acc': '{:>7.2%}'.format(batch_corrects.avg), })
        pd.update(1)
        if (pids-pids[0]).sum() == 0:
            continue

        if use_gpu:
            vids = vids.cuda()
            pids = pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # zero the parameter gradients
        optimizer.zero_grad()

        if 'infonce' in args.losses:
            y, f, x = model(vids)
            # combine hard triplet loss with cross entropy loss
            xent_loss = criterions['xent'](y, pids)
            htri_loss = criterions['htri'](f, pids)
            info_loss = criterions['infonce'](x)
            loss = xent_loss + htri_loss + 0.001 * info_loss
        else:
            y, f = model(vids)
            # combine hard triplet loss with cross entropy loss
            xent_loss = criterions['xent'](y, pids)
            htri_loss = criterions['htri'](f, pids)
            loss = xent_loss + htri_loss
            info_loss = htri_loss * 0

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        _, preds = torch.max(y.data, 1)
        batch_corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_xent_loss.update(xent_loss.item(), pids.size(0))
        batch_htri_loss.update(htri_loss.item(), pids.size(0))
        batch_info_loss.update(info_loss.item(), pids.size(0))
        batch_loss.update(loss.item(), pids.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    pd.close()

    print_time('Epoch{:>4d} '
          'Time:{batch_time.sum:>5.1f}s '
          'Data:{data_time.sum:>4.1f}s | '
          'Loss:{loss.avg:>7.4f} '
          'xent:{ce.avg:>7.4f} '
          'htri:{ht.avg:>7.4f} | '
          'infonce:{info.avg:>7.4f} | '
          'Acc:{acc.avg:>7.2%} '.format(
            epoch+1, batch_time=batch_time,
            data_time=data_time, loss=batch_loss,
            ce=batch_xent_loss, ht=batch_htri_loss,
            info=batch_info_loss, acc=batch_corrects,))

def _cal_dist(qf, gf, distance='cosine'):
    """
    :param logger:
    :param qf:  (query_num, feat_dim)
    :param gf:  (gallery_num, feat_dim)
    :param distance:
         cosine
    :return:
        distance matrix with shape, (query_num, gallery_num)
    """
    if distance == 'cosine':
        qf = F.normalize(qf, dim=1, p=2)
        gf = F.normalize(gf, dim=1, p=2)
        distmat = -torch.matmul(qf, gf.transpose(0, 1))
    else:
        raise NotImplementedError
    return distmat

def extract_feat_sampled_frames(model, vids, use_gpu=True):
    """
    :param model:
    :param vids: (b, 3, t, 256, 128)
    :param use_gpu:
    :return:
        features: (b, c)
    """
    if use_gpu: vids = vids.cuda()
    f = model(vids)    # (b, t, c)
    f = f.mean(-1)
    f = f.data.cpu()
    return f

def extract_feat_all_frames(model, vids, max_clip_per_batch=45, use_gpu=True):
    """
    :param model:
    :param vids:    (_, b, c, t, h, w)
    :param max_clip_per_batch:
    :param use_gpu:
    :return:
        f, (1, C)
    """
    if use_gpu:
        vids = vids.cuda()
    _, b, c, t, h, w = vids.size()
    vids = vids.reshape(b, c, t, h, w)

    if max_clip_per_batch is not None and b > max_clip_per_batch:
        feat_set = []
        for i in range((b - 1) // max_clip_per_batch + 1):
            clip = vids[i * max_clip_per_batch: (i + 1) * max_clip_per_batch]
            f = model(clip)  # (max_clip_per_batch, t, c)
            f = f.mean(-1)
            feat_set.append(f)
        f = torch.cat(feat_set, dim=0)
    else:
        f = model(vids) # (b, t, c)
        f = f.mean(-1)   # (b, c)

    f = f.mean(0, keepdim=True)
    f = f.data.cpu()
    return f

def _feats_of_loader(model, loader, feat_func=extract_feat_sampled_frames, use_gpu=True):
    qf, q_pids, q_camids = [], [], []

    pd = tqdm(total=len(loader), ncols=120, leave=False)
    for batch_idx, (vids, pids, camids) in enumerate(loader):
        pd.update(1)

        f = feat_func(model, vids, use_gpu=use_gpu)
        qf.append(f)
        q_pids.extend(pids.numpy())
        q_camids.extend(camids.numpy())
    pd.close()

    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    return qf, q_pids, q_camids

def _eval_format_logger(cmc, mAP, ranks, desc=''):
    print_time("Results {}".format(desc))
    ptr = "mAP: {:.2%}".format(mAP)
    for r in ranks:
        ptr += " | R-{:<3}: {:.2%}".format(r, cmc[r - 1])
    print_time(ptr)
    print_time("--------------------------------------")

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    since = time.time()
    model.eval()

    if args.all_frames:
        feat_func = extract_feat_all_frames
    else:
        feat_func = extract_feat_sampled_frames

    qf, q_pids, q_camids = _feats_of_loader(
        model,
        queryloader,
        feat_func,
        use_gpu=use_gpu)
    print_time("Extracted features for query set, obtained {} matrix".format(qf.shape))

    gf, g_pids, g_camids = _feats_of_loader(
        model,
        galleryloader,
        feat_func,
        use_gpu=use_gpu)
    print_time("Extracted features for gallery set, obtained {} matrix".format(gf.shape))

    if args.dataset == 'mars':
        # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
        gf = torch.cat((qf, gf), 0)
        g_pids = np.append(q_pids, g_pids)
        g_camids = np.append(q_camids, g_camids)

    time_elapsed = time.time() - since
    print_time('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print_time("Computing distance matrix")
    distmat = _cal_dist(qf=qf, gf=gf, distance=args.distance)
    print_time("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    _eval_format_logger(cmc, mAP, ranks, '')

    return cmc[0]


if __name__ == '__main__':
    specific_params(args)
    main()
