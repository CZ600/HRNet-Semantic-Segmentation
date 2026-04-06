import argparse
import os
import sys
import time
import logging
import numpy as np
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tensorboardX import SummaryWriter
from tqdm import tqdm

import _init_paths
import models
import datasets
from config import config, update_config
from core.losses import FocalDiceLoss
from utils.utils import AverageMeter, adjust_learning_rate


def parse_args():
    parser = argparse.ArgumentParser(description='Train road segmentation')
    parser.add_argument('--cfg', required=True, type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--focal_weight', type=float, default=1.0)
    parser.add_argument('--dice_weight', type=float, default=1.0)
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--resume_from', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('opts', help="Modify config options", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    return args


def build_model():
    import importlib
    models_module = importlib.import_module('models.' + config.MODEL.NAME)
    model_fn = getattr(models_module, 'get_seg_model')
    model = model_fn(config)
    return model


def build_dataloader(phase='train'):
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])

    if phase == 'train':
        dataset = eval('datasets.' + config.DATASET.DATASET)(
            root=config.DATASET.ROOT,
            list_path=config.DATASET.TRAIN_SET,
            num_classes=config.DATASET.NUM_CLASSES,
            multi_scale=config.TRAIN.MULTI_SCALE,
            flip=config.TRAIN.FLIP,
            ignore_label=config.TRAIN.IGNORE_LABEL,
            base_size=config.TRAIN.BASE_SIZE,
            crop_size=crop_size,
            downsample_rate=config.TRAIN.DOWNSAMPLERATE,
            scale_factor=config.TRAIN.SCALE_FACTOR,
        )
        loader = DataLoader(dataset, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
                            shuffle=True, num_workers=config.WORKERS,
                            pin_memory=True, drop_last=True)
    else:
        dataset = eval('datasets.' + config.DATASET.DATASET)(
            root=config.DATASET.ROOT,
            list_path=config.DATASET.TEST_SET,
            num_classes=config.DATASET.NUM_CLASSES,
            multi_scale=False,
            flip=False,
            ignore_label=config.TRAIN.IGNORE_LABEL,
            base_size=config.TEST.BASE_SIZE,
            crop_size=test_size,
            downsample_rate=1,
        )
        loader = DataLoader(dataset, batch_size=config.TEST.BATCH_SIZE_PER_GPU,
                            shuffle=False, num_workers=config.WORKERS,
                            pin_memory=True)
    return dataset, loader


def train_one_epoch(model, criterion, optimizer, trainloader, epoch, args, writer, global_step):
    model.train()
    ave_loss = AverageMeter()

    pbar = tqdm(trainloader, total=len(trainloader),
                desc=f'Epoch {epoch}/{config.TRAIN.END_EPOCH}',
                unit='batch', leave=True)

    for i_iter, batch in enumerate(pbar):
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            loss = criterion(outputs, labels)
        else:
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ave_loss.update(loss.item())

        lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{ave_loss.average():.4f}',
            'lr': f'{lr:.6f}'
        })

    writer.add_scalar('train_loss', ave_loss.average(), global_step)
    return ave_loss.average()


def validate(model, criterion, testloader, epoch, writer, global_step):
    model.eval()
    ave_loss = AverageMeter()
    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0

    pbar = tqdm(testloader, total=len(testloader),
                desc=f'Validating Epoch {epoch}',
                unit='batch', leave=True)

    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            images, labels, _, _ = batch
            images = images.cuda()
            labels = labels.long().cuda()

            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                loss = criterion(outputs, labels)
                pred = outputs[-1] if len(outputs) > 1 else outputs[0]
            else:
                loss = criterion(outputs, labels)
                pred = outputs

            ave_loss.update(loss.item())

            ph, pw = pred.size(2), pred.size(3)
            h, w = labels.size(1), labels.size(2)
            if ph != h or pw != w:
                pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

            pred_labels = pred.argmax(dim=1).cpu().numpy()
            gt_labels = labels.cpu().numpy()

            pred_binary = (pred_labels == 1).astype(np.int64)
            gt_binary = (gt_labels == 1).astype(np.int64)

            tp_total += (pred_binary * gt_binary).sum()
            fp_total += (pred_binary * (1 - gt_binary)).sum()
            fn_total += ((1 - pred_binary) * gt_binary).sum()
            tn_total += ((1 - pred_binary) * (1 - gt_binary)).sum()

            eps = 1e-7
            current_tp = (pred_binary * gt_binary).sum()
            current_fp = (pred_binary * (1 - gt_binary)).sum()
            current_fn = ((1 - pred_binary) * gt_binary).sum()
            current_prec = current_tp / (current_tp + current_fp + eps)
            current_recall = current_tp / (current_tp + current_fn + eps)
            current_iou = current_tp / (current_tp + current_fp + current_fn + eps)
            pbar.set_postfix({
                'loss': f'{ave_loss.average():.4f}',
                'prec': f'{current_prec:.4f}',
                'recall': f'{current_recall:.4f}',
                'iou': f'{current_iou:.4f}'
            })

    eps = 1e-7
    precision = tp_total / (tp_total + fp_total + eps)
    recall = tp_total / (tp_total + fn_total + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp_total / (tp_total + fp_total + fn_total + eps)
    accuracy = (tp_total + tn_total) / (tp_total + fp_total + fn_total + tn_total + eps)

    msg = (f'Val Epoch [{epoch}] Loss: {ave_loss.average():.4f} '
           f'IoU: {iou:.4f} F1: {f1:.4f} '
           f'Prec: {precision:.4f} Recall: {recall:.4f} Acc: {accuracy:.4f}')
    logging.info(msg)
    print(msg)

    writer.add_scalar('val_loss', ave_loss.average(), global_step)
    writer.add_scalar('val_iou', iou, global_step)
    writer.add_scalar('val_f1', f1, global_step)
    writer.add_scalar('val_precision', precision, global_step)
    writer.add_scalar('val_recall', recall, global_step)

    return ave_loss.average(), iou, f1


def main():
    args = parse_args()

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Setup logging
    output_dir = Path(config.OUTPUT_DIR) / config.DATASET.DATASET / \
                 os.path.basename(args.cfg).replace('.yaml', '')
    output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    tb_dir = Path('/root/tf-logs/logs') / f'HRNet_{time_str}'
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))

    logging.info(f'Config: {config}')
    logging.info(f'Args: {args}')

    # Build model
    model = build_model()

    if args.pretrained and os.path.isfile(args.pretrained):
        logging.info(f'Loading pretrained weights from {args.pretrained}')
        pretrained_dict = torch.load(args.pretrained, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logging.info(f'Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained params')

    model = nn.DataParallel(model).cuda()

    # Build criterion
    criterion = FocalDiceLoss(
        focal_weight=args.focal_weight,
        dice_weight=args.dice_weight,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
        ignore_index=config.TRAIN.IGNORE_LABEL,
    )

    # Build dataloaders
    train_dataset, trainloader = build_dataloader('train')
    test_dataset, testloader = build_dataloader('val')
    logging.info(f'Train: {len(train_dataset)} samples, Val: {len(test_dataset)} samples')

    # Build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.TRAIN.LR,
        momentum=config.TRAIN.MOMENTUM,
        weight_decay=config.TRAIN.WD,
        nesterov=config.TRAIN.NESTEROV,
    )

    # Resume
    start_epoch = 0
    best_iou = 0.0
    best_f1 = 0.0
    if args.resume_from and os.path.isfile(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.module.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'best_iou' in checkpoint:
            best_iou = checkpoint['best_iou']
        logging.info(f'Resumed from epoch {start_epoch}, best_iou={best_iou:.4f}')

    # Training loop
    epoch_iters = len(trainloader)
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    global_step = start_epoch

    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        adjust_learning_rate(optimizer, config.TRAIN.LR, num_iters,
                             epoch * epoch_iters)

        train_loss = train_one_epoch(model, criterion, optimizer, trainloader,
                                     epoch, args, writer, global_step)
        global_step += 1

        val_loss, val_iou, val_f1 = validate(model, criterion, testloader,
                                              epoch, writer, global_step)

        ckpt_path = os.path.join(str(output_dir), 'checkpoint.pth.tar')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_iou': best_iou,
            'best_f1': best_f1,
        }, ckpt_path)

        if val_iou > best_iou:
            best_iou = val_iou
            best_f1 = val_f1
            best_path = os.path.join(str(output_dir), 'best_model.pth')
            torch.save(model.module.state_dict(), best_path)
            logging.info(f'=> New best model saved with IoU={best_iou:.4f}, F1={best_f1:.4f}')

        logging.info(f'Epoch {epoch}: Best IoU={best_iou:.4f}, Best F1={best_f1:.4f}')

    final_path = os.path.join(str(output_dir), 'final_model.pth')
    torch.save(model.module.state_dict(), final_path)
    writer.close()
    logging.info(f'Done. Best IoU={best_iou:.4f}, Best F1={best_f1:.4f}')


if __name__ == '__main__':
    main()
