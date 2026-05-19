import argparse
import json
import os
import pickle
import random
import time
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

import models as models
from data.maskcov_dataset import MaskCOVConfig, build_maskcov_quant_loaders
from models.maskcov import load_maskcov_state_dict
from models.quan import Quantization
from quant import ofwa, ofwa_rr, ofwa_rr_dw


model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith('__') and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description='BitSplit PTQ for MaskCOV')
parser.add_argument('--dataset', default='soybean_gene', type=str,
                    help='MaskCOV dataset name, e.g. COTTON, Soybean200, soybean_gene')
parser.add_argument('--data-root', default=None, type=str,
                    help='dataset root containing images/ and anno/')
parser.add_argument('--num-classes', default=None, type=int,
                    help='override dataset class count')
parser.add_argument('-a', '--arch', default='maskcov_resnet50_quan', choices=model_names,
                    help='model architecture')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate a checkpoint without recalculating quantization')
parser.add_argument('--pretrained', required=True, type=str,
                    help='MaskCOV checkpoint to load')
parser.add_argument('--weight-bit-width', default=4, type=int,
                    help='weight quantization bit-width for backbone convs')
parser.add_argument('--classifier-bit-width', default=None, type=int,
                    help='weight quantization bit-width for MaskCOV classifier heads; default follows --weight-bit-width')
parser.add_argument('--first-conv-bit-width', default=None, type=int,
                    help='weight quantization bit-width for the first conv; default follows --weight-bit-width')
parser.add_argument('--act-bit-width', default=8, type=int,
                    help='activation quantization bit-width')
parser.add_argument('--scales', default='', type=str,
                    help='path to pre-calculated activation scales (.npy)')
parser.add_argument('--resize-resolution', default=440, type=int)
parser.add_argument('--crop-resolution', default=384, type=int)
parser.add_argument('--swap-num', default=[2, 2], nargs=2, type=int)
parser.add_argument('--mask-num', default=1, type=int)
parser.add_argument('--cls-2xmul', action='store_true',
                    help='use 2*num_classes swap classifier instead of binary swap classifier')
parser.add_argument('--no-cdrm', action='store_true',
                    help='disable MaskCOV CDRM auxiliary heads')
parser.add_argument('--calib-batches', default=30, type=int,
                    help='number of batches used for response reconstruction')
parser.add_argument('--samples-per-batch', default=400, type=int,
                    help='patch samples per calibration batch')
parser.add_argument('--act-stat-size', default=3000000, type=int,
                    help='number of activation samples used for scale search')
parser.add_argument('--output-dir', default=None, type=str,
                    help='where to save quantization artifacts')
parser.add_argument('--resume-existing', action='store_true',
                    help='skip already generated quantization artifacts and continue missing layers')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use as quantized-model device')


if torch.cuda.is_available():
    quan_device = torch.device('cuda:0')
    ngpus_per_node = torch.cuda.device_count()
    pretrained_device = torch.device('cuda:1') if ngpus_per_node > 1 else torch.device('cuda:0')
else:
    quan_device = torch.device('cpu')
    pretrained_device = torch.device('cpu')


conv_pretrained_modules = []
conv_quant_modules = []
linear_pretrained_modules = []
linear_quant_modules = []
act_quant_modules = []
feat = None
prev_feat = None
conv_feat = None


def hook(module, inputdata, outputdata):
    global feat
    feat = outputdata.detach().cpu().numpy()


def current_input_hook(module, inputdata, outputdata):
    global prev_feat
    prev_feat = inputdata[0].detach()


def conv_hook(module, inputdata, outputdata):
    global conv_feat
    conv_feat = outputdata.detach()


def main():
    args = parser.parse_args()
    if args.classifier_bit_width is None:
        args.classifier_bit_width = args.weight_bit_width
    if args.first_conv_bit_width is None:
        args.first_conv_bit_width = args.weight_bit_width

    if args.gpu is not None:
        warnings.warn('Specific --gpu is accepted for compatibility; this script uses cuda:0/cuda:1 placement.')

    config = MaskCOVConfig.from_dataset(
        dataset=args.dataset,
        data_root=args.data_root,
        num_classes=args.num_classes,
        swap_num=args.swap_num,
        mask_num=args.mask_num,
        use_cdrm=not args.no_cdrm,
        cls_2=not args.cls_2xmul,
        cls_2xmul=args.cls_2xmul,
        crop_resolution=args.crop_resolution,
    )

    model_kwargs = {
        'num_classes': config.numcls,
        'use_cdrm': config.use_cdrm,
        'cls_2': config.cls_2,
        'cls_2xmul': config.cls_2xmul,
        'swap_num': config.swap_num,
    }
    model_quant = models.__dict__[args.arch](**model_kwargs)
    model_pretrained = models.__dict__[args.arch](**model_kwargs)
    checkpoint = torch.load(args.pretrained, map_location='cpu')
    _report_load('quant', model_quant, checkpoint)
    _report_load('pretrained', model_pretrained, checkpoint)

    model_quant = model_quant.to(quan_device)
    model_pretrained = model_pretrained.to(pretrained_device)

    _collect_modules(model_pretrained, model_quant, args)
    criterion = nn.CrossEntropyLoss().to(quan_device)

    cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

    train_dataset, train_loader, val_loader = build_maskcov_quant_loaders(
        config,
        batch_size=args.batch_size,
        workers=args.workers,
        resize_resolution=args.resize_resolution,
        crop_resolution=args.crop_resolution,
    )

    args.prefix = args.output_dir or os.path.join(
        args.arch, '{}_A{}W{}'.format(args.dataset, args.act_bit_width, args.weight_bit_width)
    )
    os.makedirs(args.prefix, exist_ok=True)

    if args.evaluate:
        if args.scales:
            load_activation_scales(args.scales)
        else:
            print('no activation scales given, use FP32 activations')
        print('validate quantization...')
        metrics = validate(val_loader, model_quant, criterion, args)
        save_metrics(args, metrics, filename='eval_metrics.json')
        return

    model_pretrained.eval()
    model_quant.eval()

    print("weight quantizing ('{}-bit backbone, {}-bit heads')...".format(
        args.weight_bit_width, args.classifier_bit_width
    ))
    quantize_weights(train_loader, model_pretrained, model_quant, args)
    load_quantized_weights(model_pretrained, model_quant, args)

    print("activation quantizing ('{}-bit')...".format(args.act_bit_width))
    update(train_loader, model_quant, criterion, args, max_iter=200)
    if args.scales:
        load_activation_scales(args.scales)
    else:
        quantize_activations(train_dataset, model_quant, args)

    metrics_before_update = validate(val_loader, model_quant, criterion, args)
    save_metrics(args, metrics_before_update, filename='quant_metrics_before_update.json')
    print('update ...')
    update(train_loader, model_quant, criterion, args, max_iter=200)
    print('validate quantization...')
    final_metrics = validate(val_loader, model_quant, criterion, args)
    save_metrics(args, final_metrics, filename='quant_metrics.json')

    save_state_dict(model_quant.state_dict(), args.prefix, filename='state_dict.pth')


def _report_load(label, model, checkpoint):
    missing, unexpected, skipped = load_maskcov_state_dict(model, checkpoint, strict=False)
    if skipped:
        print('{} checkpoint skipped incompatible keys: {}'.format(label, skipped[:10]))
    if missing:
        print('{} checkpoint missing keys: {}'.format(label, missing[:10]))
    if unexpected:
        print('{} checkpoint unexpected keys: {}'.format(label, unexpected[:10]))


def _collect_modules(model_pretrained, model_quant, args):
    conv_pretrained_modules[:] = [
        (name, module) for name, module in model_pretrained.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    conv_quant_modules[:] = [
        (name, module) for name, module in model_quant.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    if len(conv_pretrained_modules) != len(conv_quant_modules):
        raise RuntimeError('conv module count mismatch between pretrained and quant models')

    linear_pretrained_modules[:] = [
        (name, module) for name, module in model_pretrained.named_modules()
        if isinstance(module, nn.Linear)
    ]
    linear_quant_modules[:] = [
        (name, module) for name, module in model_quant.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if len(linear_pretrained_modules) != len(linear_quant_modules):
        raise RuntimeError('linear module count mismatch between pretrained and quant models')

    act_quant_modules[:] = []
    for module in model_quant.modules():
        if isinstance(module, Quantization):
            module.set_bitwidth(args.act_bit_width)
            act_quant_modules.append(module)
    if len(act_quant_modules) > 0:
        act_quant_modules[-1].set_bitwidth(8)

    print('conv modules: {}'.format(len(conv_quant_modules)))
    print('linear modules: {}'.format(len(linear_quant_modules)))
    print('activation quantization modules: {}'.format(len(act_quant_modules)))


def quantize_weights(train_loader, model_pretrained, model_quant, args):
    for idx, ((name, conv), (qname, conv_quant)) in enumerate(zip(conv_pretrained_modules, conv_quant_modules)):
        if name != qname:
            print('warning: conv name mismatch {} vs {}'.format(name, qname))
        prefix = os.path.join(args.prefix, 'conv{:03d}_{}'.format(idx, _safe_name(name)))
        bitwidth = args.first_conv_bit_width if idx == 0 else args.weight_bit_width
        dw = _is_depthwise_conv(conv)
        conduct_ofwa(
            train_loader,
            model_pretrained,
            model_quant,
            conv,
            conv_quant,
            bitwidth,
            prefix=prefix,
            dw=dw,
            calib_batches=args.calib_batches,
            samples_per_batch=args.samples_per_batch,
            resume_existing=args.resume_existing,
        )

    for idx, ((name, linear), (qname, linear_quant)) in enumerate(zip(linear_pretrained_modules, linear_quant_modules)):
        if name != qname:
            print('warning: linear name mismatch {} vs {}'.format(name, qname))
        prefix = os.path.join(args.prefix, 'linear{:03d}_{}'.format(idx, _safe_name(name)))
        conduct_ofwa(
            train_loader,
            model_pretrained,
            model_quant,
            linear,
            linear_quant,
            args.classifier_bit_width,
            prefix=prefix,
            resume_existing=args.resume_existing,
        )


def load_quantized_weights(model_pretrained, model_quant, args):
    for idx, ((name, conv), (qname, conv_quant)) in enumerate(zip(conv_pretrained_modules, conv_quant_modules)):
        prefix = os.path.join(args.prefix, 'conv{:03d}_{}'.format(idx, _safe_name(name)))
        bitwidth = args.first_conv_bit_width if idx == 0 else args.weight_bit_width
        load_ofwa(conv, conv_quant, bitwidth, prefix=prefix)

    for idx, ((name, linear), (qname, linear_quant)) in enumerate(zip(linear_pretrained_modules, linear_quant_modules)):
        prefix = os.path.join(args.prefix, 'linear{:03d}_{}'.format(idx, _safe_name(name)))
        load_ofwa(linear, linear_quant, args.classifier_bit_width, prefix=prefix)


def _safe_name(name):
    return name.replace('.', '_') if name else 'root'


def _is_depthwise_conv(conv):
    return (
        conv.groups == conv.in_channels
        and conv.out_channels == conv.in_channels
        and conv.kernel_size[0] == 3
        and conv.kernel_size[1] == 3
    )


def _next_batch(iterator, loader):
    try:
        return iterator, next(iterator)
    except StopIteration:
        iterator = iter(loader)
        return iterator, next(iterator)


def _batch_to_images_targets(batch):
    if len(batch) < 2:
        raise ValueError('expected batch to contain images and targets')
    return batch[0], batch[1]


def conduct_ofwa(
    train_loader,
    model_pretrained,
    model_quant,
    conv,
    conv_quant,
    bitwidth,
    prefix=None,
    dw=False,
    calib_batches=30,
    samples_per_batch=400,
    resume_existing=False,
):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    if not hasattr(conv, 'kernel_size'):
        output_path = prefix + '_fwa.pkl'
        if resume_existing and _valid_quant_pickle(output_path):
            print('skip existing {}'.format(output_path))
            return
        W = conv.weight.data
        W_shape = W.shape
        B_sav, B, alpha = ofwa(W.cpu().numpy(), bitwidth)
        with open(output_path, 'wb') as out_file:
            pickle.dump({'B': B, 'alpha': alpha}, out_file, pickle.HIGHEST_PROTOCOL)
        return

    if conv.dilation != (1, 1):
        raise ValueError('dilated convolutions are not supported by this BitSplit extractor')

    output_path = prefix + '_rr_b{}x{}_e100.pkl'.format(calib_batches, samples_per_batch)
    if resume_existing and _valid_quant_pickle(output_path):
        print('skip existing {}'.format(output_path))
        return

    kernel_h, kernel_w = conv.kernel_size
    pad_h, pad_w = conv.padding
    stride_h, stride_w = conv.stride

    handle_prev = conv_quant.register_forward_hook(current_input_hook)
    handle_conv = conv.register_forward_hook(conv_hook)

    batch_iterator = iter(train_loader)
    X_parts = []
    Y_parts = []

    W = conv.weight.data
    if conv.bias is None:
        bias = torch.zeros(W.shape[0], device=pretrained_device)
    else:
        bias = conv.bias.data.to(pretrained_device)

    print('{} {}'.format(prefix, tuple(W.shape)))
    try:
        for batch_idx in range(calib_batches):
            batch_iterator, batch = _next_batch(batch_iterator, train_loader)
            images, _ = _batch_to_images_targets(batch)
            input_pretrained = images.to(pretrained_device, non_blocking=True)
            input_quant = images.to(quan_device, non_blocking=True)
            model_pretrained(input_pretrained)
            model_quant(input_quant)

            if prev_feat is None or conv_feat is None:
                raise RuntimeError('failed to collect conv features for {}'.format(prefix))

            prev = prev_feat
            conv_out = conv_feat
            if pad_h > 0 or pad_w > 0:
                prev = F.pad(prev, (pad_w, pad_w, pad_h, pad_h))

            patches = prev.unfold(2, kernel_h, stride_h).unfold(3, kernel_w, stride_w)
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            patches = patches.reshape(-1, patches.shape[3], kernel_h, kernel_w)

            y = conv_out.permute(0, 2, 3, 1).contiguous().reshape(-1, conv_out.shape[1]) - bias
            take = min(samples_per_batch, patches.shape[0])
            rand_index = torch.randperm(patches.shape[0], device=patches.device)[:take]
            X_parts.append(patches[rand_index].cpu())
            Y_parts.append(y[rand_index.to(y.device)].cpu())
    finally:
        handle_prev.remove()
        handle_conv.remove()

    X = torch.cat(X_parts, dim=0).numpy()
    Y = torch.cat(Y_parts, dim=0).numpy()

    W_shape = W.shape
    W_matrix = W.reshape(W_shape[0], -1)
    B_sav, B, alpha = ofwa(W_matrix.cpu().numpy(), bitwidth)
    if dw:
        B, alpha = ofwa_rr_dw(X, Y, B_sav, alpha, bitwidth, max_epoch=100)
    else:
        B, alpha = ofwa_rr(X, Y, B_sav, alpha, bitwidth, max_epoch=100)
    with open(output_path, 'wb') as out_file:
        pickle.dump({'B': B, 'alpha': alpha}, out_file, pickle.HIGHEST_PROTOCOL)


def _valid_quant_pickle(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path, 'rb') as in_file:
            payload = pickle.load(in_file)
    except (EOFError, OSError, pickle.PickleError, ValueError):
        return False
    return isinstance(payload, dict) and 'B' in payload and 'alpha' in payload


def load_ofwa(conv, conv_quant, bitwidth, prefix=None):
    if not hasattr(conv, 'kernel_size'):
        W = conv.weight.data
        W_shape = W.shape
        with open(prefix + '_fwa.pkl', 'rb') as in_file:
            B_alpha = pickle.load(in_file)
        B = B_alpha['B']
        alpha = B_alpha['alpha']
        W_r = np.multiply(B, np.expand_dims(alpha, 1)).reshape(W_shape)
        conv_quant.weight.data.copy_(torch.from_numpy(W_r).to(conv_quant.weight.device).type_as(conv_quant.weight))
        return

    W = conv.weight.data
    W_shape = W.shape
    rr_path = _find_rr_file(prefix)
    with open(rr_path, 'rb') as in_file:
        B_alpha = pickle.load(in_file)
    B = B_alpha['B']
    alpha = B_alpha['alpha']
    W_r = np.multiply(B, np.expand_dims(alpha, 1)).reshape(W_shape)
    conv_quant.weight.data.copy_(torch.from_numpy(W_r).to(conv_quant.weight.device).type_as(conv_quant.weight))


def _find_rr_file(prefix):
    directory = os.path.dirname(prefix)
    basename = os.path.basename(prefix)
    for filename in os.listdir(directory):
        if filename.startswith(basename) and filename.endswith('.pkl') and '_rr_' in filename:
            return os.path.join(directory, filename)
    raise FileNotFoundError('no response-reconstruction file found for {}'.format(prefix))


def quantize_activations(train_dataset, model, args):
    def get_safelen(x):
        x = x / 10
        y = 1
        while x >= 10:
            x = x / 10
            y = y * 10
        return int(y)

    scales = np.zeros(len(act_quant_modules))
    print('act quantization modules: ', len(act_quant_modules))

    with torch.no_grad():
        for index, q_module in enumerate(act_quant_modules):
            loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
            )
            batch_iterator = iter(loader)
            batch_iterator, batch = _next_batch(batch_iterator, loader)
            images, _ = _batch_to_images_targets(batch)
            images = images.to(quan_device, non_blocking=True)

            handle = q_module.register_forward_hook(hook)
            model(images)

            feat_len = feat.size
            per_batch = min(get_safelen(feat_len), 100000)
            n_batches = max(1, int(args.act_stat_size / per_batch))
            feat_buf = np.zeros(n_batches * per_batch)

            failed = True
            while failed:
                failed = False
                print('Extracting features for ', n_batches, ' batches...')
                for batch_idx in range(0, n_batches):
                    batch_iterator, batch = _next_batch(batch_iterator, loader)
                    images, _ = _batch_to_images_targets(batch)
                    images = images.to(quan_device, non_blocking=True)
                    model(images)

                    if q_module.signed:
                        feat_tmp = np.abs(feat).reshape(-1)
                    else:
                        feat_tmp = feat[feat > 0].reshape(-1)
                        if feat_tmp.size < per_batch:
                            per_batch = max(1, int(per_batch / 10))
                            n_batches = max(1, int(args.act_stat_size / per_batch))
                            feat_buf = np.zeros(n_batches * per_batch)
                            failed = True
                            break
                    np.random.shuffle(feat_tmp)
                    feat_buf[batch_idx * per_batch:(batch_idx + 1) * per_batch] = feat_tmp[0:per_batch]

                if not failed:
                    print('Init quantization... ')
                    scales[index] = q_module.init_quantization(feat_buf)
                    print(scales[index])
                    np.save(os.path.join(args.prefix, 'act_' + str(args.act_bit_width) + '_scales.npy'), scales)
            handle.remove()

    np.save(os.path.join(args.prefix, 'act_' + str(args.act_bit_width) + '_scales.npy'), scales)
    for index, q_module in enumerate(act_quant_modules):
        q_module.set_scale(scales[index])


def load_activation_scales(path):
    scales = np.load(path)
    if len(scales) != len(act_quant_modules):
        raise ValueError('scale count {} does not match quant modules {}'.format(len(scales), len(act_quant_modules)))
    for index, q_module in enumerate(act_quant_modules):
        q_module.set_scale(scales[index])


def update(train_loader, model, criterion, args, max_iter):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    model.train()
    global_iter = 0
    end = time.time()
    with torch.no_grad():
        while global_iter < max_iter:
            for i, batch in enumerate(train_loader):
                global_iter += 1
                data_time.update(time.time() - end)
                input, target = _batch_to_images_targets(batch)
                input = input.to(quan_device, non_blocking=True)
                target = target.to(quan_device, non_blocking=True)

                output = _classification_output(model(input))
                loss = criterion(output, target)
                acc1, acc3 = accuracy(output, target, topk=(1, min(3, output.size(1))))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top3.update(acc3[0], input.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Update: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                              i, len(train_loader), batch_time=batch_time,
                              data_time=data_time, loss=losses, top1=top1, top3=top3))
                if global_iter >= max_iter:
                    break


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            input, target = _batch_to_images_targets(batch)
            input = input.to(quan_device, non_blocking=True)
            target = target.to(quan_device, non_blocking=True)

            output = _classification_output(model(input))
            loss = criterion(output, target)
            acc1, acc3 = accuracy(output, target, topk=(1, min(3, output.size(1))))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top3.update(acc3[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top3=top3))

        print(' * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f}'.format(top1=top1, top3=top3))

    return {
        'loss': losses.avg,
        'acc1': top1.avg / 100.0,
        'acc3': top3.avg / 100.0,
        'acc1_percent': top1.avg,
        'acc3_percent': top3.avg,
    }


def _classification_output(output):
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def save_state_dict(state_dict, path, filename='state_dict.pth'):
    saved_path = os.path.join(path, filename)
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if '.module.' in key:
            new_state_dict[key.replace('.module.', '.')] = state_dict[key].cpu()
        else:
            new_state_dict[key] = state_dict[key].cpu()
    torch.save(new_state_dict, saved_path)
    print('saved model to {}'.format(saved_path))


def save_metrics(args, metrics, filename='quant_metrics.json'):
    metrics_path = os.path.join(args.prefix, filename)
    payload = OrderedDict()
    payload['arch'] = args.arch
    payload['dataset'] = args.dataset
    payload['data_root'] = args.data_root
    payload['pretrained'] = args.pretrained
    payload['weight_bit_width'] = args.weight_bit_width
    payload['first_conv_bit_width'] = args.first_conv_bit_width
    payload['classifier_bit_width'] = args.classifier_bit_width
    payload['act_bit_width'] = args.act_bit_width
    payload['state_dict'] = os.path.join(args.prefix, 'state_dict.pth')
    payload['activation_scales'] = os.path.join(args.prefix, 'act_' + str(args.act_bit_width) + '_scales.npy')
    payload.update(metrics)
    with open(metrics_path, 'w') as metrics_file:
        json.dump(payload, metrics_file, indent=2)
    print('saved metrics to {}'.format(metrics_path))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if hasattr(val, 'item'):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
