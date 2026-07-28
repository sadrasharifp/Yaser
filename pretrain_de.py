import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
import timm.optim as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# Import the dual-mask self-supervised denoising model.
# The model forward is expected to accept (x_noisy, x_clean).
import DE


from engine_pretrain_de import train_one_epoch_de
#####################################
# 1) Dataset: TimeSeriesDatasetPair
#####################################
class TimeSeriesDatasetPair(torch.utils.data.Dataset):
    """
    Loads paired "noisy" and "clean" time-series windows.

    The two folders must contain 1:1 matching .npy files (same sorted order).
    """
    def __init__(self, root_noisy, root_clean, root_earth=None, transform=None):
        self.root_noisy = Path(root_noisy)
        self.root_clean = Path(root_clean)
        self.root_earth = Path(root_earth) if root_earth else None
        self.transform = transform

        self.files_noisy = sorted(list(self.root_noisy.glob("*.npy")))
        self.files_clean = sorted(list(self.root_clean.glob("*.npy")))
        assert len(self.files_noisy) == len(self.files_clean), \
            "Noisy and clean file counts do not match."
        assert [p.name for p in self.files_noisy] == [p.name for p in self.files_clean], \
            "Noisy and clean filenames do not match."

        if self.root_earth is not None:
            self.files_earth = [self.root_earth / p.name for p in self.files_noisy]
            missing = [str(p) for p in self.files_earth if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing Earth target files, first missing: {missing[0]}")
        else:
            self.files_earth = None

    @staticmethod
    def _load_window(path):
        data = np.load(path).astype(np.float32)
        if data.ndim == 1:
            data = data[None, :]  # [1,L]
        elif data.ndim != 2:
            raise ValueError(f"Expected 1D or 2D .npy window at {path}, got shape {data.shape}")
        return torch.from_numpy(data)

    def __len__(self):
        return len(self.files_noisy)

    def __getitem__(self, index):
        data_noisy = self._load_window(self.files_noisy[index])  # [C,L]
        data_clean = self._load_window(self.files_clean[index])  # [C,L]

        if self.transform:
            data_noisy = self.transform(data_noisy)
            # data_clean = self.transform(data_clean) # 如果也需要transform

        if self.files_earth is not None:
            omega_target = torch.tensor(np.load(self.files_earth[index]), dtype=torch.float32).reshape(-1)
        else:
            # Fallback DC target: per-channel mean of the normalized clean
            # window. Physics-constrained stationary datasets should instead
            # provide root_earth so the target is theoretical Earth rate.
            omega_target = data_clean.mean(dim=1)

        return data_noisy, data_clean, omega_target

    def sample_shape(self):
        sample = self._load_window(self.files_noisy[0])
        return tuple(sample.shape)


#####################################
# 2) add_weight_decay (unchanged)
#####################################
def add_weight_decay(model, weight_decay=0.05, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay,   'weight_decay': weight_decay}
    ]

#####################################
# 3) Arg parser: adds data_path_noisy / data_path_clean
#####################################
def get_args_parser():
    parser = argparse.ArgumentParser('MAE Noise Training', add_help=False)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=41, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)

    # Model
    parser.add_argument('--model', default='', type=str,
                        help='Name of model in model_noisemae_o')

    # Dataset paths: noisy & clean
    parser.add_argument('--data_path_noisy', default=r'', type=str,
                        help='path to NOISY time series dataset (npy files)')
    parser.add_argument('--data_path_clean', default=r'', type=str,
                        help='path to CLEAN time series dataset (npy files)')
    parser.add_argument('--earth_target_path', default=r'', type=str,
                        help='optional path to per-window Earth/DC target .npy files')

    parser.add_argument('--output_dir', default='',
                        help='path where to save checkpoints')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--in_chans', default=0, type=int,
                        help='input channel count; 0=infer from first .npy window')
    parser.add_argument('--seq_len', default=0, type=int,
                        help='window length; 0=infer from first .npy window')
    parser.add_argument('--lambda_earth', default=1.0, type=float,
                        help='weight for stationary Earth-vector mean loss')
    parser.add_argument('--lambda_earth_mag', default=0.2, type=float,
                        help='weight for stationary Earth-vector magnitude loss')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    # Optimizer hyper-parameters
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=0.)
    parser.add_argument('--warmup_epochs', type=int, default=0)

    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser

#####################################
# 4) load_checkpoint (unchanged)
#####################################
def load_checkpoint(args, model_without_ddp, optimizer, loss_scaler, device):
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # NOTE: misc.save_model() stores the AMP scaler under the 'scaler' key,
        # so we must read the same key here (previously read 'loss_scaler',
        # which silently skipped scaler restoration on resume).
        if loss_scaler is not None and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed checkpoint from {args.resume}, starting from epoch {start_epoch}")
        return start_epoch
    else:
        print("No valid resume checkpoint, start from epoch 0")
        return 0

#####################################
# 5) main
#####################################
def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    transform_train = None

    # Use paired dataset
    dataset_train = TimeSeriesDatasetPair(
        root_noisy=args.data_path_noisy,
        root_clean=args.data_path_clean,
        root_earth=args.earth_target_path,
        transform=transform_train
    )
    print("Dataset with pair noisy+clean:", dataset_train)

    sample_c, sample_l = dataset_train.sample_shape()
    if args.in_chans <= 0:
        args.in_chans = int(sample_c)
    if args.seq_len <= 0:
        args.seq_len = int(sample_l)
    print(f"Detected training window shape: channels={args.in_chans}, seq_len={args.seq_len}")

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train =", str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if misc.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # Build model
    model = DE.__dict__[args.model](
        in_chans=args.in_chans,
        seq_len=args.seq_len,
        lambda_earth=args.lambda_earth,
        lambda_earth_mag=args.lambda_earth_mag,
    )
    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations:", args.accum_iter)
    print("effective batch size:", eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )
        model_without_ddp = model.module

    param_groups = add_weight_decay(model_without_ddp, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print("Optimizer:", optimizer)
    loss_scaler = NativeScaler()

    # Single resume path: load_checkpoint restores model/optimizer/scaler and
    # returns the epoch to start from. (misc.load_model is a second, duplicate
    # resume mechanism using a different key layout, so it is not called here.)
    start_epoch = load_checkpoint(args, model_without_ddp, optimizer, loss_scaler, device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # ---- train_one_epoch ----
        train_stats = train_one_epoch_de(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # Save checkpoint
        if args.output_dir and ((epoch % 5 == 0) or (epoch + 1 == args.epochs)):
            misc.save_model(
                args=args, model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch
            )

        # Logging
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)





if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
