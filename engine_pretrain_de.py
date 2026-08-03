# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch_de(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples_mix, samples_clean, omega_target, accel_win) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples_mix = samples_mix.to(device, non_blocking=True)
        samples_clean = samples_clean.to(device, non_blocking=True)
        omega_target = omega_target.to(device, non_blocking=True)
        accel_win = accel_win.to(device, non_blocking=True)

        # Conditionally enable autocast only if we are on a CUDA device
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            loss, _, _, loss_parts = model(samples_mix, samples_clean, omega_target, accel_win)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # Only synchronize if a CUDA device is being used
        if device.type == 'cuda':
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        for k, v in loss_parts.items():
            if v is not None:
                metric_logger.update(**{k: v.item()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_random_mask_half(batch_size, seq_len, patch_size, device=None):
    """
    For a sequence of length seq_len, we form patches of size patch_size.
    The number of patches is n_patches = seq_len // patch_size.

    This function randomly selects half of the patches to be masked (mask=1)
    and keeps the other half unmasked (mask=0).

    Returns a float32 mask tensor of shape [batch_size, n_patches] with values in {0, 1}.
    """
    import torch

    n_patches = seq_len // patch_size
    if device is None:
        device = torch.device("cpu")

    # Start with an all-zero mask.
    mask = torch.zeros(batch_size, n_patches, dtype=torch.float32, device=device)

    # Mask half of the patches.
    n_mask = n_patches // 2

    for b in range(batch_size):
        # Randomly permute patch indices.
        indices = torch.randperm(n_patches, device=device)
        # Take the first n_mask as masked patches.
        mask_indices = indices[:n_mask]
        # Mark masked patches with 1.
        mask[b, mask_indices] = 1.0

    return mask


def train_one_epoch_r(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None
):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples_mix, samples_clean) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # --- Adjust learning rate per iteration (if this is your intended schedule) ---
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader) + epoch,
                args
            )

        samples_mix = samples_mix.to(device, non_blocking=True)
        samples_clean = samples_clean.to(device, non_blocking=True)

        # ============ Generate a random mask (once per iteration) ============
        B = samples_mix.size(0)
        seq_len = 256    # sequence length
        patch_size = 2    # patch size
        # Note: model(...) must support mask=... if you use this training variant.
        mask = get_random_mask_half(B, seq_len, patch_size, device=device)

        # ============ Forward pass with mask ============
        # Conditionally enable autocast only if we are on a CUDA device
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            loss, _, _ = model(samples_mix, samples_clean, mask=mask)
            # Ensure your model forward supports mask=...

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Gradient accumulation
        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # Only synchronize if a CUDA device is being used
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Logging
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            # You can use iteration/epoch as the x-axis in TensorBoard.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}