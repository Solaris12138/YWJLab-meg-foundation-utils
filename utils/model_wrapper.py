import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from logzero import logger
from pathlib import Path


class PretrainPipeline(object):

    def __init__(self,
        model,
        model_path,
        batch_size,
        is_distributed,
        device,
        device_ids,
        backend,
        world_size,
        ema=[0.996, 1.0],
        chan_mask_ratio_low=0.30,
        chan_mask_ratio_high=0.80,
        use_augmentation=False,
        if_earlystop=True,
        earlystop_patience=10,
        delta=0.0005,
        **kwargs
    ):
        self.device = device
        self.model = model(device=device, **kwargs)
        
        self.is_distributed = is_distributed
        if is_distributed:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self._init_process_group(backend, world_size)
            self.model = DDP(self.model, device_ids=device_ids)
            self.n_devices = world_size
            self.writer = SummaryWriter(os.path.split(model_path)[0]) if dist.get_rank() == 0 else None
        else:
            self.n_devices = 1
            self.writer = SummaryWriter(os.path.split(model_path)[0])

        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.ema = ema
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self.chan_mask_ratio_low = chan_mask_ratio_low
        self.chan_mask_ratio_high = chan_mask_ratio_high
        self.if_earlystop = if_earlystop
        self.earlystop_patience, self.delta, self.counter = earlystop_patience, delta, 0

        self.optimizer = None
        self.criterion = nn.MSELoss()

    def _get_optimizer(self, optimizer_cls='Adam', scheduler_cls='StepLR', lr=1e-4, weight_decay=1e-3, **kwargs):
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls
        
        if isinstance(optimizer_cls, str):
            optimizer_cls_ = getattr(torch.optim, optimizer_cls)
        if isinstance(scheduler_cls, str):
            if scheduler_cls not in ['ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR']:
                raise ValueError(f'Default scheduler class should be StepLR|ReduceLROnPlateau|CosineAnnealingLR, but got {scheduler_cls}.')
            scheduler_cls_ = getattr(torch.optim.lr_scheduler, scheduler_cls)
            
        self.optimizer = optimizer_cls_(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if scheduler_cls == 'StepLR':
            self.scheduler = scheduler_cls_(self.optimizer, kwargs['step_size'], kwargs['gamma'])
        elif scheduler_cls == 'ReduceLROnPlateau':
            self.scheduler = scheduler_cls_(self.optimizer, 'min')
        else:
            self.scheduler = scheduler_cls_(self.optimizer, kwargs['T_max'])
    
    @torch.no_grad()
    def _update_momentum_encoder(self):
        momentum_factor = next(self.momentum_scheduler)
        if self.is_distributed:
            param_zip = zip(self.model.module.encoder.parameters(), 
                            self.model.module.momentum_encoder.parameters())
        else:
            param_zip = zip(self.model.encoder.parameters(), 
                            self.model.momentum_encoder.parameters())
        
        for online_param, momentum_param in param_zip:
            momentum_param.data = momentum_factor * momentum_param.data + (1 - momentum_factor) * online_param.detach().data
    
    def _make_masks(self, chan_mask_ratio=0.50, temp_mask_prob=0.50):
        C, N = self.model.module.num_patches if self.is_distributed else self.model.num_patches

        num_visible_chan_patches = math.ceil(C * (1 - chan_mask_ratio))

        while True:
            visible_idx = list()
            invisible_idx_ = list()
            invisible_idx_all = list()

            for i in range(N):
                chan_idx = torch.randperm(C) + i * C
                if np.random.uniform(0, 1) > temp_mask_prob:
                    visible_idx.append(chan_idx[:num_visible_chan_patches])
                    invisible_idx_.append(chan_idx[num_visible_chan_patches:])
                else:
                    invisible_idx_all.append(chan_idx)
            
            if len(visible_idx) == 0: continue
            if len(invisible_idx_) == 0: continue

            invisible_idx_ = torch.cat(invisible_idx_, dim=0)
            invisible_idx_ = invisible_idx_[torch.rand(invisible_idx_.shape) < 0.2] # Randomly select.
            if len(invisible_idx_) == 0: continue
            
            break

        invisible_idx_all = invisible_idx_all + [invisible_idx_]

        return torch.stack(visible_idx, dim=0), torch.cat(invisible_idx_all, dim=0)
    
    def _valid_step(self, data_temp, chan_mask_ratio=0.50, temp_mask_prob=0.50):
        visible_idx, invisible_idx = self._make_masks(chan_mask_ratio, temp_mask_prob)

        self.model.eval()

        with torch.no_grad():
            z, r = self.model(data_temp, visible_idx, invisible_idx)
            if self.is_distributed:
                h, y = self.model.module.forward_momentum(data_temp, invisible_idx)
            else:
                h, y = self.model.forward_momentum(data_temp, invisible_idx)

            loss = self.criterion(h, z) + self.criterion(y, r)
        
        return loss.item()

    def _pretrain_step(self, data_temp, chan_mask_ratio=0.50, temp_mask_prob=0.50):
        visible_idx, invisible_idx = self._make_masks(chan_mask_ratio, temp_mask_prob)
        
        self.optimizer.zero_grad()
        self.model.train()
        torch.autograd.set_detect_anomaly(True)

        z, r = self.model(data_temp, visible_idx, invisible_idx)
        with torch.no_grad():
            if self.is_distributed:
                h, y = self.model.module.forward_momentum(data_temp, invisible_idx)
            else:
                h, y = self.model.forward_momentum(data_temp, invisible_idx)

        loss = self.criterion(h, z) + self.criterion(y, r)
        loss.backward()
        self.optimizer.step(closure=None)
        self._update_momentum_encoder()
        
        return loss.item()

    def pretrain(self, train_loader, valid_loader, opt_params, epochs_num, verbose=True, **kwargs):
        self._get_optimizer(**opt_params)

        steps_num = int(math.ceil(len(train_loader) / self.n_devices) * epochs_num)
        self.momentum_scheduler = (
            self.ema[0] + i * (self.ema[-1]-self.ema[0]) / steps_num for i in range(steps_num + 1)
        )

        for epoch_idx in range(1, epochs_num + 1):
            chan_mask_ratio = np.linspace(self.chan_mask_ratio_low, self.chan_mask_ratio_high, 10)[epoch_idx - 1] if epoch_idx <= 10 else self.chan_mask_ratio_high

            # Training Phase
            train_losses = list()
            for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch_idx}', ncols=80, leave=False, dynamic_ncols=False): 
                if self.use_augmentation:
                    x = (data['temp'], data['temp_aug'])
                    x = x[math.floor(np.random.uniform(0, 1) / 0.5)].to(self.device) # Randomly select one for trainning.
                else:
                    x = data['temp'].to(self.device)
                train_loss_batch = self._pretrain_step(x, chan_mask_ratio=chan_mask_ratio, temp_mask_prob=0.50)
                train_losses.append(train_loss_batch)
                if not self.is_distributed or dist.get_rank() == 0:
                    self.writer.add_scalar('Loss/train', train_loss_batch, (epoch_idx - 1) * len(train_loader) + batch_idx)
            
            train_loss = sum(train_losses) / len(train_losses)
            if self.is_distributed: 
                train_loss = torch.Tensor([train_loss]).to(self.device)
                dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
                train_loss = train_loss.item() / dist.get_world_size()
            
            # Validation Phase
            if_valid = (epoch_idx >= 10) and (epoch_idx % 5 == 0)
            if if_valid:
                valid_losses = list()
                for batch_idx, data in tqdm(enumerate(valid_loader), total=len(valid_loader), ncols=80, leave=False, dynamic_ncols=False): 
                    x = data['temp'].to(self.device)
                    valid_loss_batch = self._valid_step(x, chan_mask_ratio=self.chan_mask_ratio_high, temp_mask_prob=0.50) # Fixed channel mask ratio.
                    valid_losses.append(valid_loss_batch)
                
                valid_loss = sum(valid_losses) / len(valid_losses)
                if self.is_distributed: 
                    valid_loss = torch.Tensor([valid_loss]).to(self.device)
                    dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
                    valid_loss = valid_loss.item() / dist.get_world_size()
                
                if self.scheduler_cls == 'ReduceLROnPlateau':
                    self.scheduler.step(valid_loss)
                else:
                    self.scheduler.step()
            
            # Logger
            if not self.is_distributed or dist.get_rank() == 0:
                if if_valid:
                    self.save_model(epoch_idx)
                if verbose:
                    logger.info(f'Epoch: {epoch_idx} '
                                f'train loss: {train_loss:.5f} '
                                f'valid loss: {valid_loss:.5f}' if if_valid else 'valid loss: N/A')
        
        if not self.is_distributed or dist.get_rank() == 0:
            self.writer.close()

    def save_model(self, epoch_idx):
        torch.save(
            self.model.state_dict(),
            Path(str(self.model_path).replace('.pt', f'_epoch-{epoch_idx}.pt'))
        )
    
    def _init_process_group(self, backend, world_size):
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=world_size,
            rank=int(os.environ['RANK'])
        )
    
    def clean_process_group(self):
        dist.destroy_process_group()