import math
import time
import json
import random
import warnings
warnings.simplefilter('module')

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import ultralytics
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.utils import dist, RANK


@smart_inference_mode()
def modified_call(self, trainer=None, model=None):
    """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
    gets priority).
    """
    self.training = trainer is not None
    augment = self.args.augment and (not self.training)
    if self.training:
        self.device = trainer.device
        self.data = trainer.data
        self.args.half = self.device.type != 'cpu'  # force FP16 val during training
        model = trainer.ema.ema or trainer.model
        model = model.half() if self.args.half else model.float()
        # self.model = model
        self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
        self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
        model.eval()
    else:
        callbacks.add_integration_callbacks(self)
        model = AutoBackend(model or self.args.model,
                            device=select_device(self.args.device, self.args.batch),
                            dnn=self.args.dnn,
                            data=self.args.data,
                            fp16=self.args.half)
        # self.model = model
        self.device = model.device  # update device
        self.args.half = model.fp16  # update half
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_imgsz(self.args.imgsz, stride=stride)
        if engine:
            self.args.batch = model.batch_size
        elif not pt and not jit:
            self.args.batch = 1  # export.py models default to batch-size 1
            LOGGER.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        if str(self.args.data).split('.')[-1] in ('yaml', 'yml'):
            self.data = check_det_dataset(self.args.data)
        elif self.args.task == 'classify':
            self.data = check_cls_dataset(self.args.data, split=self.args.split)
        else:
            raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

        if self.device.type in ('cpu', 'mps'):
            self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
        if not pt:
            self.args.rect = False
        self.stride = model.stride  # used in get_dataloader() for padding
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

        model.eval()
        model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

    self.run_callbacks('on_val_start')
    dt = Profile(), Profile(), Profile(), Profile()
    bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
    self.init_metrics(de_parallel(model))
    self.jdict = []  # empty before each val
    for batch_i, batch in enumerate(bar):
        self.batch = batch # added by Franz
        self.run_callbacks('on_val_batch_start')
        batch = self.batch # added by Franz
        self.batch_i = batch_i
        # Preprocess
        with dt[0]:
            batch = self.preprocess(batch)

        # Inference
        with dt[1]:
            preds = model(batch['img'], augment=augment)

        # Loss
        with dt[2]:
            if self.training:
                self.loss += model.loss(batch, preds)[1]
                # TODO: cls_box ist Inf, warum?

        # Postprocess
        with dt[3]:
            preds = self.postprocess(preds)

        self.update_metrics(preds, batch)
        if self.args.plots and batch_i < 3:
            self.plot_val_samples(batch, batch_i)
            self.plot_predictions(batch, preds, batch_i)

        self.run_callbacks('on_val_batch_end')
    stats = self.get_stats()
    self.check_stats(stats)
    self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1E3 for x in dt)))
    self.finalize_metrics()
    self.print_results()
    self.run_callbacks('on_val_end')
    if self.training:
        model.float()
        results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
        return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
    else:
        LOGGER.info('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
                    tuple(self.speed.values()))
        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                LOGGER.info(f'Saving {f.name}...')
                json.dump(self.jdict, f)  # flatten and save
            stats = self.eval_json(stats)  # update stats
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats


def modified_affine_transform(self, img, border):
    """
    Applies a sequence of affine transformations centered around the image center.

    Args:
        img (ndarray): Input image.
        border (tuple): Border dimensions.

    Returns:
        img (ndarray): Transformed image.
        M (ndarray): Transformation matrix.
        s (float): Scale factor.
    """

    # Center
    C = np.eye(3, dtype=np.float32)

    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3, dtype=np.float32)
    P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3, dtype=np.float32)
    a = random.uniform(-self.degrees, self.degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    # modified by Franz: +0.2, so we can scale from 0.2 to 2.0 instead of only 1.8
    s = random.uniform(1 - self.scale, 1 + self.scale + 0.2)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3, dtype=np.float32)
    S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3, dtype=np.float32)
    T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    # Affine image
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if self.perspective:
            img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
    return img, M, s


def modified_do_train(self, world_size=1):
    """Train completed, evaluate and plot if specified by arguments."""
    if world_size > 1:
        self._setup_ddp(world_size)
    self._setup_train(world_size)

    nb = len(self.train_loader)  # number of batches
    nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
    last_opt_step = -1
    self.epoch_time = None
    self.epoch_time_start = time.time()
    self.train_time_start = time.time()
    self.run_callbacks("on_train_start")
    LOGGER.info(
        f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
        f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
        f"Logging results to {colorstr('bold', self.save_dir)}\n"
        f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
    )
    if self.args.close_mosaic:
        base_idx = (self.epochs - self.args.close_mosaic) * nb
        self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
    epoch = self.start_epoch
    while True:
        self.epoch = epoch
        self.run_callbacks("on_train_epoch_start")
        self.model.train()
        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(self.train_loader)
        # Update dataloader attributes (optional)
        if epoch == (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()
            self.train_loader.reset()

        if RANK in (-1, 0):
            LOGGER.info(self.progress_string())
            pbar = TQDM(enumerate(self.train_loader), total=nb)
        self.tloss = None
        self.optimizer.zero_grad()
        for i, batch in pbar:
            self.batch = batch # added by Franz
            self.run_callbacks("on_train_batch_start")
            batch = self.batch # added by Franz
            # Warmup
            ni = i + nb * epoch
            if ni <= nw:
                xi = [0, nw]  # x interp
                self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                for j, x in enumerate(self.optimizer.param_groups):
                    # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(
                        ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

            # Forward
            with torch.cuda.amp.autocast(self.amp):
                batch = self.preprocess_batch(batch)
                self.loss, self.loss_items = self.model(batch)
                if RANK != -1:
                    self.loss *= world_size
                self.tloss = (
                    (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                )

            # Backward
            self.scaler.scale(self.loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= self.accumulate:
                self.optimizer_step()
                last_opt_step = ni

                # Timed stopping
                if self.args.time:
                    self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                    if RANK != -1:  # if DDP training
                        broadcast_list = [self.stop if RANK == 0 else None]
                        dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                        self.stop = broadcast_list[0]
                    if self.stop:  # training time exceeded
                        break

            # Log
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
            losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
            if RANK in (-1, 0):
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                    % (f"{epoch + 1}/{self.epochs}", mem, *losses, batch["cls"].shape[0], batch["img"].shape[-1])
                )
                self.run_callbacks("on_batch_end")
                if self.args.plots and ni in self.plot_idx:
                    self.plot_training_samples(batch, ni)

            self.run_callbacks("on_train_batch_end")

        self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
        self.run_callbacks("on_train_epoch_end")
        if RANK in (-1, 0):
            final_epoch = epoch + 1 == self.epochs
            self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self.metrics, self.fitness = self.validate()
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
            if self.args.time:
                self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

            # Save model
            if self.args.save or final_epoch:
                self.save_model()
                self.run_callbacks("on_model_save")

        # Scheduler
        t = time.time()
        self.epoch_time = t - self.epoch_time_start
        self.epoch_time_start = t
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.scheduler.step()
        self.run_callbacks("on_fit_epoch_end")
        torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

        # Early Stopping
        if RANK != -1:  # if DDP training
            broadcast_list = [self.stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            self.stop = broadcast_list[0]
        if self.stop:
            break  # must break all DDP ranks
        epoch += 1

    if RANK in (-1, 0):
        # Do final val with best.pt
        LOGGER.info(
            f"\n{epoch - self.start_epoch + 1} epochs completed in "
            f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
        )
        self.final_eval()
        if self.args.plots:
            self.plot_metrics()
        self.run_callbacks("on_train_end")
    torch.cuda.empty_cache()
    self.run_callbacks("teardown")


def modified_conv_forward(self, x, w=None, b=None):
    if w is None:
        w = self.conv.weight
    if b is None:
        b = self.conv.bias

    # ensure same dtype during validation run
    dtype = x.dtype
    
    # TODO: only needed when using Cuda devices != 0
    # x = x.to(w.device)
    
    w = w.to(dtype)
    if b is not None:
        b = b.to(dtype)

    x_f = F.conv2d(x, w, b, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)

    # TODO: so far it only works with Cuda device == 0
    return self.act(self.bn(x_f))


def shortened_predict_once(self, x, profile=False, visualize=False, embed=None):
    y, dt, embeddings = [], [], []  # outputs
    for idx, m in enumerate(self.model):
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

        # 1. condition: only for Conv, 2. condition: make sure that the scn module is available, 3. condition: specify layer
        if type(m) == ultralytics.nn.modules.conv.Conv and hasattr(ultralytics, 'scn') and idx == 19:
            hyper_output = ultralytics.scn(ultralytics.transformation_value)
            key = 'layer_{:03d}__{:02d}'
            conv_w = torch.stack([ultralytics.scn.model_params[key.format(idx, d)] * hyper_output[d] for d in range(ultralytics.scn.dimensions)]).sum(dim=0)

            conv_b = None
            if 'bias_' + key.format(idx, 0) in ultralytics.scn.model_params.keys():
                conv_b = torch.stack([ultralytics.scn.model_params['bias_' + key.format(idx, d)] * hyper_output[d] for d in range(ultralytics.scn.dimensions)]).sum(dim=0)
            x = m(x, conv_w, conv_b)
        else:
            x = m(x)
        y.append(x if m.i in self.save else None)
    return x




from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors

def v8_loss__call__(self, preds, batch):
    """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
    loss = torch.zeros(3, device=self.device)  # box, cls, dfl
    feats = preds[1] if isinstance(preds, tuple) else preds
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
        (self.reg_max * 4, self.nc), 1
    )

    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
    anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

    # Targets
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

    # Pboxes
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

    _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )

    target_scores_sum = max(target_scores.sum(), 1)

    # Cls loss
    # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
    loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

    # Bbox loss
    if fg_mask.sum():
        target_bboxes /= stride_tensor
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
        )

    loss[0] *= self.hyp.box  # box gain
    loss[1] *= self.hyp.cls  # cls gain
    loss[2] *= self.hyp.dfl  # dfl gain

    return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)