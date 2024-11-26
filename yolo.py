import argparse
import importlib
import warnings
from glob import glob

import torch
import torch.nn as nn
import joblib

import wandb
import ultralytics
from ultralytics import YOLO

from constants import SCALE_MIN
from constants import SCALE_MAX
from constants import NUM_EVAL_SCALE_FACTORS
from yolo_modifications import *

BORDER_COLOR = (114, 114, 114)
ULTRALYTICS_VERSION = '8.1.20'


def patch_method(module, orig_method, new_method):
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        warnings.warn(f'Monkey patching Ultralytics was not tested with version {ultralytics.__version__} of Ultralytics. In case of errors, please use version {ULTRALYTICS_VERSION} of Ultralytics.')

    orig_class = module.split('.')[-1]
    orig_module = '.'.join(module.split('.')[:-1])
    module = importlib.import_module(orig_module)

    patched_class = getattr(module, orig_class)
    setattr(patched_class, orig_method, new_method)


def patch_validator():
    # allowing batch to be modified:
    patch_method('ultralytics.engine.validator.BaseValidator', '__call__', modified_call)

    # patching scaling to work for unsymmetrical scale values: s = random.uniform(1 - self.scale, 1 + self.scale + 0.2)
    patch_method('ultralytics.data.augment.RandomPerspective', 'affine_transform', modified_affine_transform)


def transformation_fn(validator):
    cls = []
    bboxes = []
    batch_idxs = []
    transformed_images = []

    for i in range(len(validator.batch['batch_idx'])):
        batch_idx = int(validator.batch['batch_idx'][i])

        image = validator.batch['img'][batch_idx].numpy().transpose(1, 2, 0)
        h_img, w_img, _ = image.shape
        cx, cy = w_img // 2, h_img // 2

        if ultralytics.transformation == 'rotate':
            M = cv2.getRotationMatrix2D((cx, cy), ultralytics.transformation_value, 1.0)
        elif ultralytics.transformation == 'scale':
            M = cv2.getRotationMatrix2D((cx, cy), 0, ultralytics.transformation_value)
        else:
            raise ValueError(f'Unknown transformation: {ultralytics.transformation}')

        img_fn = validator.batch['im_file'][batch_idx]
        if img_fn not in transformed_images:
            transformed_img = cv2.warpAffine(image, M, (w_img, h_img), borderValue=BORDER_COLOR)
            transformed_images.append(img_fn)

        x, y, w, h = validator.batch['bboxes'][i]
        x1 = int((x - w / 2) * w_img)
        y1 = int((y - h / 2) * h_img)
        x2 = int((x + w / 2) * w_img)
        y2 = int((y + h / 2) * h_img)

        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        pts = np.concatenate([pts, np.ones((4, 1))], axis=1)
        pts = M.dot(pts.T).T

        x_min = int(min(pts[:, 0]))
        y_min = int(min(pts[:, 1]))
        x_max = int(max(pts[:, 0]))
        y_max = int(max(pts[:, 1]))

        x = (x_min + x_max) / 2
        y = (y_min + y_max) / 2

        # continue if the bounding box is outside the image (also for inverse)
        if x > w_img or y > h_img or x < 0 or y < 0:
            continue

        # adjust only if the transformation is not inverse, as adjustment is not reversible
        if not hasattr(validator, 'is_inverse'):
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            if x_max > w_img:
                x_max = w_img
            if y_max > h_img:
                y_max = h_img

            # adjust x and y, in case the bounding box is outside the image
            adjusted_x = (x_min + x_max) / 2 / w_img
            adjusted_y = (y_min + y_max) / 2 / h_img
            adjusted_w = (x_max - x_min) / w_img
            adjusted_h = (y_max - y_min) / h_img

            bboxes.append(torch.tensor([adjusted_x, adjusted_y, adjusted_w, adjusted_h]))
        else:
            bboxes.append(validator.batch['bboxes'][i])

        validator.batch['img'][batch_idx] = torch.tensor(transformed_img.transpose(2, 0, 1), dtype=torch.uint8)

        cls.append(validator.batch['cls'][i])
        batch_idxs.append(torch.tensor(batch_idx))

    if cls and bboxes and batch_idxs:
        validator.batch['cls'] = torch.stack(cls)
        validator.batch['bboxes'] = torch.stack(bboxes)
        validator.batch['batch_idx'] = torch.stack(batch_idxs)


def ofa(model_type, transformation, epochs, batch_size, lr, optimizer, device):
    model_type = model_type.replace('ofa', 'OFA') if 'ofa' in model_type else model_type.capitalize()
    patch_validator()

    if transformation == 'rotate':
        degrees = 180
        scale = 0
        val_range = range(0, 360)
    elif transformation == 'scale':
        degrees = 0
        scale = 0.8
        val_range = np.linspace(SCALE_MIN, SCALE_MAX, NUM_EVAL_SCALE_FACTORS)

    if 'OFA' in model_type:
        model = YOLO('yolov8s.pt' if 'large' in model_type else 'yolov8n.pt')
        model.train(
            data='udacity-car.yaml',
            epochs=epochs,
            degrees=degrees,
            translate=0,
            scale=scale,
            shear=0,
            perspective=0,
            mosaic=0,
            flipud=0,
            fliplr=0,
            erasing=0,
            lr0=lr,
            cos_lr=True,
            optimizer=optimizer,
            batch=batch_size,
            name=f'{model_type} {transformation.capitalize()}',
            device=device
        )
    elif model_type == 'Baseline':
        model = YOLO("yolov8n.pt")
        model.train(
            data='udacity-car.yaml',
            epochs=epochs,
            lr0=lr,
            cos_lr=True,
            optimizer=optimizer,
            batch=batch_size,
            name=f'Baseline {transformation.capitalize()}',
            augment=False,
            device=device
        )
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    model.add_callback('on_val_batch_start', transformation_fn)

    for val in val_range:
        str_val = f'{val:.3f}'
        print(f'================== {transformation.capitalize()}: {str_val} ==================')
        def setup_rotation(validator):
            ultralytics.transformation = transformation
            ultralytics.transformation_value = val

        model.add_callback('on_val_start', setup_rotation)
        results = model.val(data='udacity-car.yaml', name=f'{model_type} {transformation.capitalize()} {str_val} ', device=device)

        setattr(results, 'on_plot', None)
        joblib.dump(results, f'{results.save_dir}/results.joblib')


def inverse(orig_model_id, transformation, device):
    if not orig_model_id:
        raise ValueError('No wandb ID for Baseline model is given')

    def invert_factor(validator):
        if transformation == 'rotate':
            ultralytics.transformation_value = -ultralytics.transformation_value
        elif transformation == 'scale':
            ultralytics.transformation_value = 1 / ultralytics.transformation_value

    def def_inverse(validator):
        validator.is_inverse = True

    if transformation == 'rotate':
        val_range = range(0, 360)
    elif transformation == 'scale':
        val_range = np.linspace(SCALE_MIN, SCALE_MAX, NUM_EVAL_SCALE_FACTORS)

    run = wandb.Api().run(f'YOLOv8/{orig_model_id}')
    artifacts = run.logged_artifacts()
    model_fn = ''
    for a in artifacts:
        if a.type == 'model':
            model_fn = glob(a.download() + '/*.pt')[0]
            break
    model = YOLO(model_fn)

    model.add_callback('on_val_batch_start', def_inverse)
    model.add_callback('on_val_batch_start', transformation_fn)
    model.add_callback('on_val_batch_start', invert_factor)
    model.add_callback('on_val_batch_start', transformation_fn)

    patch_validator()

    for val in val_range:
        str_val = f'{val:.3f}'
        print(f'================== {transformation.capitalize()}: {str_val} ==================')
        def setup_transformation(validator):
            ultralytics.transformation = transformation
            ultralytics.transformation_value = val

        model.add_callback('on_val_start', setup_transformation)
        results = model.val(data='udacity-car.yaml', name=f'Inverse {transformation.capitalize()} {str_val} ', device=device)

        setattr(results, 'on_plot', None)
        joblib.dump(results, f'{results.save_dir}/results.joblib')


class SCN(nn.Module):
    def __init__(self, dimensions, device='cpu'):
        super(SCN, self).__init__()

        self.dimensions = dimensions
        self.hyper_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, dimensions),
            nn.Softmax(dim=0)
        )
        self.device = device
        self.dimensions = dimensions
        self.model_params = nn.ParameterDict()
        self.to(device)

    def forward(self, angle_deg):
        rad = torch.Tensor([np.radians(angle_deg)]).to(self.device)
        hyper_output = self.hyper_net(rad)

        # check outputs of hyper_net
        if wandb.run is not None:
            step = wandb.run.step
            log_dict = {str(i): hyper_output[i].item() for i in range(len(hyper_output))}
            log_dict['angle'] = angle_deg
            wandb.log(log_dict, step=step)

        return hyper_output


def scn(transformation, dimensions, epochs, batch_size, lr, optimizer, device):
    model = YOLO("yolov8n.pt")

    # monkey patch training methods
    patch_method('ultralytics.nn.modules.Conv', 'forward', modified_conv_forward)
    patch_method('ultralytics.nn.tasks.BaseModel', '_predict_once', shortened_predict_once)
    patch_method('ultralytics.engine.trainer.BaseTrainer', '_do_train', modified_do_train)
    # monkey patch, set fuse to True, so it will never be fused
    patch_method('ultralytics.nn.tasks.BaseModel', 'is_fused', lambda x=10: True)

    # TODO: für jetzt
    patch_method('ultralytics.utils.loss.v8DetectionLoss', '__call__', v8_loss__call__)

    patch_validator()

    def scn_setup(base):
        ultralytics.transformation = transformation
        ultralytics.transformation_value = 0

        base.model.model.requires_grad_(False)
        ultralytics.scn = SCN(dimensions, device)

        # TODO: use all Conv layer
        # TODO: val/cls_loss ist Inf, warum?
        for idx, m in enumerate(base.model.model):
            if type(m) == ultralytics.nn.modules.conv.Conv and idx == 19:
                for d in range(dimensions):
                    weight = torch.nn.parameter.Parameter(torch.empty_like(m.conv.weight))
                    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
                    ultralytics.scn.model_params[f'layer_{idx:03d}__{d:02d}'] = weight.to(device)

                    if m.conv.bias:
                        bias = torch.nn.parameter.Parameter(torch.empty_like(m.conv.bias))
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(bias, -bound, bound)
                        ultralytics.scn.model_params[f'bias_layer_{idx:03d}__{d:02d}'] = bias

                if hasattr(m, 'bn'):
                    m.bn.affine = False
                    m.bn.momentum = 0
                    m.bn.running_mean.fill_(0)
                    m.bn.running_var.fill_(1)
                    m.bn.weight.fill_(1)
                    m.bn.bias.fill_(0)

                del m.conv.weight
                if m.conv.bias:
                    del m.conv.bias
        # disable EMA, otherwise it will cause the program to crash, since I deleted weights and biases from m
        base.ema.enabled = False

        base.optimizer = type(base.optimizer)([
            {'params': ultralytics.scn.parameters(), 'initial_lr': lr}
        ])

    def scn_rotate(base):
        ultralytics.transformation_value = np.random.uniform(0, 360)

    def scn_save_params(base):
        print('Saving Configuration Network')
        torch.save(ultralytics.scn, base.wdir / 'SCN.pt')

    model.add_callback('on_train_start', scn_setup)
    model.add_callback('on_train_batch_start', scn_rotate)
    model.add_callback('on_train_batch_start', transformation_fn)
    model.add_callback('on_model_save', scn_save_params)

    model.add_callback('on_val_batch_start', scn_rotate)
    model.add_callback('on_val_batch_start', transformation_fn)

    model.train(
#        data='udacity-car.yaml', # TODO: remove for development, use coco128 instead
        data='coco128.yaml',
        epochs=epochs,
        lr0=lr,
        cos_lr=True,
        optimizer=optimizer,
        batch=batch_size,
        name=f'SCN {transformation.capitalize()}',
        augment=False,
        device=device,
        amp=False
    )

    model.clear_callback('on_val_batch_start')
    model.add_callback('on_val_batch_start', transformation_fn)

    for val in range(0, 360):
        print(f'================== {transformation.capitalize()}: {val} ==================')
        def setup_val_transformation_fixed(validator):
            ultralytics.transformation = transformation
            ultralytics.transformation_value = val

        model.add_callback('on_val_start', setup_val_transformation_fixed)
        results = model.val(data='udacity-car.yaml', name=f'SCN {transformation.capitalize()} {val} ', device=device)

        setattr(results, 'on_plot', None)
        joblib.dump(results, f'{results.save_dir}/results.joblib')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser('arguments for training')
    parser.add_argument('--epochs', type=int, default=100, help='epochs for training')
    parser.add_argument('--device', type=str, default=device, help='used device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use', choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'])
    parser.add_argument('--model_id', type=str, help='id of the original model (only for Inverse)')
    parser.add_argument('--D', type=int, default=3, help='dimension for SCN (only for SCN)')
    parser.add_argument('method', type=str, help='methods to use for training', choices=['baseline', 'ofa', 'ofa_large', 'inverse', 'scn'])
    parser.add_argument('transformation', type=str, help='methods to use for training', choices=['rotate', 'scale'])
    args = parser.parse_args()

    if args.method == 'inverse':
        inverse(args.model_id, args.transformation, device=args.device)
    elif args.method == 'scn':
        scn(args.transformation, args.D, args.epochs, args.batch_size, args.lr, args.optimizer, args.device)
    else:
        ofa(args.method, args.transformation, args.epochs, args.batch_size, args.lr, args.optimizer, args.device)
