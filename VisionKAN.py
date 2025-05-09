import math
import time
import torch
import datetime
import numpy as np
import torchvision
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as transforms
from typing import *
from functools import partial
from timm.utils import accuracy
from timm.models.layers import DropPath
from collections import defaultdict, deque
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from timm.models.vision_transformer import VisionTransformer, _cfg, Block, Attention


__all__KAN = ['deit_base_patch16_224_KAN', 'deit_small_patch16_224_KAN', 'deit_tiny_patch16_224_KAN']
__all__ViT = ['deit_base_patch16_224_ViT', 'deit_small_patch16_224_ViT', 'deit_tiny_patch16_224_ViT']


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=False,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
    
    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = (
            self.grid
        )
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()
    
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(
            0, 1
        )
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(
            A, B
        ).solution
        result = solution.permute(
            2, 0, 1
        )
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()
    
    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )
    
    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output
    
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
    
    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

class kanBlock(Block):
    def __init__(
        self,
        dim,
        num_heads=8,
        hdim_kan=192,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__(dim, num_heads)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.kan = KAN([dim, hdim_kan, dim])

    def forward(self, x):
        b, t, d = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.kan(self.norm2(x).reshape(-1, x.shape[-1])).reshape(b, t, d))
        return x

class VisionKAN(VisionTransformer):
    def __init__(self, *args, num_heads=8, batch_size=16, **kwargs):
        if 'hdim_kan' in kwargs:
            self.hdim_kan = kwargs['hdim_kan']
            del kwargs['hdim_kan']
        else:
            self.hdim_kan = 192
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        try:
            self.depth
        except AttributeError:
            if 'depth' in kwargs:
                self.depth = kwargs['depth']
            else:
                self.depth = 12
        block_list = [
            kanBlock(dim=self.embed_dim, num_heads=self.num_heads, hdim_kan=self.hdim_kan)
            for i in range(self.depth)
        ]
        if type(self.blocks) == nn.Sequential:
            self.blocks = nn.Sequential(*block_list)
        elif type(self.blocks) == nn.ModuleList:
            self.blocks = nn.ModuleList(block_list)

def create_kan(model_name, pretrained, **kwargs):
    if model_name == 'deit_tiny_patch16_224_KAN':
        model = VisionKAN(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model
    elif model_name == 'deit_small_patch16_224_KAN':
        model = VisionKAN(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model
    elif model_name == 'deit_base_patch16_224_KAN':
        model = VisionKAN(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

def create_ViT(model_name, pretrained, **kwargs):
    if 'batch_size' in kwargs:
        del kwargs['batch_size']
    if model_name == 'deit_tiny_patch16_224_ViT':
        model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model
    elif model_name == 'deit_small_patch16_224_ViT':
        model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model
    elif model_name == 'deit_base_patch16_224_ViT':
        model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

def create_model(model_name,**kwargs):
    pretrained = kwargs['pretrained'] if 'pretrained' in kwargs else False
    if 'pretrained' in kwargs:
        del kwargs['pretrained']
    print(kwargs)
    if model_name in __all__KAN:
        model = create_kan(model_name, pretrained, **kwargs)
        model.default_cfg = _cfg()
        return model
    elif model_name in __all__ViT:
        model = create_ViT(model_name, pretrained, **kwargs)
        model.default_cfg = _cfg()
        return model
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
    
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()
    
    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def global_avg(self):
        return self.total / self.count
    
    @property
    def max(self):
        return max(self.deque)
    
    @property
    def value(self):
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def plot_metrics(
        loss_history_kan,
        loss_history_vit,
        y_true_kan,
        y_pred_kan,
        y_scores_kan,
        y_true_vit,
        y_pred_vit,
        y_scores_vit,
        num_classes=10
    ):
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history_kan, color='red', label='KAN Model')
    plt.plot(loss_history_vit, color='blue', label='ViT Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.figure(figsize=(16, 5))
    cm_kan = confusion_matrix(y_true_kan, y_pred_kan)
    plt.subplot(2, 1, 1)
    sns.heatmap(cm_kan, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title('KAN Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.figure(figsize=(16, 5))
    cm_vit = confusion_matrix(y_true_vit, y_pred_vit)
    plt.subplot(2, 1, 2)
    sns.heatmap(cm_vit, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title('ViT Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve((y_true_kan == i).astype(int), y_scores_kan[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'class {i} ROC Curve (area = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('KAN ROC Curve')
    plt.legend(loc="lower right")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 2)
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve((y_true_vit == i).astype(int), y_scores_vit[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'class {i} ROC Curve (area = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ViT ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    all_targets, all_predictions, all_scores = [], [], []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(output.argmax(dim=1).cpu().numpy())
            all_scores.extend(output.softmax(dim=1).cpu().numpy())
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    print('* top1准确率 {top1.global_avg:.3f} top5准确率 {top5.global_avg:.3f} 平均损失 {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    print(f'精确率: {precision:.4f}, 召还率: {recall:.4f}, F1: {f1:.4f}')
    return all_targets, all_predictions, all_scores, loss.item()

def main():
    KAN_model = create_model(
        model_name='deit_tiny_patch16_224_KAN',
        pretrained=False,
        hdim_kan=192,
        num_classes=10,
        drop_rate=0.0,
        drop_path_rate=0.05,
        img_size=32,
        batch_size=16
    )
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    optimizer = optim.SGD(KAN_model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    KAN_model.to(device)
    y_true_kan, y_pred_kan, y_scores_kan = [], [], []
    loss_values_kan = []
    for epoch in range(50):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = KAN_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            loss_values_kan.append(running_loss / len(trainloader))
    y_true_kan, y_pred_kan, y_scores_kan, loss_kan = evaluate(testloader, KAN_model, device=device)
    print(f'KAN的交叉熵损失: {loss_kan:.4f}')
    print('Finished Training KAN')
    ViT_model = create_model(
        model_name='deit_tiny_patch16_224_ViT',
        pretrained=False,
        num_classes=10,
        drop_rate=0.0,
        drop_path_rate=0.05,
        img_size=32,
        batch_size=16
    )
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    optimizer = optim.SGD(ViT_model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ViT_model.to(device)
    y_true_vit, y_pred_vit, y_scores_vit = [], [], []
    loss_values_vit = []
    for epoch in range(50):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = ViT_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            loss_values_kan.append(running_loss / len(trainloader))
    y_true_vit, y_pred_vit, y_scores_vit, loss_vit = evaluate(testloader, ViT_model, device=device)
    print(f'ViT的交叉熵损失: {loss_vit:.4f}')
    print('Finished Training ViT')
    plot_metrics(loss_values_kan, loss_values_vit, np.array(y_true_kan), np.array(y_pred_kan), np.array(y_scores_kan),
                 np.array(y_true_vit), np.array(y_pred_vit), np.array(y_scores_vit))


if __name__ == '__main__':
    main()