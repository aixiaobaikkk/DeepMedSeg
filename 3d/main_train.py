import argparse  # 用于解析命令行参数
from monai.utils import set_determinism  # 设置确定性随机性，以便结果可复现
from monai.transforms import AsDiscrete  # 将输出结果转换为离散值
from networks.unet3d import unet_3D  # 自定义的3D U-Net模型
from networks.UXNet_3D.network_backbone import UXNET  # 自定义的UXNet 3D模型
from monai.networks.nets import UNETR, SwinUNETR, VNet  # MONAI中的一些现成3D网络模型
from networks.nnFormer.nnFormer_seg import nnFormer  # 自定义的nnFormer模型
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS  # 自定义的TransBTS模型
from monai.metrics import DiceMetric  # 用于计算Dice系数
from monai.losses import DiceCELoss  # Dice + 交叉熵损失函数
from monai.inferers import sliding_window_inference  # 用于滑动窗口推理
from monai.data import CacheDataset, DataLoader, decollate_batch  # 用于数据集加载和处理

import torch
from torch.utils.tensorboard import SummaryWriter  # 用于TensorBoard可视化
from load_datasets_transforms import data_loader, data_transforms  # 自定义的加载数据集和数据预处理模块
import os
import numpy as np
from tqdm import tqdm  # 进度条显示库

# 命令行参数设置
parser = argparse.ArgumentParser(description='3D UX-Net超参数设置用于医学图像分割')
parser.add_argument('--root', type=str,
                    default='/amax/home/Admin/work/paper_code/LUCF-Net-main/data/Subtask2_272_69_20/',
                    help='图像和标签的根目录')
parser.add_argument('--output', type=str, default='output_nnFormer', help='输出目录，用于保存tensorboard日志和最佳模型')
parser.add_argument('--dataset', type=str, default='flare', help='数据集名称: {feta, flare, amos}')
parser.add_argument('--network', type=str, default='nnFormer',
                    help='网络模型: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET, VNet, UNet3D}')
parser.add_argument('--mode', type=str, default='train', help='训练模式或测试模式')
parser.add_argument('--pretrain', default=False, help='是否加载预训练权重')
parser.add_argument('--pretrained_weights', default='', help='预训练模型的路径')
parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
parser.add_argument('--crop_sample', type=int, default=2, help='每个样本的裁剪子卷数量')
parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
parser.add_argument('--optim', type=str, default='AdamW', help='优化器类型: Adam或AdamW')
parser.add_argument('--max_iter', type=int, default=40000, help='最大训练迭代次数')
parser.add_argument('--eval_step', type=int, default=500, help='每多少步进行一次验证')
parser.add_argument('--gpu', type=str, default='1', help='GPU编号')
parser.add_argument('--cache_rate', type=float, default=1, help='缓存率，用于将数据集缓存到GPU')
parser.add_argument('--num_workers', type=int, default=2, help='加载数据时的工作线程数')

args = parser.parse_args()

# 设置要使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('使用的GPU: {}'.format(args.gpu))

# 加载数据集
train_samples, valid_samples, out_classes = data_loader(args)

# 将训练和验证数据集构建成字典列表
train_files = [{"image": image_name, "label": label_name} for image_name, label_name in
               zip(train_samples['images'], train_samples['labels'])]
val_files = [{"image": image_name, "label": label_name} for image_name, label_name in
             zip(valid_samples['images'], valid_samples['labels'])]

# 确保训练的确定性（即相同设置下，结果可重复）
set_determinism(seed=0)

# 数据预处理的变换
train_transforms, val_transforms = data_transforms(args)

# 使用缓存的数据集
print('开始缓存数据集！')
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=args.cache_rate,
                        num_workers=args.num_workers)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=True)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=args.cache_rate,
                      num_workers=args.num_workers)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

# 加载指定的网络模型
if args.network == '3DUXNET':
    model = UXNET(in_chans=1, out_chans=out_classes, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384],
                  drop_path_rate=0, layer_scale_init_value=1e-6, spatial_dims=3).cuda()
elif args.network == 'SwinUNETR':
    model = SwinUNETR(img_size=(96, 96, 96), in_channels=1, out_channels=out_classes, feature_size=48,
                      use_checkpoint=False).cuda()
elif args.network == 'nnFormer':
    model = nnFormer(input_channels=1, num_classes=out_classes).cuda()
elif args.network == 'UNETR':
    model = UNETR(in_channels=1, out_channels=out_classes, img_size=(96, 96, 96), feature_size=16, hidden_size=768,
                  mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True,
                  dropout_rate=0.0).cuda()
elif args.network == 'TransBTS':
    _, model = TransBTS(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.cuda()
elif args.network == 'UNet3D':
    model = unet_3D(n_classes=out_classes).cuda()
elif args.network == 'VNet':
    model = VNet(out_channels=out_classes).cuda()

print('选择的网络架构: {}'.format(args.network))

# 如果需要加载预训练模型，加载权重
if args.pretrain == 'True':
    print('找到预训练权重！开始从 {} 加载'.format(args.pretrained_weights))
    model.load_state_dict(torch.load(args.pretrained_weights))

# 定义损失函数和优化器
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
print('使用的损失函数: DiceCELoss')

# 根据选择的优化器类型，初始化优化器
if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

print('使用的优化器: {}, 学习率: {}'.format(args.optim, args.lr))

# 创建输出目录和TensorBoard日志目录
root_dir = os.path.join(args.output)
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

t_dir = os.path.join(root_dir, 'tensorboard')
if not os.path.exists(t_dir):
    os.makedirs(t_dir)

# 初始化TensorBoard
writer = SummaryWriter(log_dir=t_dir)


# 验证函数
def validation(epoch_iterator_val):
    model.eval()  # 设置模型为评估模式
    dice_vals = []  # 用于存储每个批次的Dice系数
    with torch.no_grad():  # 不进行梯度计算
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = batch["image"].cuda(), batch["label"].cuda()
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 2, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description("验证 (%d / %d 步) (dice=%2.5f)" % (step, 10.0, dice))
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)  # 计算平均Dice系数
    writer.add_scalar('验证Dice系数', mean_dice_val, global_step)
    return mean_dice_val


# 训练函数
def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()  # 设置模型为训练模式
    epoch_loss = 0  # 初始化损失
    epoch_iterator = tqdm(train_loader, desc="训练 (X / X 步) (损失=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        x, y = batch["image"].cuda(), batch["label"].cuda()
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()  # 反向传播
        epoch_loss += loss.item()  # 累积损失
        optimizer.step()  # 更新权重
        optimizer.zero_grad()  # 清空梯度
        epoch_iterator.set_description("训练 (%d / %d 步) (损失=%2.5f)" % (global_step, max_iterations, loss))

        # 每eval_num步或训练结束时进行验证
        if global_step % eval_num == 0 and global_step != 0 or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="验证 (X / X 步) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step  # 计算平均损失
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            # 保存最好的模型
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print("模型已保存！当前最优Dice: {} 当前Dice: {}".format(dice_val_best, dice_val))
            else:
                print("模型未保存！当前最优Dice: {} 当前Dice: {}".format(dice_val_best, dice_val))
        writer.add_scalar('训练Dice系数', loss.data, global_step)
        global_step += 1
    return global_step, dice_val_best, global_step_best


# 设置训练参数
max_iterations = args.max_iter
print('最大训练迭代次数: {}'.format(str(args.max_iter)))
eval_num = args.eval_step
post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

# 开始训练循环
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
