import os, argparse, math
import numpy as np
from glob import glob
from tqdm import tqdm
import sys
sys.path.append('E:/LUCF-Net')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from medpy.metric.binary import hd, dc, assd, jc
from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from loss import lovasz_hinge, BinaryOhemCrossEntropy
import time
from loss import lovasz_hinge, BinaryOhemCrossEntropy
from other_networks.MISSFormer.MISSFormer import MISSFormer
from other_networks.SwinUnet.vision_transformer import SwinUnet
from other_networks.HiFormer.HiFormer import HiFormer
from other_networks.DAEFormer.DAEFormer import DAEFormer
from other_networks.TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
from other_networks.CASCADE.networks import PVT_CASCADE
from other_networks.FocalNet.main import FUnet
from other_networks.UNet.unet import UNet
from other_networks.UNet.unetpp import NestedUNet
from other_networks.UNet.res_unet import ResUnet
from other_networks.UNEXT.archs import UNext


def get_cfg():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--model_name', type=str,
    #                     default='HiFormer',
    #                     help='selected model,Swin_UNet,MISSFormer,LUCF_Net,HiFormer,DAEFormer,TransUNet,PVT_Cascade')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parser.add_argument('--in_chans', type=int,
                        default=3, help='input channel')
    parser.add_argument('--output_dir', type=str,
                        default="model_pth/synapse/", help='output')
    parser.add_argument('--is_pretrained', type=bool,
                        default=False, help='whether loading pretrained weights')
    parser.add_argument('--pretrained_pth', type=str,
                        default=r'',
                        help='pretrained model weights')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='save model interval')
    parser.add_argument('--seed', type=int,
                        default=2222, help='random seed')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--z_spacing', type=int, default=1, help='')
    parser.add_argument('--dataset', type=str, default='isic2016')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--fold', type=str, default='None')
    parser.add_argument('--lr_seg', type=float, default=0.001)  # 0.0003
    parser.add_argument('--n_epochs', type=int, default=300)  # 100
    parser.add_argument('--bt_size', type=int, default=4)  # 36
    parser.add_argument('--seg_loss', type=int, default=1, choices=[0, 1])
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--patience', type=int, default=500)  # 50
    parser.add_argument('--gpu', type=str, default='1')
    # transformer
    parser.add_argument('--filter', type=int, default=0)
    parser.add_argument('--im_num', type=int, default=1)
    parser.add_argument('--ex_num', type=int, default=1)
    parser.add_argument('--xbound', type=int, default=1)
    parser.add_argument('--point_w', type=float, default=1)

    # log_dir name
    parser.add_argument('--folder_name', type=str, default='Default_folder')

    parse_config = parser.parse_args()
    print(parse_config)
    return parse_config


def train_package(model_name):




    # -------------------------- train func --------------------------#
    def train(epoch):
        iou_loss = lovasz_hinge
        ohem_ce_loss = BinaryOhemCrossEntropy()
        model.train()
        dice_loss = SoftDiceLoss()
        iteration = 0
        for batch_idx, batch_data in enumerate(train_loader):
            #         print(epoch, batch_idx)
            data = batch_data['image'].cuda().float()
            label = batch_data['label'].cuda().float()
            if parse_config.filter:
                point = (batch_data['filter_point_data'] > 0).cuda().float()
            else:
                point = (batch_data['point'] > 0).cuda().float()
            # point_All = (batch_data['point_All'] > 0).cuda().float()

            if model_name == 'LUCF_Net':

                lateral_map_4, lateral_map_3, lateral_map_2, p = model(data)
                loss4 = 0.3 * structure_loss(lateral_map_4, label) + 0.7 * dice_loss(lateral_map_4, label)
                loss3 = 0.3 * structure_loss(lateral_map_3, label) + 0.7 * dice_loss(lateral_map_3, label)
                loss2 = 0.3 * structure_loss(lateral_map_2, label) + 0.7 * dice_loss(lateral_map_2, label)
                loss1 = 0.3 * structure_loss(p, label) + 0.7 * dice_loss(p, label)

                loss = loss2 + loss3 + loss4 + loss1





            lr = parse_config.lr_seg * (1.0 - epoch / parse_config.n_epochs) ** 0.9
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (batch_idx + 1) % 10 == 0:
            #     print(
            #         'Train Epoch: {} [{}/{} ({:.0f}%)]\t[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'
            #         .format(epoch, batch_idx * len(data),
            #                 len(train_loader.dataset),
            #                 100. * batch_idx / len(train_loader), loss2.item(),
            #                 loss3.item(), loss4.item()))

            if (batch_idx + 1) % 10 == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                    .format(epoch, batch_idx * len(data),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader)))

        print("Iteration numbers: ", iteration)

    # -------------------------- eval func --------------------------#
    def evaluation(epoch, loader):
        model.eval()
        dice_value = 0
        iou_value = 0
        dice_average = 0
        iou_average = 0
        numm = 0
        for batch_idx, batch_data in enumerate(loader):
            data = batch_data['image'].cuda().float()
            label = batch_data['label'].cuda().float()
            point = (batch_data['point'] > 0).cuda().float()
            # point_All = (batch_data['point_data'] > 0).cuda().float()
            # point_All = nn.functional.max_pool2d(point_All,
            #                                 kernel_size=(16, 16),
            #                                 stride=(16, 16))

            with torch.no_grad():
                if model_name == 'transfuse':
                    _, _, output = model(data)
                    loss_fuse = structure_loss(output, label)
                    print('mistake')
                elif model_name == 'xboundformer':
                    output, point_maps_pre, point_maps_pre1, point_maps_pre2 = model(data)
                    output = output + point_maps_pre1 + point_maps_pre + point_maps_pre2
                    loss = 0
                elif model_name == 'LUCF_Net':
                    output, point_maps_pre, point_maps_pre1, point_maps_pre2 = model(data)
                    output = output + point_maps_pre1 + point_maps_pre + point_maps_pre2
                    loss = 0
                elif  model_name == 'unetpp':
                    # data = F.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
                    output = model(data)
                    loss = 0
                else:
                    data = F.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
                    output = model(data)
                    label = F.interpolate(label, size=(224, 224), mode='bilinear', align_corners=False)

                    loss = 0

                if model_name == 'transfuse':
                    loss = loss_fuse

                output = output.cpu().numpy() > 0.5

            label = label.cpu().numpy()
            assert (output.shape == label.shape)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            dice_ave = dc(output, label)
            iou_ave = jc(output, label)
            dice_value += dice_ave
            iou_value += iou_ave
            numm += 1

        dice_average = dice_value / numm
        iou_average = iou_value / numm
        writer.add_scalar('val_metrics/val_dice', dice_average, epoch)
        writer.add_scalar('val_metrics/val_iou', iou_average, epoch)
        print("Average dice value of evaluation dataset = ", dice_average)
        print("Average iou value of evaluation dataset = ", iou_average)
        return dice_average, iou_average, loss



















    print(torch.cuda.is_available())

    CUDA_VISIBLE_DEVICES = 1
    #-------------------------- get args --------------------------#
    parse_config = get_cfg()

    #-------------------------- build loggers and savers --------------------------#
    exp_name = parse_config.dataset+'/'+model_name + '/' + parse_config.exp_name + '_loss_' + str(
        parse_config.seg_loss) + '_aug_' + str(
            parse_config.aug
        ) + '/' + parse_config.folder_name + '/fold_' + str(parse_config.fold)+'/new_dice'

    os.makedirs('logs/{}'.format(exp_name), exist_ok=True)
    os.makedirs('logs/{}/model'.format(exp_name), exist_ok=True)
    writer = SummaryWriter('logs/{}/log'.format(exp_name))
    save_path = 'logs/{}/model/best.pkl'.format(exp_name)
    latest_path = 'logs/{}/model/latest.pkl'.format(exp_name)

    EPOCHS = parse_config.n_epochs
    os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu
    device_ids = range(torch.cuda.device_count())


    #-------------------------- build dataloaders --------------------------#
    if parse_config.dataset == 'isic2018':
        from utils.isbi2018_new import norm01, myDataset

        dataset = myDataset(fold=parse_config.fold,
                            split='train',
                            aug=parse_config.aug)
        dataset2 = myDataset(fold=parse_config.fold, split='valid', aug=False)
    elif parse_config.dataset == 'isic2016':
        from utils.isbi2016_new import norm01, myDataset

        dataset = myDataset(split='train', aug=parse_config.aug)
        dataset2 = myDataset(split='valid', aug=False)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=parse_config.bt_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset2,
        batch_size=1,  #parse_config.bt_size
        shuffle=False,  #True
        num_workers=2,
        pin_memory=True,
        drop_last=False)  #True

    #-------------------------- build models --------------------------#
    if model_name == 'LUCF_Net':
        from networks.LUCF_Net import LUCF_Net
        model = LUCF_Net(3,1).cuda()
        # model.load_state_dict(torch.load(r"E:\all demo\xboundformer-main\src\logs\isic2018\test_loss_1_aug_1\Default_folder\fold_4\model\best.pkl"))
        # model = _segm_pvtv2(1, parse_config.im_num, parse_config.ex_num,
        #                     parse_config.xbound, 352).cuda()
    elif model_name == 'xbound':
        from other_networks.Xbound.xboundformer import _segm_pvtv2

        model = _segm_pvtv2(1, parse_config.im_num, parse_config.ex_num,
                            parse_config.xbound, 352).cuda()

    elif model_name == 'Swin_UNet':
        model = SwinUnet(in_chans=parse_config.in_chans,num_classes=parse_config.num_classes).cuda()#in other networks  to be continue
    elif model_name == 'MISSFormer':
        model = MISSFormer().cuda()
    elif model_name == 'HiFormer':
        model = HiFormer(in_chans=3).cuda()
    elif model_name == 'DAEFormer':
        model = DAEFormer().cuda()
    elif model_name == 'TransUNet':
        model = ViT_seg(img_size=224, num_classes=parse_config.num_classes).cuda()
    elif model_name == 'PVT_Cascade':
        model = PVT_CASCADE().cuda()
    elif model_name == 'FocalUNet':
        model = FUnet().cuda()
    elif model_name == 'UNet':
        model = UNet(in_chns=3,class_num=1).cuda()
    elif model_name == 'unetpp':
        model = NestedUNet(in_ch=3,out_ch=1).cuda()
    elif model_name == 'ResUNet':
        model = ResUnet(in_chans=3,num_classes=1).cuda()
    elif model_name == 'unext':
        model = UNext(in_chans=3,num_classes=1).cuda()

    print(len(device_ids))
    if len(device_ids) > 1:  # 多卡训练
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr_seg)

    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

    criteon = [None, ce_loss][parse_config.seg_loss]

    #-------------------------- start training --------------------------#

    max_dice = 0
    max_iou = 0
    best_ep = 0

    min_loss = 10
    min_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        start = time.time()

        dice, iou, loss = evaluation(epoch, val_loader)
        train(epoch)
        scheduler.step()

        if loss < min_loss:
            min_epoch = epoch
            min_loss = loss
        else:
            if epoch - min_epoch >= parse_config.patience:
                print('Early stopping!')
                break
        if iou > max_iou:
            max_iou = iou
            best_ep = epoch
            torch.save(model.state_dict(), save_path)
        else:
            if epoch - best_ep >= parse_config.patience:
                print('Early stopping!')
                break
        torch.save(model.state_dict(), latest_path)
        time_elapsed = time.time() - start
        print(
            'Training and evaluating on epoch:{} complete in {:.0f}m {:.0f}s'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))
