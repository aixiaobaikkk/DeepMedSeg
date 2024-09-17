from loss import lovasz_hinge, BinaryOhemCrossEntropy
import numpy as np
from glob import glob
from tqdm import tqdm
import sys
sys.path.append('..')
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from medpy.metric.binary import hd, dc, assd, jc
from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import time
import logging
import os,  argparse, math
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from networks.LUCF_Net import LUCF_Net
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str,
                    default='LUCF_Net',
                    help='selected model,Swin_UNet,MISSFormer,LUCF_Net,HiFormer,DAEFormer,TransUNet,PVT_Cascade')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--in_chans', type=int,
                    default=3, help='input channel')
# parser.add_argument('--output_dir', type=str,
#                     default="model_pth/synapse/", help='output')
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
parser.add_argument('--z_spacing', type=int,
                    default=1, help='')
parser.add_argument('--dataset', type=str, default='isic2018')
parser.add_argument('--exp_name', type=str, default='test')
# parser.add_argument('--fold', type=str, default='1')
parser.add_argument('--lr_seg', type=float, default=0.025)  # 0.0003
parser.add_argument('--n_epochs', type=int, default=200)  # 100
parser.add_argument('--bt_size', type=int, default=4)  # 36
parser.add_argument('--seg_loss', type=int, default=1, choices=[0, 1])
parser.add_argument('--aug', type=int, default=1)
parser.add_argument('--patience', type=int, default=200)  # 50
parser.add_argument('--gpu', type=str, default='2')
# transformer
parser.add_argument('--filter', type=int, default=0)
parser.add_argument('--im_num', type=int, default=1)
parser.add_argument('--ex_num', type=int, default=1)
parser.add_argument('--xbound', type=int, default=1)
parser.add_argument('--point_w', type=float, default=1)
# log_dir name
parser.add_argument('--folder_name', type=str, default='Default_folder')
parse_config = parser.parse_args()
def get_cfg():
    print(parse_config)
    return parse_config
def fold0_4(fold_num):
    # -------------------------- train func --------------------------#
    def train(epoch):
        iou_loss = lovasz_hinge
        ohem_ce_loss = BinaryOhemCrossEntropy()
        model.train()
        iteration = 0
        for batch_idx, batch_data in enumerate(train_loader):

            data = batch_data['image'].cuda().float()
            label = batch_data['label'].cuda().float()
            p1, p2, p3, p4 = model(data)

            loss_iou1 = iou_loss(p1, label)
            loss_iou2 = iou_loss(p2, label)
            loss_iou3 = iou_loss(p3, label)
            loss_iou4 = iou_loss(p4, label)
            loss_ohem_ce1 = ohem_ce_loss(p1, label[:])
            loss_ohem_ce2 = ohem_ce_loss(p2, label[:])
            loss_ohem_ce3 = ohem_ce_loss(p3, label[:])
            loss_ohem_ce4 = ohem_ce_loss(p4, label[:])

            loss1 = 0.3 * loss_ohem_ce1 + 0.7 * loss_iou1
            loss2 = 0.3 * loss_ohem_ce2 + 0.7 * loss_iou2
            loss3 = 0.3 * loss_ohem_ce3 + 0.7 * loss_iou3
            loss4 = 0.3 * loss_ohem_ce4 + 0.7 * loss_iou4

            loss = loss2 + loss3 + loss4 + loss1

            lr = parse_config.lr_seg * (1.0 - epoch / parse_config.n_epochs) ** 0.9
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 10 == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\t[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'
                    .format(epoch, batch_idx * len(data),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss2.item(),
                            loss3.item(), loss4.item()))
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'
                    .format(epoch, batch_idx * len(data),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss2.item(),
                            loss3.item(), loss4.item()))

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
            with torch.no_grad():
                output, point_maps_pre, point_maps_pre1, point_maps_pre2 = model(data)
                output = output + point_maps_pre1 + point_maps_pre + point_maps_pre2
                loss = 0
                output = output.cpu().numpy() > 0.5

            label = label.cpu().numpy()
            assert (output.shape == label.shape)
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
        logging.info("Average dice value of evaluation dataset = ", dice_average)
        logging.info("Average iou value of evaluation dataset = ", iou_average)
        return dice_average, iou_average, loss

    print(torch.cuda.is_available())
    #-------------------------- get args --------------------------#
    parse_config = get_cfg()
    #-------------------------- build loggers and savers --------------------------#
    exp_name = parse_config.dataset + '/' + parse_config.exp_name + '_loss_' + str(
        parse_config.seg_loss) + '_aug_' + str(
            parse_config.aug
        ) + '/' + parse_config.folder_name + '/fold_' + str(fold_num)

    def logger_init(log_file_name='monitor', log_level=logging.DEBUG, log_dir=exp_name+'/'+'logs.txt', only_file=False):
        formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
        logging.basicConfig(filename=log_dir, level=log_level, format=formatter, datefmt='%Y-%d-%m %H:%M:%S')
    
    os.makedirs('logs/{}'.format(exp_name), exist_ok=True)
    os.makedirs('logs/{}/model'.format(exp_name), exist_ok=True)
    writer = SummaryWriter('logs/{}/log'.format(exp_name))
    save_path = 'logs/{}/model/best.pkl'.format(exp_name)
    latest_path = 'logs/{}/model/latest.pkl'.format(exp_name)

    EPOCHS = parse_config.n_epochs
    # os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu


    #-------------------------- build dataloaders --------------------------#
    if parse_config.dataset == 'isic2018':
        from utils.isbi2018_new import norm01, myDataset

        dataset = myDataset(fold=fold_num,
                            split='train',
                            aug=parse_config.aug)
        dataset2 = myDataset(fold=fold_num, split='valid', aug=False)
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
    model = LUCF_Net(3, 1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr_seg)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

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
        logging.info(
            'Training and evaluating on epoch:{} complete in {:.0f}m {:.0f}s'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))
        print(
            'Training and evaluating on epoch:{} complete in {:.0f}m {:.0f}s'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))
