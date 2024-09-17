from medpy.metric.binary import hd, hd95, dc, jc, assd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
import sys
import cv2
import matplotlib.pyplot as plt
from other_networks.MISSFormer.MISSFormer import MISSFormer
from other_networks.SwinUnet.vision_transformer import SwinUnet
from other_networks.HiFormer.HiFormer import HiFormer
from other_networks.DAEFormer.DAEFormer import DAEFormer
from other_networks.TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
from other_networks.CASCADE.networks import PVT_CASCADE
from other_networks.FocalNet.main import FUnet
from other_networks.UNet.unet import UNet
from other_networks.UNet.res_unet import ResUnet
from networks.LUCF_Net import LUCF_Net
from other_networks.UNEXT.archs import UNext
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from other_networks.Xbound.xboundformer import _segm_pvtv2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_point_pred = False
target_size = (512, 512)

def isbi2016(model_name):

    if model_name == 'xboundformer':
        model = _segm_pvtv2(1, 2, 2, 1, 352).to(device)
        model.load_state_dict(
            torch.load(
                f'logs/isbi2016/test_loss_1_aug_1/{model_name}/fold_None/model/best.pkl'
            ))
    elif model_name == 'LUCF_Net':
        model = LUCF_Net(in_chns=3,class_num=1).to(device)
        model.load_state_dict(torch.load(r'E:\LUCF-Net\SRC_ISIC\logs\isic2016\LUCF_Net\test_loss_1_aug_1\Default_folder\fold_None\new_dice\model\best.pkl'))
    elif model_name == 'UNet':
        model = UNet(in_chns=3,class_num=1).to(device)
        model.load_state_dict(torch.load(
            r'E:\LUCF-Net\SRC_ISIC\logs\isic2016\UNet\test_loss_1_aug_1\Default_folder\fold_None\new_dice\model\best.pkl'))
    elif model_name == 'Swin_UNet':
        model = SwinUnet(in_chans=3,num_classes=1).to(device)
        model.load_state_dict(torch.load(
            r'best.pkl'))

    elif model_name == 'unetpp':
        model = ResUnet(in_chans=3, num_classes=1).to(device)
        model.load_state_dict(torch.load(
            r'E:\LUCF-Net\SRC_ISIC\logs\isic2016\unetpp\test_loss_1_aug_1\Default_folder\fold_None\new_dice\model\best.pkl'))

    elif model_name == 'HiFormer':
        model = HiFormer(n_classes=1, in_chans=3).to(device)
        model.load_state_dict(torch.load(
            r'E:\LUCF-Net\SRC_ISIC\logs\isic2016\HiFormer\test_loss_1_aug_1\Default_folder\fold_None\new_dice\model\best.pkl'))

    elif model_name == 'unext':
        model =UNext(num_classes=1,in_chans=3).to(device)
        model.load_state_dict(torch.load(
            r'E:\LUCF-Net\SRC_ISIC\logs\isic2016\unext\test_loss_1_aug_1\Default_folder\fold_None\new_dice\model\latest.pkl'))


    else:
        # TODO
        raise NotImplementedError

    for fold in ['PH2', 'Test']:
        save_dir = f'results/ISIC-2016-pictures/{model_name}/{fold}'
        os.makedirs(save_dir, exist_ok=True)
        from utils.isbi2016_new import norm01, myDataset
        if fold == 'PH2':
            dataset = myDataset(split='test', aug=False)
        else:
            dataset = myDataset(split='valid', aug=False)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        model.eval()
        for batch_idx, batch_data in tqdm(enumerate(test_loader)):
            data = batch_data['image'].to(device).float()
            label = batch_data['label'].to(device).float()
            path = batch_data['image_path'][0]
            if model_name == 'LUCF_Net':
                with torch.no_grad():
                    p1,p2,p3,p4 = model(data)
                    output = p1 + p2 + p3 + p4
            elif model_name == 'UNet':
                with torch.no_grad():
                    output = model(data)
            elif model_name == 'Swin_UNet':
                with torch.no_grad():
                    data = F.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
                    output = model(data)
            elif model_name == 'unetpp':
                with torch.no_grad():
                    output = model(data)
            elif model_name == 'HiFormer':
                with torch.no_grad():
                    data = F.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
                    output = model(data)
            elif model_name == 'unext':
                with torch.no_grad():
                    # data = F.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
                    output = model(data)

            # if save_point_pred:
            #     os.makedirs(save_dir.replace('pictures', 'point_maps'),
            #                 exist_ok=True)
            #     point_pred1 = F.interpolate(point_pred1[-1], target_size)
            #     point_pred1 = point_pred1.cpu().numpy()[0, 0]
            #     plt.imsave(
            #         save_dir.replace('pictures', 'point_maps') + '/' +
            #         os.path.basename(path)[:-4] + '.png', point_pred1)
            output = torch.sigmoid(output)[0][0]
            output = (output.cpu().numpy() > 0.5).astype('uint8')
            output = (cv2.resize(output, target_size, cv2.INTER_NEAREST) >
                      0.5) * 1


            plt.imsave(
                save_dir + '/' + os.path.basename(path)[:-4] + '.png',
                output)


def isbi2018():
    from networks.LUCF_Net import LUCF_Net
    model = LUCF_Net(3, 1).cuda()
    for fold in range(5):
        model.load_state_dict(
            torch.load(
                r'best.pkl'
            ))
        save_dir = f'results/ISIC-2018-pictures/xboundformer/fold-{int(fold)+1}'
        os.makedirs(save_dir, exist_ok=True)
        from utils.isbi2018_new import norm01, myDataset
        dataset = myDataset(fold=str(fold), split='valid', aug=False)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        model.eval()
        for batch_idx, batch_data in tqdm(enumerate(test_loader)):
            data = batch_data['image'].to(device).float()
            label = batch_data['label'].to(device).float()
            path = batch_data['image_path'][0]
            with torch.no_grad():
                output, _, _, _ = model(data)
            output = torch.sigmoid(output)[0][0]
            output = (output.cpu().numpy() > 0.5).astype('uint8')
            output = (cv2.resize(output, target_size, cv2.INTER_NEAREST) >
                      0.5) * 1
            plt.imsave(
                save_dir + '/' + os.path.basename(path).split('_')[1][:-4] +
                '.png', output)


def isbi2018_ablation(folder_name):
    vs = list(map(int, folder_name.split('_')[1:]))
    # from networks.unet_lgl import UNet_lgl
    model = LUCF_Net(3, 1).cuda()
    target_size = (512, 512)
    for fold in range(5):
        model.load_state_dict(
            torch.load(
                r'best.pkl'
            ))
        save_dir = f'results/ISIC-2018-pictures/{folder_name}/fold-{int(fold)+1}'
        os.makedirs(save_dir, exist_ok=True)
        from utils.isbi2018_new import norm01, myDataset
        dataset = myDataset(fold=str(fold), split='valid', aug=False)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        model.eval()
        for batch_idx, batch_data in tqdm(enumerate(test_loader)):
            data = batch_data['image'].to(device).float()
            label = batch_data['label'].to(device).float()
            path = batch_data['image_path'][0]
            with torch.no_grad():
                output, _, _, _ = model(data)
            output = torch.sigmoid(output)[0][0]
            output = (output.cpu().numpy() > 0.5).astype('uint8')
            output = (cv2.resize(output, target_size, cv2.INTER_NEAREST) >
                      0.5) * 1
            plt.imsave(
                save_dir + '/' + os.path.basename(path).split('_')[1][:-4] +
                '.png', output)


if __name__ == '__main__':
    isbi2016(model_name="LUCF_Net")

