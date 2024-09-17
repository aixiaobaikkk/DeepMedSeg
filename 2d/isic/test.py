from medpy.metric.binary import hd, hd95, dc, jc, assd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
import sys
sys.path.append('../')
import cv2
import matplotlib.pyplot as plt
from networks.LUCF_Net import LUCF_Net,LUCF_Net_unet
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_point_pred = False
target_size = (512, 512)

def isbi2016(model_name):
    dice_value = 0
    iou_value = 0
    numm = 0

    model = LUCF_Net(in_chns=3, class_num=1).to(device)
    # model.load_state_dict(torch.load(r'E:\LUCF-Net\SRC_ISIC\logs\isic2016\LUCF_Net\test_loss_1_aug_1\Default_folder\fold_None\new_dice\model\best.pkl'))
    model.load_state_dict(torch.load(r'/amax/users/Admin/demo/local_demo/LUCF_Net/code/SRC_ISIC/logs/exp4/isic2016/test_loss_1/from2018_2016/model/best.pkl'))

    # for fold in ['PH2', 'Test']:
    for fold in [ 'Test']:
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
            with torch.no_grad():
                p1, p2, p3, p4 = model(data)
                output = p1 + p2 + p3 + p4
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
        print("Average dice value of evaluation dataset = ", dice_average)
        print("Average iou value of evaluation dataset = ", iou_average)

            # output = torch.sigmoid(output)[0][0]
            # output = (output.cpu().numpy() > 0.5).astype('uint8')
            # output = (cv2.resize(output, target_size, cv2.INTER_NEAREST) >
            #           0.5) * 1
            # plt.imsave(
            #     save_dir + '/' + os.path.basename(path)[:-4] + '.png',
            #     output)




def isbi2018():
    dice_value = 0
    iou_value = 0
    numm = 0
    model = LUCF_Net(3, 1).cuda()
    for fold in range(5):
        model.load_state_dict(
            torch.load(
                f'/amax/users/Admin/demo/local_demo/LUCF_Net/code/SRC_ISIC/logs/isic2018/test_loss_1_aug_1/Default_folder/fold_{fold}/model/best.pkl'
            ))
        save_dir = f'results/ISIC-2018-pictures/LUCF_Net/fold-{int(fold)+1}'
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
                p1, p2, p3, p4 = model(data)
                output = p1 + p2 + p3 + p4
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
        print("Average dice value of evaluation dataset = ", dice_average)
        print("Average iou value of evaluation dataset = ", iou_average)
        output = torch.sigmoid(output)[0][0]
        output = (output.cpu().numpy() > 0.5).astype('uint8')
        output = (cv2.resize(output, target_size, cv2.INTER_NEAREST) >
                  0.5) * 1
        plt.imsave(
            save_dir + '/' + os.path.basename(path).split('_')[1][:-4] +
            '.png', output)



if __name__ == '__main__':
    isbi2016('LUCF')
    # isbi2018()

