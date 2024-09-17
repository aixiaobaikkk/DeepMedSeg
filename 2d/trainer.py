import random
from tensorboardX import SummaryWriter
from torchvision import transforms
from utils.dataset_2d import Synapse_dataset, RandomGenerator
from utils.loss import DiceLoss, LovaszSoftmax, OhemCrossEntropy
from utils.utils import val_single_volume
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from utils.dataset_ACDC import ACDCdataset, RandomGenerator
from utils.utils import test_single_volume
import numpy as np
from tqdm import tqdm
import sys
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from medpy.metric.binary import hd, dc, assd, jc
from utils.loss import lovasz_hinge, BinaryOhemCrossEntropy
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import time
import logging
from utils.isbi2016_new import norm01, myDataset
# --------------------------- multi organs trainer ---------------------------

# 推理函数，用于在测试集上验证模型性能
def inference_organ(args, model, best_performance):
    # 创建测试集数据加载器
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir,
                              nclass=args.num_classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)
    logging.info("{} test iterations per epoch".format(len(testloader)))

    # 设置模型为评估模式
    model.eval()
    metric_list = 0.0

    # 遍历测试集，逐个计算性能指标
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]  # 获取图像尺寸
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        # 调用验证函数计算当前图像的Dice系数
        metric_i = val_single_volume(image, label, model, classes=args.num_classes,
                                     patch_size=[args.img_size, args.img_size],
                                     case=case_name, z_spacing=args.z_spacing, model_name=args.model_name)
        metric_list += np.array(metric_i)

    # 计算所有测试样本的平均性能指标
    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)

    logging.info('测试模型的性能: 平均Dice系数 : %f, 最佳Dice系数 : %f' % (performance, best_performance))

    return performance


# 训练函数，用于训练模型
def trainer_organs(args, model, snapshot_path):

    # Set up logging
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Set learning rate, number of classes, and batch size
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size

    # Create the training dataset loader
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("Number of samples in the training set: {}".format(len(db_train)))

    # Initialize the random seed for the data loader
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    # Use data parallel mode if using multiple GPUs
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()  # Set model to training mode

    # Define various loss functions
    iou_loss = LovaszSoftmax()
    dice_loss = DiceLoss(num_classes)
    ce_loss = nn.CrossEntropyLoss()
    ohem_ce_loss = OhemCrossEntropy()

    # Use SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # Use TensorBoard to log training
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations.".format(len(trainloader), max_iterations))

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    # Start training loop
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # Select different forward and loss calculation methods based on the model name
            if args.model_name == 'LUCF_Net':
                p1, p2, p3, p4 = model(image_batch)

                loss_iou1 = iou_loss(p1, label_batch)
                loss_iou2 = iou_loss(p2, label_batch)
                loss_iou3 = iou_loss(p3, label_batch)
                loss_iou4 = iou_loss(p4, label_batch)

                loss_ohem_ce1 = ohem_ce_loss(p1, label_batch[:].long())
                loss_ohem_ce2 = ohem_ce_loss(p2, label_batch[:].long())
                loss_ohem_ce3 = ohem_ce_loss(p3, label_batch[:].long())
                loss_ohem_ce4 = ohem_ce_loss(p4, label_batch[:].long())

                loss_p1 = 0.2 * loss_ohem_ce1 + 0.8 * loss_iou1
                loss_p2 = 0.2 * loss_ohem_ce2 + 0.8 * loss_iou2
                loss_p3 = 0.2 * loss_ohem_ce3 + 0.8 * loss_iou3
                loss_p4 = 0.2 * loss_ohem_ce4 + 0.8 * loss_iou4

                alpha, beta, gamma, zeta = 1., 1., 1., 1.
                loss = alpha * loss_p1 + beta * loss_p2 + gamma * loss_p3 + zeta * loss_p4
            elif args.model_name == 'HiFormer':
                B, C, H, W = image_batch.shape
                image_batch = image_batch.expand(B, 3, H, W)
                outputs = model(image_batch)
                loss = 0.2 * ohem_ce_loss(outputs, label_batch) + 0.8 * iou_loss(outputs, label_batch)
            elif args.model_name == 'PVT_Cascade':

                p1, p2, p3, p4 = model(image_batch)  # Forward pass
                outputs = p1 + p2 + p3 + p4  # Sum of outputs
                loss_iou1 = iou_loss(p1, label_batch)
                loss_iou2 = iou_loss(p2, label_batch)
                loss_iou3 = iou_loss(p3, label_batch)
                loss_iou4 = iou_loss(p4, label_batch)

                loss_ohem_ce1 = ohem_ce_loss(p1, label_batch[:].long())
                loss_ohem_ce2 = ohem_ce_loss(p2, label_batch[:].long())
                loss_ohem_ce3 = ohem_ce_loss(p3, label_batch[:].long())
                loss_ohem_ce4 = ohem_ce_loss(p4, label_batch[:].long())

                loss_p1 = 0.2 * loss_ohem_ce1 + 0.8 * loss_iou1
                loss_p2 = 0.2 * loss_ohem_ce2 + 0.8 * loss_iou2
                loss_p3 = 0.2 * loss_ohem_ce3 + 0.8 * loss_iou3
                loss_p4 = 0.2 * loss_ohem_ce4 + 0.8 * loss_iou4
                alpha, beta, gamma, zeta = 1., 1., 1., 1.
                loss = alpha * loss_p1 + beta * loss_p2 + gamma * loss_p3 + zeta * loss_p4
            else:
                outputs = model(image_batch)
                loss = 0.2 * ohem_ce_loss(outputs, label_batch) + 0.8 * iou_loss(outputs, label_batch)

            # Optimizer update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Adjust learning rate dynamically
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

        logging.info(
            'Epoch %d: Loss: %f, Learning Rate: %f, Best Dice Coefficient: %f' % (epoch_num, loss.item(), lr_, best_performance))

        # Validate the model and save
        if (epoch_num >= 150) and (epoch_num % args.save_interval) == 0:
            performance = inference_organ(args, model, best_performance)
            model.train()
        else:
            performance = 0.0

        # Save the model if new validation performance is better than the current best
        if (best_performance <= performance) and (epoch_num >= 150):
            best_performance = performance
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Model saved to {}".format(save_mode_path))

        # Save the model periodically
        if (epoch_num + 1) % args.save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Model saved to {}".format(save_mode_path))

        # Save the final model and stop training when reaching the maximum epoch
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Model saved to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training completed!"



# --------------------------- ACDC trainer ---------------------------
def inference(args, model, testloader, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing,model_name=args.model_name)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        mean_jacard = np.mean(metric_list, axis=0)[2]
        mean_asd = np.mean(metric_list, axis=0)[3]
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
        logging.info("Testing Finished!")
        return performance, mean_hd95, mean_jacard, mean_asd

def trainer_ACDC(args, net, snapshot_path):
    train_dataset = ACDCdataset(args.root_path, args.list_dir, split="train", transform = transforms.Compose(
        [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(train_dataset)))
    Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    db_val = ACDCdataset(base_dir=args.root_path, list_dir=args.list_dir, split="valid")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False)
    db_test = ACDCdataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)

    if args.n_gpu > 1:
        net = nn.DataParallel(net)

    net = net.cuda()
    net.train()
    ce_loss = nn.CrossEntropyLoss()
    iou_loss = LovaszSoftmax()
    dice_loss = DiceLoss(args.num_classes)
    ohem_ce_loss = OhemCrossEntropy()
    save_interval = args.n_skip

    iterator = tqdm(range(0, args.max_epochs), ncols=70)
    iter_num = 0

    Loss = []
    Test_Accuracy = []

    Best_dcs = 0.85
    best_test_dcs = 0.0

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    max_iterations = args.max_epochs * len(Train_loader)
    base_lr = args.base_lr
    # optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0001)#optional
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    def val():
        logging.info("Validation ===>")
        dc_sum = 0
        metric_list = 0.0
        net.eval()
        for i, val_sampled_batch in enumerate(valloader):
            val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
            val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(
                torch.FloatTensor)
            val_image_batch, val_label_batch = val_image_batch.cuda().unsqueeze(1), val_label_batch.cuda().unsqueeze(1)

            if args.model_name == 'LUCF_Net' or args.model_name == 'PVT_Cascade' or args.model_name == 'Swin_cat':
                p1, p2, p3, p4, = net(val_image_batch)
                val_outputs = p1 + p2 + p3 + p4
            else:
                val_outputs = net(val_image_batch)
            val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)

            dc_sum += dc(val_outputs.cpu().data.numpy(), val_label_batch[:].cpu().data.numpy())
        performance = dc_sum / len(valloader)

        return performance

    for epoch in iterator:
        net.train()
        train_loss = 0
        for i_batch, sampled_batch in enumerate(Train_loader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            if args.model_name == 'LUCF_Net':
                p1, p2, p3, p4 = net(image_batch)
                outputs = p1 + p2 + p3 + p4

                loss_iou1 = iou_loss(p1, label_batch)
                loss_iou2 = iou_loss(p2, label_batch)
                loss_iou3 = iou_loss(p3, label_batch)
                loss_iou4 = iou_loss(p4, label_batch)

                loss_ohem_ce1 = ohem_ce_loss(p1, label_batch[:].long())
                loss_ohem_ce2 = ohem_ce_loss(p2, label_batch[:].long())
                loss_ohem_ce3 = ohem_ce_loss(p3, label_batch[:].long())
                loss_ohem_ce4 = ohem_ce_loss(p4, label_batch[:].long())

                loss_p1 = 0.2 * loss_ohem_ce1 + 0.8 * loss_iou1
                loss_p2 = 0.2 * loss_ohem_ce2 + 0.8 * loss_iou2
                loss_p3 = 0.2 * loss_ohem_ce3 + 0.8 * loss_iou3
                loss_p4 = 0.2 * loss_ohem_ce4 + 0.8 * loss_iou4

                alpha, beta, gamma, zeta = 1., 1., 1., 1.
                loss = alpha * loss_p1 + beta * loss_p2 + gamma * loss_p3 + zeta * loss_p4
            elif args.model_name == 'HiFormer':
                B, C, H, W = image_batch.shape
                image_batch = image_batch.expand(B, 3, H, W)
                outputs = net(image_batch)
                loss = 0.2 * ohem_ce_loss(outputs, label_batch) + 0.8 * iou_loss(outputs, label_batch)
            elif args.model_name == 'PVT_Cascade':
                ce_loss = nn.CrossEntropyLoss()
                p1, p2, p3, p4 = net(image_batch)  # forward

                outputs = p1 + p2 + p3 + p4  # additive output aggregation

                loss_iou1 = iou_loss(p1, label_batch)
                loss_iou2 = iou_loss(p2, label_batch)
                loss_iou3 = iou_loss(p3, label_batch)
                loss_iou4 = iou_loss(p4, label_batch)

                loss_ohem_ce1 = ohem_ce_loss(p1, label_batch[:].long())
                loss_ohem_ce2 = ohem_ce_loss(p2, label_batch[:].long())
                loss_ohem_ce3 = ohem_ce_loss(p3, label_batch[:].long())
                loss_ohem_ce4 = ohem_ce_loss(p4, label_batch[:].long())

                loss_p1 = 0.2 * loss_ohem_ce1 + 0.8 * loss_iou1
                loss_p2 = 0.2 * loss_ohem_ce2 + 0.8 * loss_iou2
                loss_p3 = 0.2 * loss_ohem_ce3 + 0.8 * loss_iou3
                loss_p4 = 0.2 * loss_ohem_ce4 + 0.8 * loss_iou4

                alpha, beta, gamma, zeta = 1., 1., 1., 1.
                loss = alpha * loss_p1 + beta * loss_p2 + gamma * loss_p3 + zeta * loss_p4  # current setting is for additive aggregation.
            else:
                outputs = net(image_batch)
                loss = 0.2 * ohem_ce_loss(outputs, label_batch[:].long()) + 0.8 *iou_loss(outputs, label_batch,
                                                                                       softmax=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            if iter_num % 55 == 0:
                # logging.info('iteration %d : loss : %f lr_: %f,best_test_dcs : %f' % (iter_num, loss.item(), lr_,best_test_dcs))
                print(
                    'iteration %d : loss : %f lr_: %f best_dcs : %f' % (iter_num, loss.item(), lr_, Best_dcs))
            train_loss += loss.item()
        Loss.append(train_loss / len(train_dataset))
        logging.info('Epoch %d : loss : %f lr_: %f best_dcs : %f' % (epoch, loss.item(), lr_, Best_dcs))

        avg_dcs = val()

        if avg_dcs > Best_dcs:
            save_model_path = os.path.join(snapshot_path, 'val_best.pth')
            torch.save(net.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
            Best_dcs = avg_dcs

            # performence, avg_hd, avg_jacard, avg_asd = inference(args, net, testloader, args.test_save_dir)
            # print("test avg_dsc: %f" % (performence))
            # Test_Accuracy.append(avg_dcs)
            # if performence >= best_test_dcs:
            #     best_test_dcs = performence
            #     save_model_path = os.path.join(snapshot_path, 'test_best.pth')
            #     torch.save(net.state_dict(), save_model_path)
            #     logging.info("save model to {}".format(save_model_path))

        if epoch >= args.max_epochs - 1:
            save_model_path = os.path.join(snapshot_path,
                                           'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, best_test_dcs))
            torch.save(net.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
            iterator.close()
            break


# --------------------------- ISIC trainer ---------------------------

def trainer_ISIC(args, net, exp_name):
    def train(epoch):
        iou_loss = lovasz_hinge
        ohem_ce_loss = BinaryOhemCrossEntropy()
        net.train()
        iteration = 0
        optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

        for batch_idx, batch_data in enumerate(train_loader):
            image_batch = batch_data['image'].cuda().float()
            label_batch = batch_data['label'].cuda().float()

            optimizer.zero_grad()

            if args.model_name == 'LUCF_Net':
                p1, p2, p3, p4 = net(image_batch)
                outputs = p1 + p2 + p3 + p4
                losses = [
                    (0.2 * ohem_ce_loss(p, label_batch) + 0.8 * iou_loss(p, label_batch))
                    for p in [p1, p2, p3, p4]
                ]
                loss = sum(losses)
            elif args.model_name == 'HiFormer':
                B, C, H, W = image_batch.shape
                image_batch = image_batch.expand(B, 3, H, W)
                outputs = net(image_batch)
                loss = 0.2 * ohem_ce_loss(outputs, label_batch) + 0.8 * iou_loss(outputs, label_batch)
            elif args.model_name == 'PVT_Cascade':
                p1, p2, p3, p4 = net(image_batch)
                outputs = p1 + p2 + p3 + p4
                losses = [
                    (0.2 * ohem_ce_loss(p, label_batch) + 0.8 * iou_loss(p, label_batch))
                    for p in [p1, p2, p3, p4]
                ]
                loss = sum(losses)
            else:
                outputs = net(image_batch)
                loss = 0.2 * ohem_ce_loss(outputs, label_batch) + 0.8 * iou_loss(outputs, label_batch)

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(image_batch)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]')
                logging.info(f'Train Epoch: {epoch} [{batch_idx * len(image_batch)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]')

        print("Iteration numbers: ", iteration)

    def evaluation(epoch, loader):
        net.eval()
        dice_value, iou_value, numm = 0, 0, 0

        for batch_idx, batch_data in enumerate(loader):
            image_batch = batch_data['image'].cuda().float()
            label = batch_data['label'].cuda().float()

            with torch.no_grad():
                if args.model_name == 'LUCF_Net':
                    p1, p2, p3, p4 = net(image_batch)
                    outputs = p1 + p2 + p3 + p4
                elif args.model_name == 'HiFormer':
                    B, C, H, W = image_batch.shape
                    image_batch = image_batch.expand(B, 3, H, W)
                    outputs = net(image_batch)
                elif args.model_name == 'PVT_Cascade':
                    p1, p2, p3, p4 = net(image_batch)  # forward
                    outputs = p1 + p2 + p3 + p4  # additive output aggregation
                else:
                    outputs = net(image_batch)


                output = outputs.cpu().numpy() > 0.5

            label = label.cpu().numpy()
            dice_ave = dc(output, label)
            iou_ave = jc(output, label)
            dice_value += dice_ave
            iou_value += iou_ave
            numm += 1

        dice_average = dice_value / numm
        iou_average = iou_value / numm
        writer.add_scalar('val_metrics/val_dice', dice_average, epoch)
        writer.add_scalar('val_metrics/val_iou', iou_average, epoch)
        print(f"Average dice value of evaluation dataset = {dice_average}")
        print(f"Average iou value of evaluation dataset = {iou_average}")
        logging.info(f"Average dice value of evaluation dataset = {dice_average}")
        logging.info(f"Average iou value of evaluation dataset = {iou_average}")

        return dice_average, iou_average

    def logger_init(log_file_name='monitor', log_level=logging.DEBUG, log_dir=exp_name + '/' + 'logs.txt', only_file=False):
        formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
        logging.basicConfig(filename=log_dir, level=log_level, format=formatter, datefmt='%Y-%d-%m %H:%M:%S')

    os.makedirs(f'logs/{exp_name}', exist_ok=True)
    os.makedirs(f'logs/{exp_name}/model', exist_ok=True)
    writer = SummaryWriter(f'logs/{exp_name}/log')
    save_path = f'logs/{exp_name}/model/best.pkl'
    latest_path = f'logs/{exp_name}/model/latest.pkl'

    EPOCHS = args.max_epochs

    dataset = myDataset(args.root_path, split='train', aug=True)
    dataset2 = myDataset(args.root_path, split='valid', aug=False)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

    max_dice, max_iou, best_ep, min_loss, min_epoch = 0, 0, 0, 10, 0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    for epoch in range(1, EPOCHS + 1):
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        start = time.time()

        dice, iou = evaluation(epoch, val_loader)
        train(epoch)
        scheduler.step()

        if dice > max_dice:
            max_dice = dice
            best_ep = epoch
            torch.save(net.state_dict(), save_path)
        elif epoch - best_ep >= args.max_epochs:
            print('Early stopping!')
            break

        torch.save(net.state_dict(), latest_path)
        time_elapsed = time.time() - start
        logging.info(f'Training and evaluating on epoch:{epoch} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Training and evaluating on epoch:{epoch} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
