import sys
import logging
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import os
from networks_factory import net_maker
from utils.dataset_ACDC import ACDCdataset, RandomGenerator
from tester import inference_ACDC, inference_organs
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='Synapse',help='')
    parser.add_argument('--model_name', type=str, default='',help='selected model,Swin_UNet,MISSFormer,LUCF_Net,HiFormer,DAEFormer,TransUNet,PVT_Cascade')
    parser.add_argument("--img_size", default=224)
    parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
    parser.add_argument("--volume_path", default="./data/ACDC/test")
    parser.add_argument('--output_dir', type=str, default="test_log/ACDC", help='output')
    parser.add_argument("--z_spacing", default=10)
    parser.add_argument("--num_classes", default=4)
    parser.add_argument('--in_chans', type=int, default=1, help='input channel')
    parser.add_argument('--test_weights', type=str,default=r'',help='testing weights path')
    parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int,default=2222, help='random seed')
    parser.add_argument('--n_skip', type=int,default=3, help='using number of skip-connect, default is num')

    args = parser.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    snapshot_name = args.output_dir + '/' + args.model_name + '_'+args.datasets

    net = net_maker(args)
    net.load_state_dict(torch.load(args.test_weights))


    log_folder = snapshot_name
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+args.model_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    args.test_save_dir = os.path.join(snapshot_name, args.test_save_dir)
    os.makedirs(args.test_save_dir, exist_ok=True)
    if args.dataset == 'Synapse' or args.dataset == 'AMOS' or args.dataset == 'Flare21':
        inference_organs(args, net, snapshot_name)

    elif args.dataset == 'ACDC':
        db_test =ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
        testloader = DataLoader(db_test, batch_size=1, shuffle=False)
        results = inference_ACDC(args, net, testloader, args.test_save_dir)


