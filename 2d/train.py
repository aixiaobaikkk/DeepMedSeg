import torch.backends.cudnn as cudnn
import argparse
import logging
import random
import yaml
import numpy as np
import torch
import os
from trainer import trainer_ACDC, trainer_ISIC, trainer_organs

# Define an argument parser to get parameters from the command line input
parser = argparse.ArgumentParser()

# Add command-line argument definitions; each argument has its name, type, default value, and help information
parser.add_argument('--config', type=str, default='configs/ISIC.yaml', help='Path to the config file')
parser.add_argument('--default_configuration', type=bool, default='True', help='Whether to use default configuration')
parser.add_argument('--model_name', type=str,
                    default='LUCF_Net', help='Choose a model: Swin_UNet, MISSFormer, LUCF_Net, HiFormer, etc.')
parser.add_argument('--root_path', type=str,
                    default='../data/synapse/synapse_2d/train_npz_new', help='Root directory of the training data')
parser.add_argument('--volume_path', type=str,
                    default='../data/synapse/synapse_2d/test_vol_h5_new', help='Root directory of the validation data')
parser.add_argument('--dataset', type=str,
                    default='ACDC', help='Name of the experiment')
parser.add_argument('--list_dir', type=str,
                    default='../data/synapse/synapse_2d/lists_Synapse', help='Directory of the data list')
parser.add_argument('--num_classes', type=int,
                    default=9, help='Number of output channels of the network, i.e., number of classes')
parser.add_argument('--in_chans', type=int,
                    default=1, help='Number of input channels of the network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='Maximum number of iterations')
parser.add_argument('--max_epochs', type=int,
                    default=600, help='Maximum number of training epochs')
parser.add_argument('--batch_size', type=int,
                    default=16, help='Batch size per GPU')
parser.add_argument('--workers', type=int,
                    default=4, help='')
parser.add_argument('--output_dir', type=str,
                    default="model_pth/Flare21/exp_new", help='Model output directory')
parser.add_argument('--is_pretrained', type=bool,
                    default=False, help='Whether to load a pre-trained model')
parser.add_argument('--pretrained_pth', type=str,
                    default=r'', help='Path to the pre-trained model')
parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--gpu', type=str, default='1', help='GPU ID to use')

parser.add_argument('--deterministic', type=int, default=1,
                    help='Whether to use deterministic training to ensure reproducible results')
parser.add_argument('--base_lr', type=float, default=0.05,
                    help='Base learning rate for the segmentation network')
parser.add_argument('--save_interval', type=int, default=20,
                    help='Interval epoch for saving the model')
parser.add_argument('--img_size', type=int,
                    default=224, help='Input image size to the network')
parser.add_argument('--seed', type=int,
                    default=2222, help='Random seed for reproducibility')
parser.add_argument('--n_skip', type=int,
                    default=3, help='Number of skip connections')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='Z-axis spacing of the image')
parser.add_argument('--exp', type=int,
                    default='1', help='Experiment number')
# Parse command-line arguments
args = parser.parse_args()

if args.default_configuration:
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Update parameters with values from the configuration file
    for key, value in config.items():
        setattr(args, key, value)
        print(args.__dict__[key])


# Main function entry
if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Configure cuDNN library based on whether deterministic training is used
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # Set random seed to ensure reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    from networks_factory import net_maker
    net = net_maker(args)
    # Load weights from the specified path if a pre-trained model is needed
    if args.is_pretrained:
        net.load_state_dict(torch.load(args.pretrained_pth))

    # Set the experiment path and create directories
    args.exp = 'exp' + str(args.exp) + '/' + args.model_name + '_' + args.dataset + str(args.img_size)
    snapshot_path = args.output_dir + args.exp
    print(snapshot_path)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # Select the appropriate dataset trainer and start training
    if(args.dataset == 'Synapse' or args.dataset == 'Flare21' or args.dataset == 'AMOS' ):
        trainer_organs(args,net, snapshot_path)
    elif (args.dataset == 'ACDC'):
        trainer_ACDC(args, net, snapshot_path)
    elif(args.dataset == 'ISIC'):
        trainer_ISIC(args, net,snapshot_path)
