from networks.LUCF_Net import LUCF_Net
from other_networks.MISSFormer.MISSFormer import MISSFormer
from other_networks.SwinUnet.vision_transformer import SwinUnet
from other_networks.HiFormer.HiFormer import HiFormer
from other_networks.DAEFormer.DAEFormer import DAEFormer
from other_networks.TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
from other_networks.CASCADE.networks import PVT_CASCADE
from other_networks.FocalNet.main import FUnet
from other_networks.UNet.res_unet import ResUnet
from other_networks.UNet.unet import UNet

def net_maker(args):
    # 根据不同的模型名称，实例化相应的网络
    if args.model_name == 'LUCF_Net':
        net = LUCF_Net(in_chns=args.in_chans, class_num=args.num_classes).cuda()
    elif args.model_name == 'Swin_UNet':
        net = SwinUnet(num_classes=args.num_classes).cuda()
    elif args.model_name == 'MISSFormer':
        net = MISSFormer(num_classes=args.num_classes).cuda()
    elif args.model_name == 'HiFormer':
        net = HiFormer(in_chans=3, n_classes=args.num_classes).cuda()
    elif args.model_name == 'DAEFormer':
        net = DAEFormer(num_classes=args.num_classes).cuda()
    elif args.model_name == 'TransUNet':
        net = ViT_seg(img_size=224, num_classes=args.num_classes).cuda()
    elif args.model_name == 'PVT_Cascade':
        net = PVT_CASCADE(n_class=args.num_classes).cuda()
    elif args.model_name == 'ResUNet':
        net = ResUnet(in_chans=args.in_chans, num_classes=args.num_classes).cuda()
    elif args.model_name == 'UNet':
        net = UNet(in_chns=args.in_chans, class_num=args.num_classes).cuda()
    return net