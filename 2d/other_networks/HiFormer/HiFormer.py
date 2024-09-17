import torch.nn as nn
from einops.layers.torch import Rearrange

from other_networks.HiFormer.Encoder import All2Cross
from other_networks.HiFormer.Decoder import ConvUpsample, SegmentationHead
from other_networks.HiFormer.HiFormer_configs import *

class HiFormer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, n_classes=9):
        super().__init__()
        config = get_hiformer_b_configs()
        self.img_size = img_size
        self.patch_size = [4, 16]
        self.n_classes = n_classes
        self.All2Cross = All2Cross(config = config, img_size= img_size, in_chans=in_chans)
        
        self.ConvUp_s = ConvUpsample(in_chans=384, out_chans=[128,128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=96, upsample=False)
    
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=3,
        )    

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                128, 16,
                kernel_size=1, stride=1,
                padding=0, bias=True),
            # nn.GroupNorm(8, 16), 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
    
    def forward(self, x):


        xs = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        reshaped_embed = []
        for i, embed in enumerate(embeddings):

            embed = Rearrange('b (h w) d -> b d h w', h=(self.img_size//self.patch_size[i]), w=(self.img_size//self.patch_size[i]))(embed)
            embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)
            
            reshaped_embed.append(embed)

        C = reshaped_embed[0] + reshaped_embed[1]
        C = self.conv_pred(C)

        out = self.segmentation_head(C)
        
        return out

if __name__ =="__main__":
    import torch
    from thop import profile
    from pytorch_model_summary import summary
    m = HiFormer(in_chans=1)
    x = torch.rand(1,3,224,224)

    model = m
    print(summary(model, x, show_input=False, show_hierarchical=False))
    flops, params = profile(model, (x,))
    print('GFLOPs: ', flops/1000000000, 'params: ', params/1000000)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
