
---
# DeepMedSeg

## 项目简介

**DeepMedSeg** 是一个用于 **医学图像分割** 的开源框架，支持处理 **2D** 和 **3D** 医学数据，并集成了多种流行的分割网络架构。本项目旨在简化研究人员在医学图像分割领域的开发和实验过程，具有模块化、可扩展性强的特点，能够快速定制和测试各种分割模型。

该框架支持的主要功能包括：
- **2D 与 3D 医学数据的处理**，如CT、MRI、X光片等。
- **流行的分割网络集成**：包括 U-Net、3D U-Net、Attention U-Net、ResUNet、Swin UNet、Missformer、TransUNet、Swin UNetr、UNETR、nnUNet、nnFormer 等（目前正在拓展中）。
- **高扩展性**：可以轻松集成新模型和数据处理方法。
- **完整的训练与测试流程**，包括数据预处理、训练、验证、评估等模块。

---

## 目录结构

```
DeepMedSeg/
├── 2d/                             # 2D医学图像分割部分
│   ├── data/                       # 存放2D切片数据
│   ├── networks/                   # 2D分割模型
│   ├── other_networks/             # 其他扩展的网络
│   ├── utils/                      # 2D工具函数
│   ├── test.py                     # 2D分割的测试代码
│   └── train.py                    # 2D分割的训练代码
├── 3d/                             # 3D医学图像分割部分
│   ├── lib/                        # 存放3D分割相关的库
│   ├── networks/                   # 3D分割模型
│   ├── dataset_AMOS.json           # AMOS数据集配置文件
│   ├── dataset_FeTA.json           # FeTA数据集配置文件
│   ├── dataset_FLARE.json          # FLARE数据集配置文件
│   ├── load_datasets_transforms.py # 数据加载及增强
│   ├── main_finetune.py            # 3D模型微调代码
│   ├── main_train.py               # 3D模型训练代码
│   ├── test_seg.py                 # 3D分割测试代码
├── README.md                       # 项目说明文件
├── requirements.txt                # Python依赖包文件
└── data                            # 训练数据
```

---

## 安装指南

### 环境依赖

请确保您的系统安装了以下依赖：

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- matplotlib
- SimpleITK
- scikit-learn
- albumentations

### 安装步骤

1. 克隆项目到本地：
   ```bash
   git clone https://github.com/your-username/DeepMedSeg.git
   cd DeepMedSeg
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

---

## 数据集准备

项目支持多个公开医学数据集，支持对3D数据的切片

- **[Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)**：2D、3D
- **[ACDC](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view)**：2D
- **[ISIC](https://challenge.isic-archive.com/data/#2018)** 数据集：2D
- **[AMOS](https://amos22.grand-challenge.org/)** 数据集：2D、3D
- **[FeTA](https://feta.grand-challenge.org/feta-2021/)** 数据集：3D
- **[FLARE](https://flare.grand-challenge.org/)** 数据集：2D、3D

### Synapse、AMOS、FLARE
此项目支持以上数据集的二维分割与三维分割，二维分割3D数据的深度切片

将以上数据集按照如下方式组织以支持三维分割
```
root_dir/
├── imagesTr
├── labelsTr
├── imagesVal
├── labelsVal
├── imagesTs
```
对于二维分割 ，运行以下命令进行预处理3D数据（以Synapse数据集为例）：
```bash
cd data/Synapse
python Slice_Synapse.py
```

此后会得到切片后的2D数据，请按照以下结构组织切片数据：

```
data/synapse
├── Synapse/synapse_2d/
│   ├── train_npz_new/ 
│   ├── test_vol_h5_new/
│   ├── test_h5_new/
```
### ACDC
以上下载链接为预处理后的数据，将ACDC数据集下载后并放到data/ACDC下面

---

## 使用说明

### 2D 医学图像分割

#### 训练

启动2D模型的训练：

```bash
cd 2d
python train.py --args.model <网络名称>
```

#### 测试

训练完成后，测试2D模型：

```bash
python test.py --args.model <网络名称> --pretrained_pth <权重路径>
```

### 3D 医学图像分割

#### 训练

使用3D模型进行训练：

> **注意**：此部分参考自3DUX-Net，修复了monai的裁剪问题，更多的细节请移步原作者主页 [3DUX-Net](https://github.com/MASILab/3DUX-Net)。

```bash
cd 3d
python main_train.py --root <root_folder> --output <output_folder> \
--dataset flare --network 3DUXNET --mode train --pretrain False \
--batch_size 1 --crop_sample 2 --lr 0.0001 --optim AdamW --max_iter 40000 \ 
--eval_step 500 --gpu 0 --cache_rate 0.2 --num_workers 2
```

#### 测试

测试3D分割模型：

```bash
python test_seg.py --root <path_to_image_folder> --output <path_to_output> \
--dataset flare --network 3DUXNET --trained_weights <path_to_trained_weights> \
--mode test --sw_batch_size 4 --overlap 0.7 --gpu 0 --cache_rate 0.2
```

---

## 模型集成

### 2D 网络

`2d/other_networks/` 文件夹中包含了各种2D分割模型，如U-Net、Swin UNet等。您可以根据需要修改或添加新的网络。

### 3D 网络

`3d/networks/` 文件夹用于存储3D模型，如3D U-Net、nnFormer等。您可以在此处扩展新的3D模型架构。

---

## 贡献指南

欢迎对本项目的贡献！您可以通过以下方式参与：

1. Fork 此仓库。
2. 创建新的特性分支 (`git checkout -b feature/AmazingFeature`)。
3. 提交您的修改 (`git commit -m 'Add some AmazingFeature'`)。
4. 推送到分支 (`git push origin feature/AmazingFeature`)。
5. 提交 Pull Request。

---

## 说明
此项目集成了较多开源项目，感谢他们的贡献。如果您有问题，请及时提交Issues，有时间我会及时回复。未来考虑集成一些训练Loss和基于Mamba的网络。

---

## References

- [TransUnet](https://github.com/Beckschen/TransUNet)
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
- [DAEFormer](https://github.com/xmindflow/DAEFormer)
- [MissFormer](https://github.com/ZhifangDeng/MISSFormer)
- [FocalUnet](https://github.com/shandao/MP-FocalUnet)
- [CASCADE](https://github.com/SLDGroup/CASCADE)
- [3DUX-Net](https://github.com/MASILab/3DUX-Net)

---

