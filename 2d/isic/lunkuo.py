import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_and_save_segmentation(original_images_folder, true_masks_folder, predicted_masks_folder, save_folder):
    # 获取原始图像文件列表
    original_image_files = [f for f in os.listdir(original_images_folder) if f.endswith('.jpg')]

    # 确保保存结果的文件夹存在
    os.makedirs(save_folder, exist_ok=True)

    for original_image_file in original_image_files:
        # 加载原始图像
        original_image = cv2.imread(os.path.join(original_images_folder, original_image_file))

        # 构建真实标签和预测的二进制掩膜文件路径
        true_mask_file = os.path.join(true_masks_folder, original_image_file)
        predicted_mask_file = os.path.join(predicted_masks_folder, original_image_file
        )
        print(predicted_mask_file)


        # 加载真实标签和预测的二进制掩膜
        true_mask = cv2.imread(true_mask_file, cv2.IMREAD_GRAYSCALE)
        predicted_mask = cv2.imread(predicted_mask_file, cv2.IMREAD_GRAYSCALE)

        # 创建一个三通道的图像副本，用于在上面绘制标签和预测的轮廓
        visualization_image = original_image.copy()

        # 将真实标签和预测的轮廓绘制在图像上
        contours_true, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_predicted, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(visualization_image, contours_true, -1, (0, 255, 0), 2)  # 真实标签用绿色绘制
        cv2.drawContours(visualization_image, contours_predicted, -1, (0, 0, 255), 2)  # 预测的轮廓用红色绘制

        # 保存结果图像
        result_file = os.path.join(save_folder, f"result_{original_image_file}")
        cv2.imwrite(result_file, visualization_image)

# 替换下面的路径为你实际的文件夹路径
visualize_and_save_segmentation(
    original_images_folder='../data/skin_lesion/isic2016_jpg_512_smooth/Validation/Image',
    true_masks_folder='../data/skin_lesion/isic2016_jpg_512_smooth/Validation/Label',
    predicted_masks_folder=r'results\ISIC-2016-pictures\LUCF_Net_new',
    save_folder='lunkuo_results'
)
