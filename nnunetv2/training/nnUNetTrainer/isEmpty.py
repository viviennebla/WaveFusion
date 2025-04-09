import os
import cv2
import numpy as np

def check_empty_masks(mask_directory):
    empty_masks = []

    # 遍历掩码目录中的所有文件
    for filename in os.listdir(mask_directory):
        # 构建完整的文件路径
        mask_path = os.path.join(mask_directory, filename)

        # 读取掩码图像
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 检查掩码是否为空（即所有像素值为0）
        if mask is not None and np.all(mask == 0):
            empty_masks.append(filename)

    # 输出为空的掩码文件名
    if empty_masks:
        print("Empty mask files:")
        for empty_mask in empty_masks:
            print(empty_mask)
    else:
        print("No empty mask files found.")

# 指定掩码目录路径
mask_directory_path = "/mnt/d/zfy/MUK/KUM/dataset/nnUnet_raw/Dataset112_thyroid/labelsTs"

# 调用函数检查空掩码
check_empty_masks(mask_directory_path)