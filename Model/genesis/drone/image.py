import matplotlib.pyplot as plt
import numpy as np

def plt_imshow(rgb_image, depth_image):
    plt.clf()  # 清除当前图像
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_image)
    plt.title('RGB')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if depth_image.ndim == 3:
        depth_image = depth_image.squeeze()
    plt.imshow(depth_image, cmap='jet')
    plt.title('Depth')
    plt.axis('off')

    plt.pause(0.01)  # 非阻塞显示
