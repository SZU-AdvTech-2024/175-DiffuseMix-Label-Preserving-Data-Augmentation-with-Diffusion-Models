import torch
from ..utils import *
from ..gradient.mifgsm import MIFGSM
import random
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from accelerate import Accelerator
import torchvision.transforms.functional as F
import os

def is_black_image(image):
    histogram = image.convert("L").histogram()
    return histogram[-1] > 0.9 * image.size[0] * image.size[1] and max(histogram[:-1]) < 0.1 * image.size[0] * image.size[1]

def split_and_rotate(image, n, t):
    """
    Splits the image into n x n blocks and applies a random rotation within [-t, t] degrees to each block.
    """
    _, h, w = image.shape
    block_h, block_w = h // n, w // n

    blocks = []
    for i in range(n):
        for j in range(n):
            block = image[:, i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
            angle = random.uniform(-t, t)
            rotated_block = F.rotate(block.unsqueeze(0), angle, expand=False).squeeze(0)
            blocks.append(rotated_block)

    # Reconstruct the image from the rotated blocks
    reconstructed_image = torch.zeros_like(image)
    for idx, block in enumerate(blocks):
        i, j = divmod(idx, n)
        reconstructed_image[:, i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w] = block

    return reconstructed_image

def combine_images(original_img, augmented_img, blend_width=20):
    _, height, width = original_img.shape
    combine_choice = random.choice(['horizontal', 'vertical'])

    if combine_choice == 'vertical':
        mask = torch.linspace(0, 1, blend_width).view(1, 1, -1)
        mask = mask.repeat(1, width, 1)
        mask = torch.cat([torch.zeros(1, width, height // 2 - blend_width // 2), mask,
                          torch.ones(1, width, height // 2 - blend_width // 2 + blend_width % 2)], dim=2)
        mask = mask.permute(0, 2, 1)

    else:
        mask = torch.linspace(0, 1, blend_width).view(1, -1, 1)
        mask = mask.repeat(1, 1, height)
        mask = torch.cat([torch.zeros(1, width // 2 - blend_width // 2, height), mask,
                          torch.ones(1, width // 2 - blend_width // 2 + blend_width % 2, height)], dim=1)
        mask = mask.permute(0, 2, 1)

    mask = mask.to(original_img.device)
    blended_tensor = (1 - mask) * original_img + mask * augmented_img

    return blended_tensor

def load_fractal_images(fractal_img_dir):
    fractal_img_paths = [os.path.join(fractal_img_dir, fname) for fname in os.listdir(fractal_img_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
    imgs_Image = [Image.open(path).convert('RGB').resize((224, 224)) for path in fractal_img_paths]

    imgs_tensor = [F.to_tensor(img) for img in imgs_Image]
    return torch.stack(imgs_tensor)

def blend_images_with_resize(base_img, overlay_img, alpha=0.2):
    overlay_img_resized = F.resize(overlay_img, base_img.shape[1:])
    blended_img = (1 - alpha) * base_img + alpha * overlay_img_resized
    blended_img = torch.clamp(blended_img, 0, 1)
    return blended_img

class fractal(MIFGSM):

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_scale=1, targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='fractal', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        fractal_path = 'transferattack/input_transformation/fractal_image'
        model_id = "timbrooks/instruct-pix2pix"

        self.num_scale = num_scale
        self.fractal_imgs = load_fractal_images(fractal_path)

    def transform(self, x, n=2, t=2, num_transforms=3, **kwargs):
        augmented_data = []

        for i in range(x.size(0)):  # 遍历每张输入图像
            img_tensor = x[i]
            transformed_imgs = []  # 存储当前图像的多次变换
            for _ in range(num_transforms):
                rotated_img = split_and_rotate(img_tensor, n=n, t=t)
                random_fractal_img = random.choice(self.fractal_imgs)
                blended_img = blend_images_with_resize(rotated_img.cuda(), random_fractal_img.cuda())
                transformed_imgs.append(blended_img)

            # 对多次变换的结果进行合并（如取平均）
            merged_img = torch.mean(torch.stack(transformed_imgs), dim=0)
            augmented_data.append(merged_img)

        augmented_data = torch.stack(augmented_data)
        return augmented_data

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale)) if self.targeted else self.loss(logits, label.repeat(self.num_scale))