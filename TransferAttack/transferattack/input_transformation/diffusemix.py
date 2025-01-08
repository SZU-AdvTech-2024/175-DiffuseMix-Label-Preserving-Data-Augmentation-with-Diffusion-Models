import torch
from ..utils import *
from ..gradient.mifgsm import MIFGSM
import random
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from accelerate import Accelerator
#
import torchvision.transforms.functional as F
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def is_black_image(image):
    histogram = image.convert("L").histogram()
    return histogram[-1] > 0.9 * image.size[0] * image.size[1] and max(histogram[:-1]) < 0.1 * image.size[0] * \
        image.size[1]


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

    # original_array = original_img.float() / 255.0
    # augmented_array = augmented_img.float() / 255.0

    # blended_array = (1 - mask) * original_array + mask * augmented_array
    # blended_tensor = torch.clamp(blended_array * 255, 0, 255)

    blended_tensor = (1 - mask) * original_img + mask * augmented_img

    return blended_tensor


def load_fractal_images(fractal_img_dir):
    fractal_img_paths = [os.path.join(fractal_img_dir, fname) for fname in os.listdir(fractal_img_dir) if
                         fname.endswith(('.png', '.jpg', '.jpeg'))]
    imgs_Image = [Image.open(path).convert('RGB').resize((224, 224)) for path in fractal_img_paths]

    imgs_tensor = [F.to_tensor(img) for img in imgs_Image]
    return torch.stack(imgs_tensor)


class ModelHandler:
    def __init__(self, model_id, device):
        self.accelerator = Accelerator()
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None
        ).to(device)
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)

    def generate_images(self, prompt, tensor, num_images, guidance_scale):
        # image = Image.open(img_path).convert('RGB').resize((256, 256))
        image = transforms.ToPILImage()(tensor).resize((224, 224))
        return self.pipeline(prompt, image=image, num_images_per_prompt=num_images,
                             guidance_scale=guidance_scale).images


def blend_images_with_resize(base_img, overlay_img, alpha=0.20):
    overlay_img_resized = F.resize(overlay_img, base_img.shape[1:])
    blended_img = (1 - alpha) * base_img + alpha * overlay_img_resized

    blended_img = torch.clamp(blended_img, 0, 1)

    return blended_img


class SDIM(MIFGSM):
    """
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of scaled copies in each iteration.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=5
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1., num_scale=1, targeted=False,
                 random_start=False, norm='linfty', loss='crossentropy', device=None, attack='SDIM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        fractal_path = 'transferattack/input_transformation/fractal_image'
        model_id = "timbrooks/instruct-pix2pix"

        self.num_scale = num_scale
        self.fractal_imgs = load_fractal_images(fractal_path)
        self.model_handler = ModelHandler(model_id=model_id, device='cuda')

    def transform(self, x, **kwargs):
        # Initialize the model

        augmented_data = []

        prompts = ["Autumn", "snowy", "watercolor art", "sunset", "rainbow", "aurora",
                   "mosaic", "ukiyo-e", "a sketch with crayon"]

        for i in range(x.size(0)):  # x.size(0)
            img_tensor = x[i]

            # random_prompt = random.choice(prompts)
            # num_images = 1
            # guidance_scale = 4
            # generated_images = self.model_handler.generate_images(random_prompt, img_tensor, num_images, guidance_scale)
            # generated_images = [F.to_tensor(img).cuda() for img in generated_images]

            # combined_img = combine_images(img_tensor, generated_images[0])

            random_fractal_img = random.choice(self.fractal_imgs)

            blended_img = blend_images_with_resize(img_tensor.cuda(), random_fractal_img.cuda())
            # blended_img = blend_images_with_resize(combined_img.cuda(), random_fractal_img.cuda())
            augmented_data.append(blended_img)

        augmented_data = torch.stack(augmented_data)

        return augmented_data

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale)) if self.targeted else self.loss(logits, label.repeat(
            self.num_scale))