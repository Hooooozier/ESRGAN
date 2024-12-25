import os
import torch
from torch import nn
import config
from PIL import Image
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import dataset


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_examples(low_res_folder, gen):
    files = os.listdir(low_res_folder)

    gen.eval()
    for file in files:
        image = Image.open(os.path.join(low_res_folder, file))
        # image = dataset2.test_transform(image)
        image = image.resize((config.LOW_RES, config.LOW_RES))
        image = dataset.lowres_transform(image)
        image = dataset.to_tensor(image)
        with torch.no_grad():
            upscaled_img = gen(
                image
                .unsqueeze(0)
                .to(config.DEVICE)
            )
        save_image(upscaled_img, os.path.join(config.VALID_SAVE_PATH, file))
    gen.train()

def initialize_weights_kaiming(model, scale=0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale

def calculate_psnr_ssim(folder1, folder2, output_file):
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))
    
    psnr_values = []
    ssim_values = []
    results = []

    for file1, file2 in tqdm(zip(files1, files2)):
        if file1 == file2 and file1.endswith('.png'):
            img1 = Image.open(os.path.join(folder1, file1)).convert('RGB')
            img2 = Image.open(os.path.join(folder2, file2)).convert('RGB')
            
            img1 = np.array(img1)
            img2 = np.array(img2)
            
            psnr_value = psnr(img1, img2)
            ssim_value = ssim(img1, img2, channel_axis=2)
            
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            
            results.append((file1, psnr_value, ssim_value))
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    with open(output_file, 'w') as f:
        for file_name, psnr_value, ssim_value in results:
            f.write(f'{file_name}: PSNR={psnr_value}, SSIM={ssim_value}\n')
        f.write(f'Average PSNR: {avg_psnr}\n')
        f.write(f'Average SSIM: {avg_ssim}\n')
    
    return avg_psnr, avg_ssim
