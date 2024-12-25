import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import config

LOW_RES = config.LOW_RES
HIGH_RES = config.HIGH_RES
INTERMEDIA_SIZE = config.INTERMEDIA_SIZE

to_tensor = transforms.ToTensor()

def pre_transforms(image):
    width, height = image.size
    left = np.random.randint(0, width - HIGH_RES + 1)
    top = np.random.randint(0, height - HIGH_RES + 1)
    image = image.crop((left, top, left + HIGH_RES, top + HIGH_RES))
    
    if np.random.rand() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    return image

def lowres_transform(image):
    if INTERMEDIA_SIZE != LOW_RES:
        image = image.resize((INTERMEDIA_SIZE, INTERMEDIA_SIZE), Image.Resampling.LANCZOS)
    image = image.resize((LOW_RES, LOW_RES), Image.Resampling.LANCZOS)
    return image

def highres_transform(image):
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    return image

class DIV2K_DS(Dataset):
    def __init__(self, data_path:str):
        super().__init__()
        self.data = []
        self.data_path = data_path

        for img in sorted(os.listdir(self.data_path)):
            self.data.append(os.path.join(self.data_path, img))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.data[index]
        image = Image.open(img_path).convert('RGB')
        image = pre_transforms(image)

        low_res = lowres_transform(image)
        high_res = highres_transform(image)

        low_res = to_tensor(low_res)
        high_res = to_tensor(high_res)

        return low_res, high_res



def save_image(tensor, path):
    print(f"Saving image with shape: {tensor.shape}")
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    image = tensor.clone().detach().cpu().numpy()
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = (image * 0.5 + 0.5) * 255.0
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)

if __name__ == "__main__":
    data_path = "./SUNRGBD-valid_images_384"
    low_res_save_path = "./low_res"
    high_res_save_path = './high_res'
    dataset = DIV2K_DS(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    original_name = sorted(os.listdir('./SUNRGBD-valid_images_384'))

    for i, (low_res, high_res) in enumerate(dataloader):
        low_res_path = os.path.join(low_res_save_path, original_name[i])
        save_image(low_res, low_res_path)
        high_res_path = os.path.join(high_res_save_path, original_name[i])
        save_image(high_res, high_res_path)