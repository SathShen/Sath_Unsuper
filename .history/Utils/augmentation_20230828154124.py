import torchvision.transforms as transforms
import PIL.Image as Image
import torch
import random


class HazeSimulation(object):
    """
    Haze simulation augmentation

    Args:
        p (float): probability of applying HazeSimulation
        t (float, float): minimum / maximum transmission ratio
    """
    def __init__(self, p=0.2, t=(0.3, 0.7)):
        self.p = p
        if isinstance(t, tuple) and len(t) == 2:
            self.tm = t
        elif isinstance(t, float):
            self.tm = (t, 1.)
        else:
            raise TypeError('t should be float or tuple with length 2')

    def __call__(self, img):
        if random.random() > self.p:
            return img
        trans_ratio = random.uniform(*self.tm)
        self.transmition_map = torch.full(img.shape, trans_ratio)
        num_pixels = img.shape[-2] * img.shape[-1]
        num_A = num_pixels // 100
        A = (torch.sort(img.sum(dim=0).reshape(-1), descending=True)[0][:num_A] / 3).mean() 
        img = img * self.transmition_map + A * (1 - self.transmition_map)
        return img
    

class Cutout(object):
    """
    Cutout augmentation
    Args:
        p (float): probability of applying Cutout
        s (float, float): minimum / maximum proportion of cutout area against input image size
        r (float, float): minimum / maximum ratio of cutout area against input image area
    """
    def __init__(self, p=0.2, scale=(0.1, 0.4), ratio=(3./5, 5./3)):
        self.p = p

        if isinstance(scale, tuple) and len(scale) == 2:
            self.scale = scale
        elif isinstance(scale, float):
            self.scale = (0, scale)
        else:
            raise TypeError('scale should be float or tuple with length 2')
        
        if isinstance(ratio, tuple) and len(ratio) == 2:
            self.ratio = ratio
        elif isinstance(ratio, float):
            self.ratio = (1 - ratio, 1 + ratio)
        else:
            raise TypeError('ratio should be float or tuple with length 2')

    def __call__(self, img):
        if random.random() > self.p:
            return img
        scale = random.uniform(*self.scale)
        ratio = random.uniform(*self.ratio)
        imgh, imgw = img.shape[-2:]
        centery = random.randint(0, imgh - 1)
        centerx = random.randint(0, imgw - 1)
        h = int(imgh * scale)
        w = int(imgw * scale * ratio)
        y1 = max(0, centery - h // 2)
        y2 = min(imgh, centery + h // 2)
        x1 = max(0, centerx - w // 2)
        x2 = min(imgw, centerx + w // 2)
        img[:, y1:y2, x1:x2] = 1.
        return img


class DinoV1Augmentation(object):
    def __init__(self, cfgs):
        color_jitter = transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=cfgs.AUG.INTENSITY, contrast=cfgs.AUG.CONTRAST,
                                   saturation=cfgs.AUG.SATURATION, hue=cfgs.AUG.HUE)]), p=0.8)
        flips = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5)])
        hazesimu = HazeSimulation(p=0.2, t=(0.3, 0.7))
        cutout = Cutout(p=0.2, scale=(0.1, 0.4), ratio=(3./5, 5./3))

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(cfgs.AUG.CROP_SIZ, scale=cfgs.AUG.GLOBAL_SCALE, interpolation=Image.BICUBIC),
            flips,
            color_jitter,
            hazesimu,
            cutout
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(cfgs.AUG.CROP_SIZ, scale=cfgs.AUG.GLOBAL_SCALE, interpolation=Image.BICUBIC),
            flips,
            color_jitter,
            hazesimu
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops