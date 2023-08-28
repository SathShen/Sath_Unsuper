import torchvision.transforms as transforms
import PIL.Image as Image
import torch
import random


class HazeSimulation(object):
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


class DinoV1Augmentation(object):
    def __init__(self, cfgs):
        color_jitter = transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8)
        flips = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5)])
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
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