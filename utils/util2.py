from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F

def get_text_criterion(cfg):
    if cfg["text_criterion"] == "spherical":
        text_criterion = spherical_dist_loss
    elif cfg["text_criterion"] == "cosine":
        text_criterion = cosine_loss
    else:
        return NotImplementedError("text criterion [%s] is not implemented", cfg["text_criterion"])
    return text_criterion


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return ((x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)).mean()


def cosine_loss(x, y, scaling=1):
    return scaling * (1 - F.cosine_similarity(x, y).mean())


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(0.0, 1.0).cpu().float().numpy()  
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  
    else: 
        image_numpy = input_image
    return image_numpy.astype(imtype)


# Final text prompt template used in all experiments
def get_augmentations_template():         
    templates = "a photo of a {}"
    return templates


def compose_text_with_templates(text, templates):
    return templates.format(text)
