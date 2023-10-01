import torch
import torch.nn.functional as F
import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from .utils import xywh2xyxy_np
import torchvision.transforms as transforms

import cv2
from PIL import Image

class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        # Unpack data
        img, boxes = data

        # Convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes],
            shape=img.shape)

        # Apply augmentations
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes)

        # Clip out of image boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = ((x1 + x2) / 2)
            boxes[box_idx, 2] = ((y1 + y2) / 2)
            boxes[box_idx, 3] = (x2 - x1)
            boxes[box_idx, 4] = (y2 - y1)

        return img, boxes


class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes


class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return img, boxes


class PadSquare(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ])


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes

# Adjust color brightness strategy 
class adjustGrassColor(object):
    def __init__(self, ):
        self.saturation = 1.25
        self.brightness = 1.15

    def rgb_to_hsv(self, rgb_img):
        # Extract RGB channels
        r, g, b = rgb_img.unbind(0)

        # Get the max and min values across RGB
        max_val, _ = torch.max(rgb_img, dim=0)
        min_val, _ = torch.min(rgb_img, dim=0)
        diff = max_val - min_val

        # Calculate HUE
        h = torch.zeros_like(r)
        mask = max_val == min_val
        h[~mask] = 60.0 * ((g[~mask] - b[~mask]) / diff[~mask] % 6)
        mask = max_val == b
        h[mask] = 60.0 * ((r[mask] - g[mask]) / diff[mask] + 4)
        mask = max_val == g
        h[mask] = 60.0 * ((b[mask] - r[mask]) / diff[mask] + 2)

        # Calculate SATURATION
        s = torch.zeros_like(r)
        mask = max_val != 0
        s[mask] = (diff[mask] / max_val[mask])

        # Calculate VALUE
        v = max_val

        return torch.stack([h, s, v])

    def hsv_to_rgb(self, hsv_img):
        h, s, v = hsv_img.unbind(0)
        c = v * s
        hh = h / 60.0
        x = c * (1 - torch.abs(hh % 2 - 1))
        m = v - c

        segments = hh.to(torch.int32)
        r = c * (segments == 0) + x * (segments == 1) + m * (segments == 4) + m * (segments == 5)
        g = x * (segments == 0) + c * (segments == 1) + c * (segments == 2) + x * (segments == 3)
        b = m * (segments == 0) + m * (segments == 1) + x * (segments == 2) + c * (segments == 3)

        return torch.stack([r, g, b])

    def adjust_grass_color(self, rgb_img):
        hsv_img = self.rgb_to_hsv(rgb_img)
        
        # Adjust saturation
        hsv_img[1] = torch.clamp(hsv_img[1] * self.saturation, 0, 1)
        
        # Adjust brightness = 1.15
        hsv_img[2] = torch.clamp(hsv_img[2] * self.brightness, 0, 1)
    
        return self.hsv_to_rgb(hsv_img)


    def __call__(self, data):
        img, boxes = data

        img = self.adjust_grass_color(img)

        return img, boxes



# Normalize the data to Image Net if weight were trained this way, need to explore darknet code

class Normalize(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data

        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) (img) # Normalize using ImageNet statistics

        return img, boxes




def load_image(path, device):
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)







def compute_cumulative_histogram(image):
    bins = torch.linspace(0, 1, 256)
    hist = torch.histc(image, bins=256, min=0, max=1)
    cdf = torch.cumsum(hist, dim=1)
    cdf_normalized = cdf / cdf[:, -1:]
    return cdf_normalized


def match_histogram(source, reference):
    reference_cdf = compute_cumulative_histogram(reference)
    source_cdf = compute_cumulative_histogram(source)


    matched_image = torch.zeros_like(source)

    for b in range(source.size(0)):
        for c in range(source.size(1)):
            for i in range(256):
                source_val = (i + 0.5) / 256
                ref_idx = torch.searchsorted(reference_cdf[b, c], source_cdf[b, c, i])
                ref_val = (ref_idx + 0.5) / 256
                mask = (source[b, c] >= source_val - 0.5/256) & (source[b, c] < source_val + 0.5/256)
                matched_image[b, c, mask] = ref_val

    return matched_image   


# Convert back to PIL Image and save

class balance_image:
    def __init__(self):
        self.imageName =  "C:\\Users\\stevf\\OneDrive\\Documents\\Projects\\PyTorch-YOLOv3\\data\\turfgrass_VOC\\images\\YOLODataset\\images\\20230210_152530.png"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reference_image = load_image(self.imageName , self.device)

    def __call__(self, data):
        matched_image, boxes = data
        transform = transforms.ToPILImage()
        matched_image = match_histogram(matched_image, self.reference_image)
        # matched_image = transform(matched_image.squeeze())
        # matched_image_pil.save('matched.jpg')
        return matched_image , boxes
    


class WhiteBalanceTransform:
    def __call__(self, data):
        img, boxes = data


        # img_np = np.array(img) # Convert PIL Image to numpy array
        # print("TYPE  ",img.shape)
        # Convert to BGR format for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Split the channels
        b, g, r = cv2.split(img_bgr)
        
        # Compute the mean of each channel
        r_avg = cv2.mean(r)[0]
        g_avg = cv2.mean(g)[0]
        b_avg = cv2.mean(b)[0]

        # Calculate scaling factors
        k = (r_avg + g_avg + b_avg) / 3
        kr = k / r_avg
        kg = k / g_avg
        kb = k / b_avg

        # White balance correction
        r = cv2.normalize(r * kr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        g = cv2.normalize(g * kg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        b = cv2.normalize(b * kb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Merge channels and convert back to RGB format
        img_balanced = cv2.merge([b, g, r])
        img_balanced = cv2.cvtColor(img_balanced, cv2.COLOR_BGR2RGB)

        # Convert numpy array back to PIL Image
        return np.array(Image.fromarray(img_balanced)), boxes

class correctImage(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential(
            [
            iaa.AddToBrightness((-100, 0)),
            ],
        )

class correctImageAspectRatio(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential(
            [
            iaa.Resize({"height": 416, "width": "keep-aspect-ratio"}),
            iaa.CropToFixedSize(height=416, width=416)
            ],
        )




DEFAULT_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])
