import imgaug.augmenters as iaa
from torchvision import transforms
from pytorchyolo.utils.transforms import ToTensor, PadSquare, RelativeLabels, AbsoluteLabels, ImgAug, adjustGrassColor
import imgaug  as ia

class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.1)),
            iaa.AddToBrightness((-20, 100)),
            iaa.AddToHue((-10, 10)),
            iaa.Fliplr(0.5),
        ])

class greenAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.6, 1.2)),
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.WithChannels(0, iaa.Add((4))),           # Adjust hue
            iaa.WithChannels(1, iaa.LinearContrast((1))),  # Adjust saturation
            iaa.WithChannels(1, iaa.Add((5))),
            iaa.WithChannels(2, iaa.LinearContrast((1))),  # Adjust value/brightness
            iaa.WithChannels(2, iaa.Add((92))),
            iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
        ])

class StrongAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            # iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-15, 15), translate_percent=(-0.1, 0.1), scale=(0.8, 1.1)),
            iaa.AddToBrightness((-10, 60)),
            iaa.AddToHue((-5, 10)),
            iaa.Fliplr(0.5),
        ])

class greyAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-45, 45), translate_percent=(-0.1, 0.1), scale=(0.8, 1.1)),
            iaa.AddToBrightness((0, 80)),
            iaa.AddToHue((10, 20)),
            iaa.Fliplr(0.5),
           # iaa.ChangeColorTemperature((1100,10000)),
            iaa.Grayscale(alpha=(0.0, 1.0)),
        ])

class newAug(ImgAug):
    def __init__(self, ):

        self.augmentations = iaa.Sequential([
            iaa.Fliplr(0.5),  # Horizontally flip 50% of the images
            iaa.Affine(
                rotate=(-10, 10),  # Rotate images between -25 and 25 degrees
                shear=(-8, 8),     # Shear images
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}  # Scale images
            ),
            iaa.GaussianBlur(sigma=(0, 1.0)),  # Apply gaussian blur with a sigma between 0 and 1.0
            iaa.Multiply((0.8, 4.2)),  # Change brightness (50-150% of original value)
            iaa.LinearContrast((0.8, 1.2)),  # Adjust contrast
            iaa.AddToHueAndSaturation((-20, 20)),  # Add/Subtract hue and saturation
        ])


class GrassAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-15, 15), translate_percent=(-0.1, 0.1), scale=(0.8, 1.1)),
            iaa.AddToBrightness((0, 20)),
            iaa.WithColorspace(
                    to_colorspace="HSV",
                    from_colorspace="RGB",
                    children=iaa.Sequential([
                        iaa.WithChannels(1, iaa.Add((-5, 5))),  # Randomly adjust saturation
                        iaa.WithChannels(2,iaa.Add((-20, 90)))  # Randomly adjust value/brightness
                    ]) 
                ),
            iaa.Fliplr(0.5),
        ])







AUGMENTATION_TRANSFORMS_Version1 = transforms.Compose([
    AbsoluteLabels(),
    StrongAug(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

AUGMENTATION_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    StrongAug(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

AUGMENTATION_TRANSFORMS_VersionHSV_PAPER = transforms.Compose([
    AbsoluteLabels(),
    GrassAug(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

AUGMENTATION_NONE = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])