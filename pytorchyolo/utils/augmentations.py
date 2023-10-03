import imgaug.augmenters as iaa
from torchvision import transforms
from pytorchyolo.utils.transforms import ToTensor, PadSquare, RelativeLabels, AbsoluteLabels, ImgAug, adjustGrassColor
import imgaug  as ia

class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.1)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-10, 10)),
            iaa.Fliplr(0.5),
        ])


class StrongAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            # iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.1)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-20, 20)),
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





class GrassAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            # iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.1)),
            iaa.WithColorspace(
                    to_colorspace="HSV",
                    from_colorspace="RGB",
                    children=iaa.Sequential([
                      #  iaa.WithChannels(1, iaa.Add((10, 60))),  # Randomly adjust saturation
                      #  iaa.WithChannels(2, iaa.Multiply((0.9, 1.3)))  # Randomly adjust value/brightness
                    iaa.WithChannels(1, iaa.Add((0, 200))),  # Randomly adjust saturation
                    iaa.WithChannels(2, iaa.Multiply((1.0, 4.0)))  # Randomly adjust value/brightness
                    ])
                ),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)
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