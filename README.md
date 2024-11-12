# Semantic Segmentation
Comparing the semantic segmentation performance of two finetuned models. The key difference between this model and the baseline VGG16- FCN32 is the depth of the model, deeplabv3-resnet50 is much larger than the baseline model, so the performance is much higher. The deeplabv3-resnet50 model can handle details more accurately, the comparison is as follows:

1. VGG16-FCN32s: Since the upsampling is done directly from a low-resolution feature map, fine details and boundaries are often blurred or lost, making it less accurate for tasks that require precise boundary delineation.
2. DeepLabV3-ResNet50: Atrous convolutions combined with the ASPP module help
DeepLabV3 retain much more detail along object boundaries, making it significantly better at handling tasks where fine object details or thin structures need to be segmented.

## Validation Result
VGG16-FCN32 mIoU: 0.5021 

DeepLabv3_Resnet50 mIoU: 0.7353

## Usage
```
bash get_dataset.sh
bash download_ckpt.sh
bash test.sh <val_img_dir> <output_dir>
