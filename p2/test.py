import os
import sys
import csv
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models import ResNet50_Weights
import imageio.v2 as imageio
#Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "DLCV_hw1_models/p2_model.pth"
val_img_dir, output_dir = sys.argv[1], sys.argv[2]
val_transform = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_size = (512, 512)
# Define the color-to-class mapping
color_to_class = {
    (0, 255, 255): 0,   # Urban
    (255, 255, 0): 1,   # Agriculture
    (255, 0, 255): 2,    # Rangeland
    (0, 255, 0): 3,     # Forest
    (0, 0, 255): 4,     # Water
    (255, 255, 255): 5, # Barren
    (0, 0, 0): 6        # Unknown
}
# Reverse the color_to_class dictionary to create a class_to_color mapping
class_to_color = {v: k for k, v in color_to_class.items()}

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir

        self.img_names = sorted([img_name for img_name in os.listdir(self.image_dir) if img_name.endswith(".jpg")])
        self.transform = transform
        
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.img_names[idx])).copy()
        if self.transform:
            img = self.transform(img)
        return img, self.img_names[idx]

    def __len__(self):
        return len(self.img_names)

class Deeplabv3_Resnet50_Model(nn.Module):
    def __init__(self):
        super(Deeplabv3_Resnet50_Model, self).__init__()
        self.model = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.DEFAULT,
            weights_backbone=ResNet50_Weights.DEFAULT,
        )
        self.model.classifier[4] = nn.Sequential(nn.Conv2d(256, 7, 1, 1))

    def forward(self, x):
        output = self.model(x)
        return output["out"]
    
def mean_iou_score(pred, labels):
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp) if (tp_fp + tp_fn - tp) != 0 else 0
        mean_iou += iou / 6
    return mean_iou

def pred2image(batch_preds, batch_names, out_path):
    colors = {
        0: [0, 255, 255],
        1: [255, 255, 0],
        2: [255, 0, 255],
        3: [0, 255, 0],
        4: [0, 0, 255],
        5: [255, 255, 255],
        6: [0, 0, 0],
    }
    for pred, name in zip(batch_preds, batch_names):
        pred = pred.detach().cpu().numpy()
        pred_img = np.zeros((512, 512, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            pred_img[np.where(pred == class_id)] = color
        imageio.imwrite(os.path.join(out_path, name.replace("sat.jpg", "mask.png")), pred_img)

# Define an inference function
def visualize_predictions(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)
            # print(outputs.size())
            outputs = torch.argmax(outputs, dim=1)  # Get the class with the highest score
            
            for i in range(images.size(0)):
                mask_pred = outputs[i].cpu().numpy()
                
                # Convert class indices back to RGB colors
                mask_rgb = np.zeros((mask_pred.shape[0], mask_pred.shape[1], 3), dtype=np.uint8)
                for class_idx, rgb in class_to_color.items():  # Use class_to_color instead of color_to_class
                    mask_rgb[mask_pred == class_idx] = rgb  # No need to convert to a list
                mask_output_path = os.path.join(output_dir, filenames[i].replace("sat.jpg", "mask.png"))
                imageio.imwrite(mask_output_path, mask_rgb)
                # print(f"Saved predicted mask for {filenames[i]} as {mask_output_path}")
                
# Load data
val_data = SegmentationDataset(val_img_dir, transform=val_transform)
val_loader = DataLoader(dataset=val_data, batch_size=16)

# Define model
model = Deeplabv3_Resnet50_Model().to(device)
model.load_state_dict(torch.load(model_path)['model_state_dict'])
visualize_predictions(model, val_loader, device)


