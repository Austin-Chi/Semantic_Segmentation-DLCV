import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights,resnet101, ResNet101_Weights, alexnet, AlexNet_Weights, efficientnet_b0, EfficientNet_B0_Weights, vgg16, VGG16_Weights, mobilenet_v2, MobileNet_V2_Weights, ResNet50_Weights, ResNet101_Weights
from torchvision.models.segmentation import (deeplabv3_resnet50,deeplabv3_resnet101,DeepLabV3_ResNet50_Weights,DeepLabV3_ResNet101_Weights)
import os
import imageio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import logging
import csv
import random



#Configuration
logging.basicConfig(filename='VGG.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
checkpoint_dir = './p2/checkpoints_VGG'
os.makedirs(checkpoint_dir, exist_ok=True)
log_file = 'p2/VGG_log.csv'
train_dir = './train'
val_dir = './validation'
num_epochs = 100
num_classes = 7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(device)
learning_rate = 1e-3
batch_size = 8
momentum = 0.9
weight_decay = 0.0001

# Load pretrained DeepLabV3 model with ResNet50 backbone
# Modify VGG16 to FCN32s
class VGG16_FCN32s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Use VGG16 features
        self.features = models.vgg16(weights=VGG16_Weights.DEFAULT).features

        # Optimized classifier with reduced number of parameters and batch normalization
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.BatchNorm2d(4096),  # Add batch normalization
            nn.ReLU(),
            nn.Conv2d(4096, 4096, 1),
            nn.BatchNorm2d(4096),  # Add batch normalization
            nn.ReLU(),
        )

        # Adjusted upsample layer
        self.upsample = nn.ConvTranspose2d(
            in_channels=4096,
            out_channels=num_classes,
            kernel_size=44,
            stride=52,
            padding=0
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.upsample(x)
        return x

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

# Function to save a checkpoint
def save_checkpoint(epoch, model, optimizer, val_loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(checkpoint, checkpoint_path)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, use_aug=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.use_aug = use_aug
        self.image_names = sorted(list(set([f.split('_')[0] for f in os.listdir(image_dir)])))
        self.resize = transforms.Compose([transforms.Resize(512, transforms.InterpolationMode.NEAREST)])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, f"{image_name}_sat.jpg")
        mask_path = os.path.join(self.mask_dir, f"{image_name}_mask.png")
        
        image = Image.open(image_path).convert("RGB")
        # mask = np.array(Image.open(mask_path).convert("RGB"))  # single channel (grayscale)
        mask = Image.open(mask_path).convert("RGB")  # single channel (grayscale)

        if self.use_aug:
            image, mask = self._apply_transform(image, mask)
        if self.transform is None:
            image = self.totensor(image)
        mask = np.array(mask)

        
        # Convert mask to class labels
        mask_labels = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for rgb, label in color_to_class.items():
            mask_labels[np.all(mask == rgb, axis=-1)] = label
        
        if self.transform:
            image = self.transform(image)
        
        mask_tensor = torch.tensor(mask_labels, dtype=torch.long)
        
        return image, mask_tensor

    def _apply_transform(self, image, mask):
        if random.random() < 0.5:
            image.transpose(Image.FLIP_LEFT_RIGHT)
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image.transpose(Image.FLIP_TOP_BOTTOM)
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            image.rotate(45)
            mask.rotate(45)
        params = transforms.RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(1, 1))
        image = self.resize(transforms.functional.crop(image, *params))
        mask = self.resize(transforms.functional.crop(mask, *params))
        return image, mask

# class SegmentationDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None, use_aug=False):
#         self.image_dir = image_dir
#         self.path = image_dir
#         self.mask_dir = mask_dir
#         self.data = []
#         # self.p = 0.5

#         self.imgfile = sorted([img for img in os.listdir(self.image_dir) if img.endswith(".jpg")])
#         self.maskfile = sorted([mask for mask in os.listdir(self.mask_dir) if mask.endswith(".png")])
#         self.mask_label = np.empty((len(self.maskfile), 512, 512))

#         if len(self.maskfile):
#             for i, (img, mask) in enumerate(zip(self.imgfile, self.maskfile)):
#                 path = os.path.join(self.path, img)
#                 self.data.append(Image.open(path).copy())
#                 mask = imageio.imread(os.path.join(self.path, mask)).copy()
#                 mask = (mask >= 128).astype(int)
#                 mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
#                 self.mask_label[i, mask == 3] = 0  # (Cyan: 011) Urban land
#                 self.mask_label[i, mask == 6] = 1  # (Yellow: 110) Agriculture land
#                 self.mask_label[i, mask == 5] = 2  # (Purple: 101) Rangeland
#                 self.mask_label[i, mask == 2] = 3  # (Green: 010) Forest land
#                 self.mask_label[i, mask == 1] = 4  # (Blue: 001) Water
#                 self.mask_label[i, mask == 7] = 5  # (White: 111) Barren land
#                 self.mask_label[i, mask == 0] = 6  # (Black: 000) Unknown

#             else:
#                 for img in self.imgfile:
#                     self.data.append(Image.open(os.path.join(self.path, img)).copy())
#         self.transform = transform
#         self.resize = transforms.Compose([transforms.Resize(512, transforms.InterpolationMode.NEAREST)])
#         self.use_aug = use_aug

    # def __getitem__(self, idx):
    #     data = self.data[idx]
    #     data = self.transform(data)

    #     if len(self.mask_label):
    #         mask_label = self.mask_label[idx].copy()
    #         if self.use_aug:
    #             if np.random.rand() < 0.5:
    #                 mask_label = np.flip(mask_label, axis=1)
    #                 data = transforms.functional.hflip(data)
    #             if np.random.rand() < 0.5:
    #                 mask_label = np.flip(mask_label, axis=0)
    #                 data = transforms.functional.vflip(data)
    #             params = transforms.RandomResizedCrop.get_params(data, scale=(0.8, 1.0), ratio=(1, 1))
    #             data = self.resize(transforms.functional.crop(data, *params))
    #             mask_label = self.resize(transforms.functional.crop(mask_label, *params))

    #         return data, mask_label.copy()
    #     else:
    #         return data

    # def __len__(self):
    #     return len(self.imgfile)

# Data transformations
transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and Dataloader
train_dataset = SegmentationDataset(train_dir, train_dir, transform=transform, use_aug=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

val_dataset = SegmentationDataset(val_dir, val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

def train(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images, masks = images.to(device), torch.Tensor(masks).long().to(device)
        
        optimizer.zero_grad()
        outputs = model(images)#['out']  # Deeplabv3 returns a dictionary with 'out' as the key
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    scheduler.step()
    
    return running_loss / len(dataloader)

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    # print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss, val_pred_list, val_mask_list = [], [], []

    # with torch.no_grad():
    #     for images, masks in dataloader:
    #         images = images.to(device)
            
    #         outputs = model(images)#['out']  # Deeplabv3 returns a dictionary with 'out' as the key
    #         preds = torch.Tensor.numpy(torch.argmax(outputs, dim=1).cpu())
    #         all_pred.append(preds)
    #         all_mask.append(masks)
    #     loss = mean_iou_score(np.concatenate(all_pred, axis=0), np.concatenate(all_mask, axis=0))
    #     val_loss += loss.item()

    for imgs, masks in dataloader:
        imgs = imgs.float().to(device)
        masks_t = torch.Tensor(masks).long().to(device)
        with torch.no_grad():
            output = model(imgs)
        val_pred_list.append(output.cpu().argmax(dim=1))
        val_mask_list.append(masks)

    val_pred_list = np.concatenate(val_pred_list, axis=0)
    val_mask_list = np.concatenate(val_mask_list, axis=0)
    val_iou = mean_iou_score(val_pred_list, val_mask_list)

    return val_iou# / len(dataloader)


def visualize_predictions(model, dataloader, device, epoch):
    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
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
                
                # Plot original image, true mask, and predicted mask
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                true_mask = masks[i].cpu().numpy()
                # Convert class indices back to RGB colors
                true_mask_rgb = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)
                for class_idx, rgb in class_to_color.items():  # Use class_to_color instead of color_to_class
                    true_mask_rgb[true_mask == class_idx] = rgb  # No need to convert to a list
                
                
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.title('Original Image')
                
                plt.subplot(1, 3, 2)
                plt.imshow(true_mask_rgb)
                plt.title('True Mask')
                
                plt.subplot(1, 3, 3)
                plt.imshow(mask_rgb)
                plt.title('Predicted Mask')
                
                # Save the plot
                plot_filename = f'Epoch_{epoch}_{i}.png'
                plt.savefig(os.path.join('p2/images/', plot_filename))
                plt.close()  # Close the figure to avoid displaying

            break  # Visualize only the first batch



model = VGG16_FCN32s(num_classes).to(device)
# checkpoint = torch.load('p2/checkpoints_DEEPLAB_RESNET50/vgg_best_epoch_248.pth')

# # Load the model state dict from the checkpoint
# model.load_state_dict(checkpoint['model_state_dict'])

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150 * len(train_loader), eta_min=1e-6)
criterion = FocalLoss(alpha=1.0, gamma=2.0)


# Training loop
best_val_loss = 0
for epoch in range(751, 751+num_epochs):
    train_loss = train(model, train_loader, optimizer, scheduler, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)
    
    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Save validation accuracy to CSV file
    with open(log_file, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_loss, val_loss])
    
    # Save the model if validation accuracy improves
    if val_loss > best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = os.path.join(checkpoint_dir, f'vgg_best_epoch_{epoch+1}.pth')
        save_checkpoint(epoch + 1, model, optimizer, val_loss, checkpoint_path)
        torch.save(model, 'p2/best_vgg.pth')
        logging.info(f'Best model saved at epoch {epoch+1} with Val Score: {val_loss:.4f}')
        # visualize_predictions(model, val_loader, device, epoch)
        logging.info('Predictions Visualized')
    elif epoch == 0 or epoch == num_epochs/2 or epoch == num_epochs-1:
        checkpoint_path = os.path.join(checkpoint_dir, f'vgg_best_epoch_{epoch+1}.pth')
        save_checkpoint(epoch + 1, model, optimizer, val_loss, checkpoint_path)
        logging.info(f'Checkpoint model saved at epoch {epoch+1} with Val Score: {val_loss:.4f}')
        
    