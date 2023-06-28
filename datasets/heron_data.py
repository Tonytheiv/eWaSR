import os
import numpy as np
import json
import torch

import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

CUTMIX_IMG_PER_IMG = 10

def colorconstant(rgb, alpha=0.29):
    # check if the input tensor is already on the gpu, if not move it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb = rgb.to(device)

    # ensure the tensor is of type float32
    if rgb.dtype != torch.float32:
        rgb = rgb.float() / 255.0

    # set constants
    beta = 1 - alpha
    K = torch.tensor(0.003, device=device)
    R1 = torch.clamp(rgb[2, :, :], K, 1.0)
    R2 = torch.clamp(rgb[1, :, :], K, 1.0)
    R3 = torch.clamp(rgb[0, :, :], K, 1.0)
    
    F = torch.log(R2) - alpha * torch.log(R1) + beta * torch.log(R3)

    return F

class HeronData(Dataset):
    '''
    datastructure:
    
    root_folder
    │
    ├── data
    │   ├── img1
    │   │   ├── 1.jpg
    │   │   ├── ...
    │   ├── img2
    │   │   ├── 2.jpg
    │   │   ├── ...
    │   ├── ...
    │
    │   ├── jsn1
    │   │   ├── 1.json
    │   │   ├── ...
    │   ├── jsn2
    │   │   ├── 2.json
    │   │   ├── ...
    │   ├── ...
    where jsn1/1.json is the corresponding mask for img1/1.jpg, jsn999/888.json is the corresponding mask for img999/888.jpg, etc.
    '''
    def __init__(self, base_dir, transform=None, normalize_t=None, four_channel_in=False, device='cuda'):
        '''
        Initialize the HeronData dataset.
        
        Args:
            base_dir (str): The base directory path.
            transform (callable, optional): A callable function or transform object to apply to the image and mask.
            normalize_t (callable, optional): A callable function or transform object to normalize the image.
            four_channel_in (bool, optional): Flag indicating whether to add a fourth channel to the input image.
        '''
        self.base_dir = base_dir
        self.transform = transform
        self.normalize_t = normalize_t
        self.four_channel_in = four_channel_in
        self.device = device

        self.data = []
        img_dirs = sorted([d for d in os.listdir(base_dir) if 'img' in d])
        jsn_dirs = sorted([d for d in os.listdir(base_dir) if 'jsn' in d])

        num_labeled_images = 0
        for img_dir, jsn_dir in zip(img_dirs, jsn_dirs):
            img_path = os.path.join(base_dir, img_dir)
            jsn_path = os.path.join(base_dir, jsn_dir)
            img_files = sorted(os.listdir(img_path))

            for img_file in img_files:
                img_file_name = os.path.splitext(img_file)[0]
                jsn_file = f'{img_file_name}.json'
                jsn_file_path = os.path.join(jsn_path, jsn_file)

                if os.path.exists(jsn_file_path):
                    num_labeled_images += 1
                    frame = {
                        'image_path': os.path.join(img_path, img_file),
                        'label_path': jsn_file_path,
                        'name': img_file_name
                    }
                    self.data.append(frame)
        
        print('Number of labeled images: {}'.format(num_labeled_images))

    def __len__(self):
        return len(self.data) * (CUTMIX_IMG_PER_IMG + 1)
    
    def do_rotation(self, image, mask, angle_range=(-45, 45)):
        """
        Rotate the image and mask within the given angle range.
        """
        # convert tensors to PIL Images
        image = TF.to_pil_image(image.cpu())
        mask = TF.to_pil_image(mask.cpu().type(torch.uint8))

        # get random rotation angle
        angle = torch.FloatTensor(1).uniform_(angle_range[0], angle_range[1]).item()

        # rotate image and mask
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # convert back to tensors
        image = TF.to_tensor(image).to(self.device)
        mask = torch.from_numpy(np.array(mask)).to(self.device)

        return image, mask
    
    def adjust_exposure(self, img, exposure_factor):
        # Adjust exposure
        return TF.adjust_brightness(img, exposure_factor)
    
    def do_cutmix(self, img, binary_mask, rand_img, rand_binary_mask):
        # CutMix augmentation
        lam = torch.tensor(np.random.beta(1.0, 1.0)).to(self.device)
        cut_rat = torch.sqrt(1. - lam)
        h, w = img.shape[1:]
        cut_w = torch.round(w * cut_rat).type(torch.long)
        cut_h = torch.round(h * cut_rat).type(torch.long)

        cx = torch.randint(w, (1,)).to(self.device)
        cy = torch.randint(h, (1,)).to(self.device)
        bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
        bby1 = torch.clamp(cy - cut_h // 2, 0, h)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
        bby2 = torch.clamp(cy + cut_h // 2, 0, h)

        img[:, bbx1:bbx2, bby1:bby2] = rand_img[:, bbx1:bbx2, bby1:bby2]
        binary_mask[bbx1:bbx2, bby1:bby2] = rand_binary_mask[bbx1:bbx2, bby1:bby2]

        return img, binary_mask
    
    def do_cutmix_with_rotation(self, img, binary_mask, rand_img, rand_binary_mask, exposure_factor_range=(0, 0)):
        # Generate a random exposure factor within the given range
        exposure_factor = torch.empty(1).uniform_(exposure_factor_range[0], exposure_factor_range[1]).item()

        # Adjust exposure of the random image
        rand_img = self.adjust_exposure(rand_img, 1 + exposure_factor)
        
        # CutMix augmentation
        # Ensure both images have the same number of channels
        if img.shape[0] != rand_img.shape[0]:
            if img.shape[0] == 4:
                # Convert rand_img to RGBA
                rand_img = torch.cat([rand_img, torch.ones((1, rand_img.shape[1], rand_img.shape[2])).to(self.device)], dim=0)
            else:
                # Remove the alpha channel from img
                img = img[:3,:,:]

        # Perform the CutMix operation
        lam = torch.tensor(np.random.beta(1.0, 1.0)).to(self.device)
        cut_rat = torch.sqrt(1. - lam)
        h, w = img.shape[1:]
        cut_w = torch.round(w * cut_rat).type(torch.long)
        cut_h = torch.round(h * cut_rat).type(torch.long)

        cx = torch.randint(w, (1,)).to(self.device)
        cy = torch.randint(h, (1,)).to(self.device)
        bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
        bby1 = torch.clamp(cy - cut_h // 2, 0, h)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
        bby2 = torch.clamp(cy + cut_h // 2, 0, h)

        img[:, bbx1:bbx2, bby1:bby2] = rand_img[:, bbx1:bbx2, bby1:bby2]
        binary_mask[bbx1:bbx2, bby1:bby2] = rand_binary_mask[bbx1:bbx2, bby1:bby2]

        return img, binary_mask

    def do_cutmix_with_rotation_and_overlay(self, img, binary_mask, rand_img, rand_binary_mask):
        """Performs CutMix, rotation, and overlays the result on another randomly selected image."""

        # Apply CutMix
        img, binary_mask = self.do_cutmix(img, binary_mask, rand_img, rand_binary_mask)

        # If img and binary_mask are PyTorch tensors, convert them to CPU and then to PIL images
        if torch.is_tensor(img):
            img = TF.to_pil_image(img.cpu())
        if torch.is_tensor(binary_mask):
            binary_mask = TF.to_pil_image(binary_mask.cpu().type(torch.uint8))

        # Rotate img and binary_mask
        img, binary_mask = self.do_rotation(img, binary_mask)

        # Convert img (RGB) to img_rgba (RGBA) and set black pixels to be completely transparent
        img_rgba = img.convert("RGBA")
        datas = img_rgba.getdata()
        new_data = []
        for item in datas:
            # change all black (also shades of blacks) pixels to be transparent
            if item[0] < 10 and item[1] < 10 and item[2] < 10:
                new_data.append((item[0], item[1], item[2], 0))
            else:
                new_data.append(item)
        img_rgba.putdata(new_data)

        # Convert img_rgba and binary_mask back to tensor
        img_rgba = TF.to_tensor(img_rgba).float().to(self.device)
        binary_mask = TF.to_tensor(binary_mask).to(self.device)

        # Select another image randomly and convert it to tensor
        overlay_img_idx = torch.randint(0, len(self.data), size=(1,)).item()
        overlay_img_path = self.data[overlay_img_idx]['image_path']
        overlay_img = Image.open(overlay_img_path).convert('RGB')
        overlay_img = TF.to_tensor(overlay_img).float().to(self.device)

        # If the overlay image has less than 4 channels, pad the remaining channels
        if overlay_img.shape[0] < 4:
            overlay_img = torch.cat([overlay_img, torch.zeros((1, overlay_img.shape[1], overlay_img.shape[2])).to(self.device)])

        # Overlay img_rgba on overlay_img
        overlay_img = overlay_img * (img_rgba[3:4,:,:] == 0).float() + img_rgba

        # Convert the image back to RGB by dropping the alpha channel
        overlay_img = overlay_img[:3,:,:]

        return overlay_img, binary_mask

    def __getitem__(self, idx, classes=2):
        '''
        Get a sample from the dataset at the given index.
        
        Args:
            idx (int): The index of the sample.
            classes (int): Number of classes for one-hot encoding. Default is 2.
        
        Returns:
            tuple: A tuple containing the features and metadata of the sample.
        '''
        idx, cutmix_number = divmod(idx, CUTMIX_IMG_PER_IMG + 1)
        frame = self.data[idx]
        img = Image.open(frame['image_path']).convert('RGB')
        with open(frame['label_path']) as f:
            mask_dict = json.load(f)
        binary_mask = np.array(mask_dict['binary_mask'])

        if self.transform is not None:
            res = self.transform(image=np.array(img), mask=binary_mask)
            img, binary_mask = res['image'], res['mask']

        if self.normalize_t is not None:
            img = self.normalize_t(img)
        else:
            img = TF.to_tensor(img)

        img = img.to(self.device)
        binary_mask = torch.from_numpy(binary_mask).to(self.device)

        if self.four_channel_in:
            # Convert img to tensor if not and send to GPU
            if type(img) != torch.Tensor:
                img = torch.from_numpy(np.array(img)).to(self.device)

            Fg = colorconstant(img, alpha=0.29)

            # Make sure Fg has an extra dimension in the beginning
            if len(Fg.shape) == 2:
                Fg = Fg.unsqueeze(0)

            img = torch.cat([img, Fg], dim=0)

        if cutmix_number > 0:
            rand_idx = np.random.choice(len(self.data))
            rand_frame = self.data[rand_idx]
            rand_img = Image.open(rand_frame['image_path']).convert('RGB')
            with open(rand_frame['label_path']) as f:
                rand_mask_dict = json.load(f)
            rand_binary_mask = np.array(rand_mask_dict['binary_mask'])

            if self.transform is not None:
                res = self.transform(image=np.array(rand_img), mask=rand_binary_mask)
                rand_img, rand_binary_mask = res['image'], res['mask']

            if self.normalize_t is not None:
                rand_img = self.normalize_t(rand_img)
            else:
                rand_img = TF.to_tensor(rand_img)

            rand_img = rand_img.to(self.device)
            rand_binary_mask = torch.from_numpy(rand_binary_mask).to(self.device)
            
            # regular cutmix
            # img, binary_mask = self.do_cutmix(img, binary_mask, rand_img, rand_binary_mask)
            
            # cutmix pro
            img, binary_mask = self.do_cutmix_with_rotation(img, binary_mask, rand_img, rand_binary_mask, exposure_factor_range=(-0.1, 0.4))
            
            # cutmix pro max
            # img, binary_mask = self.do_cutmix_with_rotation_and_overlay(img, binary_mask, rand_img, rand_binary_mask)

        features = {'image': img}

        metadata_fields = ['image_path', 'name']
        metadata = {field: frame[field] for field in metadata_fields}

        onehot_mask = torch.zeros((binary_mask.shape[0], binary_mask.shape[1], classes), device=self.device)
        for class_idx in range(classes):
            onehot_mask[binary_mask == class_idx] = torch.eye(classes, device=self.device)[class_idx]

        metadata['segmentation'] = onehot_mask.permute(2, 0, 1)
        metadata['label'] = binary_mask

        return features, metadata

if __name__ == "__main__":
    from transforms import get_augmentation_transform

    transform = get_augmentation_transform(albu=False)

    dataset = HeronData("/home/tony/Videos/data", transform=transform)
    print("Number of entries:", len(dataset))
    num_lilypads = 0

    for i, (feature, metadata) in enumerate(dataset):
        img = feature['image']
        if img.shape[0] == 4:  # If four channels
            img = img[0:3]

        label = metadata['label']
        if label.max() == 2:
            num_lilypads += 1
        
        # Show image and label
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        plt.title('RGB')
        plt.subplot(1, 2, 2)
        plt.imshow(label.cpu().numpy(), cmap='gray')
        plt.title('Label')
        plt.tight_layout()
        plt.show()

    print("Number images with lilypads:", num_lilypads)
