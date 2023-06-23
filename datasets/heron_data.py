import os
import numpy as np
import json
import torch

import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image

CUTMIX_IMG_PER_IMG = 10

def colorconstant(rgb, alpha=0.29):
    if rgb.dtype == torch.float32:
        rgb = rgb.numpy().transpose(1, 2, 0)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    
    
    beta = 1 - alpha
    K = 0.003
    R1 = np.clip(rgb[:, :, 2], K, 1.0) 
    R2 = np.clip(rgb[:, :, 1], K, 1.0)
    R3 = np.clip(rgb[:, :, 0], K, 1.0)
    
    F = np.log(R2) -  alpha * np.log(R1) + beta * np.log(R3)
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
    
    def __init__(self, base_dir, transform=None, normalize_t=None, four_channel_in=False):
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

        self.data = []
        img_dirs = sorted([d for d in os.listdir(base_dir) if 'img' in d])
        jsn_dirs = sorted([d for d in os.listdir(base_dir) if 'jsn' in d])
        # import pdb; pdb.set_trace()
        num_labeled_images = 0
        for img_dir, jsn_dir in zip(img_dirs, jsn_dirs):
            img_path = os.path.join(base_dir, img_dir)
            jsn_path = os.path.join(base_dir, jsn_dir)
            img_files = sorted(os.listdir(img_path))

            for img_file in img_files:
                img_file_name = os.path.splitext(img_file)[0]  # extract filename without extension
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
        '''
        Get the length of the dataset.
        
        Returns:
            int: The length of the dataset.
        '''
        return len(self.data) * (CUTMIX_IMG_PER_IMG + 1)
    
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

        if self.four_channel_in:
            Fg = colorconstant(np.array(img), alpha=0.29)
            Fg = torch.from_numpy(Fg).unsqueeze(0)
            img = torch.cat([img, Fg], dim=0)

        if cutmix_number > 0:
            # CutMix augmentation
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

            # Do CutMix
            lam = np.random.beta(1.0, 1.0)
            cut_rat = np.sqrt(1. - lam)
            h, w = img.shape[1:]
            cut_w = int(w * cut_rat)
            cut_h = int(h * cut_rat)

            cx = np.random.randint(w)
            cy = np.random.randint(h)
            bbx1 = np.clip(cx - cut_w // 2, 0, w)
            bby1 = np.clip(cy - cut_h // 2, 0, h)
            bbx2 = np.clip(cx + cut_w // 2, 0, w)
            bby2 = np.clip(cy + cut_h // 2, 0, h)

            img[:, bbx1:bbx2, bby1:bby2] = rand_img[:, bbx1:bbx2, bby1:bby2]
            binary_mask[bbx1:bbx2, bby1:bby2] = rand_binary_mask[bbx1:bbx2, bby1:bby2]

        features = {'image': img}

        metadata_fields = ['image_path', 'name']
        metadata = {field: frame[field] for field in metadata_fields}

        onehot_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], classes))
        for class_idx in range(classes):
            onehot_mask[binary_mask == class_idx] = np.eye(classes)[class_idx]

        metadata['segmentation'] = TF.to_tensor(onehot_mask)
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
        plt.imshow(img.permute(1, 2, 0))
        plt.title('RGB')
        plt.subplot(1, 2, 2)
        plt.imshow(label, cmap='gray')
        plt.title('Label')
        plt.tight_layout()
        plt.show()

    print("Number images with lilypads:", num_lilypads)
