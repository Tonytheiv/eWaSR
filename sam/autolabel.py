import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import cv2
import json
import shutil
import os
import glob
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class Autolabel:
  def __init__(self, checkpoint, type, imgs_dir, jsons_dir):
    '''
    sorts the images and jsons in the directories, renames them properly and autolabels the images using sam
    '''
    # checkpoint is a path
    # type is one of the following: {'vit_h', 'vit_l', 'vit_b'}

    self.checkpoint = checkpoint
    self.type = type
    self.imgs_dir = imgs_dir
    self.jsons_dir = jsons_dir

    # Fetch all images from the directory and convert to RGB
    # Also, fetch all image paths and store in self.imgpaths
    self.imgs = []
    self.imgpaths = []
    img_files = sorted(os.listdir(self.imgs_dir))  # sort the file names

    # First, add a prefix to every image file in the directory
    prefix = 'old_'
    for img in img_files:
      if img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg'):
        os.rename(os.path.join(self.imgs_dir, img), os.path.join(self.imgs_dir, prefix + img))

    # Then, perform the renaming to the ordered names
    img_files = sorted(os.listdir(self.imgs_dir))  # get the new list with the prefixed names
    for idx, img in enumerate(img_files, start=1):  # start enumerating from 1
      if img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg'):
        img_path = os.path.join(imgs_dir, img)
        self.imgs.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        self.imgpaths.append(img_path)

        new_name = str(idx) + '.' + img.split('.')[-1]  # construct new name
        os.rename(img_path, os.path.join(imgs_dir, new_name))

    # Lastly, delete all the files that have the prefix
    for img in os.listdir(self.imgs_dir):
      if img.startswith(prefix):
        os.remove(os.path.join(self.imgs_dir, img))

    # Fetch all json files from the directory
    self.jsons = [json.load(open(os.path.join(jsons_dir, json_file))) 
                  for json_file in os.listdir(jsons_dir) if json_file.endswith('.json')]
    
    self.sam = sam_model_registry[self.type](checkpoint=self.checkpoint)
    self.sam.to(device='cuda')
    self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    # Generate masks for each image
    self.masks = [self.mask_generator.generate(img) for img in self.imgs]
    
    # Initialize
    self.coco_data = [None] * len(self.jsons)
  
  def showAnns(self, anns):
    '''
    process masks segmented by sam
    '''
    if len(anns) == 0:
      return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
      m = ann['segmentation']
      color_mask = np.concatenate([np.random.random(3), [0.35]])
      img[m] = color_mask
    ax.imshow(img)

  def displayMask(self, index):
    '''
    visualize masks segmented by sam
    '''
    plt.figure(figsize=(20,20))
    plt.imshow(self.imgs[index])
    self.showAnns(self.masks[index])
    plt.axis('off')
    plt.show()
    
  def displayJson(self, index):
    '''
    visualizes a labeled mask using the json file in COCO format
    '''
    # Add JSON file
    json_path = f"{self.jsons_dir}/{index + 1}.json"
    
    # Load the COCO data
    with open(json_path, 'r') as f:
      self.coco_data = json.load(f)
      
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(self.imgs[index])
    
    # Iterate over the coco_data list
    for data_dict in self.coco_data:
      # Extract relevant information from the COCO data
      annotations = data_dict['annotation']['annotationGroups'][0]['annotationEntities']
      
      # Iterate over the annotations
      for annotation in annotations:
        segmentations = annotation['annotationBlocks'][0]['segments']
        name = annotation['name']

        # Handle different segmentation formats
        if isinstance(segmentations, list):
          polygons = [np.array(segmentation).reshape(-1, 2) for segmentation in segmentations]
        else:
          polygons = [np.array(segmentations).reshape(-1, 2)]

        # Set the color and opacity based on the name
        if name == 'water':
          facecolor = 'red'
          alpha = 0.4
        else:
          facecolor = 'none'
          alpha = 1.0

        # Create polygon patches and add them to the axes
        for polygon in polygons:
          patch = Polygon(polygon, edgecolor='blue', facecolor=facecolor, alpha=alpha)
          ax.add_patch(patch)
        
    # Turn off the axis labels
    ax.axis('off')

    # Show the plot
    plt.show()
    
  def visualizeBinary(self, index, cmpr_height, output_dir=None, t=3):
    '''
    heuristic is applied here to find where water is. This can also visualize the label using plt
    '''
    mask_data = self.masks[index]
    img_shape = self.imgs[index].shape[:2]
    # Find the largest bottom touching mask
    largest_bottom_touching_mask_id = -1
    largest_bottom_touching_mask_area = 0

    for i, annotation in enumerate(mask_data):
      mask = annotation["segmentation"]
      if np.any(mask[cmpr_height, :]):  # If any pixel in the bottom row of the mask is True
        mask_area = np.sum(mask[cmpr_height, :])  # Count of True values in the last row
        print(i, mask_area)
        if mask_area > largest_bottom_touching_mask_area:
          largest_bottom_touching_mask_id = i
          largest_bottom_touching_mask_area = mask_area

    # Create a new binary mask filled with False values
    binary_mask = np.full(img_shape, False)

    # Set the pixels of the largest bottom touching mask in the new binary mask to True
    if largest_bottom_touching_mask_id != -1:
        binary_mask = np.where(mask_data[largest_bottom_touching_mask_id]["segmentation"], True, binary_mask)

    # Apply the second heuristic to the binary mask
    binary_mask = self.fixBottom(binary_mask, t)

    # Replace the original masks with the new binary mask
    self.masks[index] = [{"segmentation": binary_mask, "area": largest_bottom_touching_mask_area}]

    # Display the binary mask
    plt.figure(figsize=(20,20))
    plt.imshow(self.imgs[index])
    self.showAnns(self.masks[index])
    plt.axis('off')

    if output_dir:
      # Create the directory if not exists
      os.makedirs(output_dir, exist_ok=True)
      
      # Save the figure to the output_dir if provided
      filename = os.path.join(output_dir, f"{str(index)}.jpg")
      plt.savefig(filename)
    else:
      # Show the image
      plt.show()
    plt.close()

  def fixBottom(self, binary_mask, t):
    binary_mask_np = np.array(binary_mask)

    # Loop through rows starting from t until 1
    for i in range(t, 0, -1):
      target_row = binary_mask_np[-i]
      
      for column_index, value in enumerate(target_row):
        if value == 1:
          for j in range(i - 1, 0, -1):  # Fill all subsequent rows
            binary_mask_np[-j, column_index] = 1
            
    return binary_mask_np
      
  def outputJsonBinary(self, index):
    '''
    outputs the label as a json file in binary mask format
    '''
    binary_mask_data = self.masks[index]
    # Convert boolean mask to int mask as json doesn't support numpy array and boolean datatype
    int_mask = binary_mask_data[0]['segmentation'].astype(int).tolist()
    
    # Create a dictionary to store mask data
    binary_mask_dict = {
      "image": self.imgpaths[index],
      "binary_mask": int_mask,
      "area": int(binary_mask_data[0]['area'])
    }

    # Write the dictionary to a JSON file
    with open(f"{self.jsons_dir}/{index + 1}.json", 'w') as outfile:
      json.dump(binary_mask_dict, outfile)

  def visualizeJsonBinary(self, index):
    '''
    using the json file, visualize the binary mask of an img
    '''
    # Load the binary mask from the JSON file
    json_path = f"{self.jsons_dir}/{index + 1}.json"
    with open(json_path, 'r') as f:
      binary_mask_dict = json.load(f)
      
    # Define a colormap where 0 values are transparent and 1 values are red
    cmap = ListedColormap(['none', 'red'])
    
    # Convert binary mask back to a numpy array
    binary_mask = np.array(binary_mask_dict["binary_mask"]).astype(bool)
    
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display the original image
    ax.imshow(self.imgs[index])
    
    # Overlay the binary mask on top of the image
    ax.imshow(binary_mask, alpha=0.5, cmap=cmap)  # Show mask as an overlay with transparency
    
    # Turn off the axis labels
    ax.axis('off')

    # Show the plot
    plt.show()

  def outputJson(self, index, hull=True):
    '''
    outputs the label as a json file in COCO format
    '''
    # Read mask data
    mask_data = self.masks[index]
    img_id = index

    output_json = []  # Initialize output

    for annotation in mask_data:
      binary_mask = annotation['segmentation']
      all_points = []
      # find_contours returns a list of found contours in the image
      for contour in find_contours(binary_mask, 0.5):
        # reshape contour array into sequence of coordinate pairs
        contour = np.fliplr(contour).reshape(-1, 2).tolist()  # Flip x and y
        if len(contour) > 2:  # Only consider contours with more than 1 point (x,y)
          all_points.extend(contour)

      if hull:  # If hull is True, compute the convex hull of the points
        h = ConvexHull(all_points)
        hull_points = [all_points[i] for i in h.vertices]
          
        # Create the output json dictionary
        block = {
          "id": str(img_id),
          "settings": {"color": "red", "closed": True},
          "segments": [hull_points]
        }

      else:
        # Create the output json dictionary
        block = {
          "id": str(img_id),
          "settings": {"color": "red", "closed": True},
          "segments": [all_points]
        }

      entity = {
        "id": str(img_id),
        "name": "water",
        "annotationBlocks": [block]
      }

      group = {
        "id": str(img_id),
        "annotationEntities": [entity]
      }

      doc = {
        "directory": self.jsons_dir,
        "name": self.imgpaths[index],
        "id": str(img_id)
      }

      output_json.append({"documents": [doc], "annotation": {"id": str(img_id), "version": "2.0", "annotationGroups": [group]}})

    # Output to a json file
    with open(os.path.join(self.jsons_dir, str(index + 1) + ".json"), 'w') as f:
      json.dump(output_json, f)

  def cutMix(self, num):
    '''
    num is the number of cutMix images to generate
    '''
    imgs = sorted(os.listdir(self.imgs_dir), key=lambda x: int(x.split('.')[0]))  # sort the file names
    last_file_number = int(os.path.splitext(imgs[-1])[0])

    for i in range(num):
      img1_filename = np.random.choice(imgs)
      img2_filename = np.random.choice(imgs)

      img1 = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_dir, img1_filename)), cv2.COLOR_BGR2RGB)
      img2 = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_dir, img2_filename)), cv2.COLOR_BGR2RGB)

      mask1 = self.loadMask(os.path.join(self.jsons_dir, os.path.splitext(img1_filename)[0] + '.json'))
      mask2 = self.loadMask(os.path.join(self.jsons_dir, os.path.splitext(img2_filename)[0] + '.json'))

      lam = np.random.beta(1.0, 1.0)
      cut_rat = np.sqrt(1. - lam)
      h, w = img1.shape[0], img1.shape[1]
      cut_w = int(w * cut_rat)
      cut_h = int(h * cut_rat)

      cx = np.random.randint(w)
      cy = np.random.randint(h)
      bbx1 = np.clip(cx - cut_w // 2, 0, w)
      bby1 = np.clip(cy - cut_h // 2, 0, h)
      bbx2 = np.clip(cx + cut_w // 2, 0, w)
      bby2 = np.clip(cy + cut_h // 2, 0, h)

      img1[bbx1:bbx2, bby1:bby2, :] = img2[bbx1:bbx2, bby1:bby2, :]
      mask1[bbx1:bbx2, bby1:bby2] = mask2[bbx1:bbx2, bby1:bby2]

      new_file_number = last_file_number + i + 1
      new_img_filename = f"{new_file_number}.jpg"
      new_json_filename = f"{new_file_number}.json"

      cv2.imwrite(os.path.join(self.imgs_dir, new_img_filename), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
      self.saveMask(os.path.join(self.jsons_dir, new_json_filename), mask1)

  def loadMask(self, path):
    with open(path, 'r') as f:
      json_data = json.load(f)
      binary_mask = np.array(json_data['binary_mask'], dtype=np.uint8)
    return binary_mask

  def saveMask(self, path, mask):
    binary_mask_dict = {
      "image": path,
      "binary_mask": mask.tolist(),
      "area": int(np.sum(mask))
    }

    with open(path, 'w') as outfile:
      json.dump(binary_mask_dict, outfile)

class CutMix:
  def __init__(self, imgs_dir, jsons_dir):
    self.imgs_dir = imgs_dir
    self.jsons_dir = jsons_dir

  def cutMix(self, num):
    '''
    num is the number of cutMix images to generate
    '''
    imgs = sorted(os.listdir(self.imgs_dir), key=lambda x: int(x.split('.')[0]))  # sort the file names
    last_file_number = int(os.path.splitext(imgs[-1])[0])

    for i in range(num):
      img1_filename = np.random.choice(imgs)
      img2_filename = np.random.choice(imgs)

      img1 = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_dir, img1_filename)), cv2.COLOR_BGR2RGB)
      img2 = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_dir, img2_filename)), cv2.COLOR_BGR2RGB)

      mask1 = self.loadMask(os.path.join(self.jsons_dir, os.path.splitext(img1_filename)[0] + '.json'))
      mask2 = self.loadMask(os.path.join(self.jsons_dir, os.path.splitext(img2_filename)[0] + '.json'))

      lam = np.random.beta(1.0, 1.0)
      cut_rat = np.sqrt(1. - lam)
      h, w = img1.shape[0], img1.shape[1]
      cut_w = int(w * cut_rat)
      cut_h = int(h * cut_rat)

      cx = np.random.randint(w)
      cy = np.random.randint(h)
      bbx1 = np.clip(cx - cut_w // 2, 0, w)
      bby1 = np.clip(cy - cut_h // 2, 0, h)
      bbx2 = np.clip(cx + cut_w // 2, 0, w)
      bby2 = np.clip(cy + cut_h // 2, 0, h)

      img1[bbx1:bbx2, bby1:bby2, :] = img2[bbx1:bbx2, bby1:bby2, :]
      mask1[bbx1:bbx2, bby1:bby2] = mask2[bbx1:bbx2, bby1:bby2]

      new_file_number = last_file_number + i + 1
      new_img_filename = f"{new_file_number}.jpg"
      new_json_filename = f"{new_file_number}.json"

      cv2.imwrite(os.path.join(self.imgs_dir, new_img_filename), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
      self.saveMask(os.path.join(self.jsons_dir, new_json_filename), mask1)

  def loadMask(self, path):
    with open(path, 'r') as f:
      json_data = json.load(f)
      binary_mask = np.array(json_data['binary_mask'], dtype=np.uint8)
    return binary_mask

  def saveMask(self, path, mask):
    binary_mask_dict = {
      "image": path,
      "binary_mask": mask.tolist(),
      "area": int(np.sum(mask))
    }

    with open(path, 'w') as outfile:
      json.dump(binary_mask_dict, outfile)

class Autodelete:
  def __init__(self, img_dir, jsn_dir, img_id):
    self.img_dir = img_dir
    self.jsn_dir = jsn_dir
    self.img_id = sorted(img_id)  # sort the ids for proper renaming later

  def delete(self):
    '''
    delete the selected files from the directories, both the img and its associated json file
    '''
    if self.img_id == []:
      print("No images to delete!")
      return
    try:
      # Delete the selected files
      for id in self.img_id:
        img_files = glob.glob(f"{self.img_dir}/{id}.*")  # look for jpg/jpeg/png files
        jsn_file = f"{self.jsn_dir}/{id}.json"
        
        # Delete the matching files
        for img_file in img_files:
          if os.path.exists(img_file):
            os.remove(img_file)
        if os.path.exists(jsn_file):
          os.remove(jsn_file)
      print("Deletion successful!")
    except Exception as e:
      print(f"An error occurred while deleting: {e}")

  def reorder(self):
    '''
    reorder the files in the directories to be in order, i.e. 1.jpg, 2.jpg, 3.jpg, etc.
    '''
    if self.img_id == []:
      print("No images to reorder!")
      return
    try:
      # Identify all remaining ids in the directories
      remaining_img_ids = sorted(int(os.path.splitext(filename)[0]) for filename in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, filename)))
      remaining_jsn_ids = sorted(int(os.path.splitext(filename)[0]) for filename in os.listdir(self.jsn_dir) if os.path.isfile(os.path.join(self.jsn_dir, filename)))
      remaining_ids = sorted(set(remaining_img_ids) & set(remaining_jsn_ids))  # intersection of the two lists
      
      # Create mappings of old ids to new ids
      id_mapping = {old: new for new, old in enumerate(remaining_ids, start=1)}
      
      # Rename the files based on the new ids
      for old_id, new_id in id_mapping.items():
        old_img_files = glob.glob(f"{self.img_dir}/{old_id}.*")  # look for jpg/jpeg/png files
        old_jsn_file = f"{self.jsn_dir}/{old_id}.json"
        new_jsn_file = f"{self.jsn_dir}/{new_id}.json"
        
        for old_img_file in old_img_files:
          file_ext = os.path.splitext(old_img_file)[1]  # get the file extension
          new_img_file = f"{self.img_dir}/{new_id}{file_ext}"
          os.rename(old_img_file, new_img_file)  # rename the image file
        if os.path.exists(old_jsn_file):
          os.rename(old_jsn_file, new_jsn_file)  # rename the json file
      print("Reordering successful!")
    except Exception as e:
      print(f"An error occurred while reordering: {e}")

def visualizeAll(img_dir, jsn_dir, output_dir=None):
  """
  Visualize all images and corresponding annotations in the provided directories.

  img_dir: Path to directory containing image files.
  jsn_dir: Path to directory containing JSON files with annotations.
  output_dir: Path to directory to save visualized images. If None, images are displayed and not saved.
  """
  # Sort the files in both directories
  img_files = sorted(glob.glob(img_dir + '/*.*'))
  json_files = sorted(glob.glob(jsn_dir + '/*.json'))

  for idx, (img_file, json_file) in enumerate(zip(img_files, json_files)):
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

    with open(json_file, 'r') as f:
      coco_data = json.load(f)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    for data_dict in coco_data:
      annotations = data_dict['annotation']['annotationGroups'][0]['annotationEntities']
      
      for annotation in annotations:
        segmentations = annotation['annotationBlocks'][0]['segments']
        name = annotation['name']

        if isinstance(segmentations, list):
          polygons = [np.array(segmentation).reshape(-1, 2) for segmentation in segmentations]
        else:
          polygons = [np.array(segmentations).reshape(-1, 2)]

        if name == 'water':
          facecolor = 'red'
          alpha = 0.4
        else:
          facecolor = 'none'
          alpha = 1.0

        for polygon in polygons:
          patch = Polygon(polygon, edgecolor='blue', facecolor=facecolor, alpha=alpha)
          ax.add_patch(patch)
        
    ax.axis('off')

    if output_dir:
      plt.savefig(f"{output_dir}/{idx + 1}.jpg")
      plt.close(fig)
    else:
      plt.show()

def visualizeAllBinary(img_dir, jsn_dir, output_dir=None):
  """
  Visualize all images and corresponding binary masks in the provided directories.

  img_dir: Path to directory containing image files.
  jsn_dir: Path to directory containing JSON files with binary masks.
  output_dir: Path to directory to save visualized images. If None, images are displayed and not saved.
  """
  # Sort the files in both directories
  img_files = sorted(glob.glob(img_dir + '/*.*'))
  json_files = sorted(glob.glob(jsn_dir + '/*.json'))
  cmap = ListedColormap(['none', 'red'])

  for idx, (img_file, json_file) in enumerate(zip(img_files, json_files)):
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

    with open(json_file, 'r') as f:
      mask_data = json.load(f)
    
    # Convert the mask data into a numpy array
    binary_mask = np.array(mask_data["binary_mask"], dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.imshow(binary_mask, alpha=0.5, cmap=cmap)
    
    ax.axis('off')

    if output_dir:
      plt.savefig(f"{output_dir}/{idx + 1}.jpg")
      plt.close(fig)
    else:
      plt.show()

def concatAll(img_dirs, jsn_dirs, output_dir, delete=False):
  try:
    img_output_dir = os.path.join(output_dir, "img")
    jsn_output_dir = os.path.join(output_dir, "jsn")

    # Create output directories if they don't exist
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(jsn_output_dir, exist_ok=True)

    current_max_id = 0  # keep track of the current maximum id to prevent naming conflicts

    for img_dir, jsn_dir in zip(img_dirs, jsn_dirs):
      img_files = sorted(glob.glob(f"{img_dir}/*.*"), key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))  # sort by id
      jsn_files = sorted(glob.glob(f"{jsn_dir}/*.json"), key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))  # sort by id

      for img_file, jsn_file in zip(img_files, jsn_files):
        new_id = current_max_id + 1
        file_ext = os.path.splitext(img_file)[1]  # get the file extension

        new_img_file = f"{img_output_dir}/{new_id}{file_ext}"
        new_jsn_file = f"{jsn_output_dir}/{new_id}.json"

        shutil.copy(img_file, new_img_file)
        shutil.copy(jsn_file, new_jsn_file)

        current_max_id = new_id  # update the current maximum id

    if delete:
      for dir in img_dirs + jsn_dirs:
        shutil.rmtree(dir)

  except Exception as e:
    print(f"An error occurred: {e}")

def fixBottom(binary_mask, t):
  binary_mask_np = np.array(binary_mask)

  # Loop through rows starting from t until 1
  for i in range(t, 0, -1):
    target_row = binary_mask_np[-i]
    
    for column_index, value in enumerate(target_row):
      if value == 1:
        for j in range(i - 1, 0, -1):  # Fill all subsequent rows
          binary_mask_np[-j, column_index] = 1
          
  return binary_mask_np.tolist()

def modifyBottom(directory_path, t):
  for file_name in os.listdir(directory_path):
    if file_name.endswith('.json'):
      file_path = os.path.join(directory_path, file_name)
      
      with open(file_path, 'r') as infile:
        data = json.load(infile)
        
      # Apply the heuristic to the binary mask
      data['binary_mask'] = fixBottom(data['binary_mask'], t)
      
      with open(file_path, 'w') as outfile:
        json.dump(data, outfile)

