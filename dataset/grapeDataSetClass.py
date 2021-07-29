import glob
import os
import cv2
import numpy as np
import json
import torch

class GrapeDataset():
  def __init__(self, data_root, mode):

    self.mask_npz = sorted(glob.glob(os.path.join(data_root, "*.npz")))
    images_jpg = sorted(glob.glob(os.path.join(data_root, "*.jpg")))
    bbox_txt = sorted(glob.glob(os.path.join(data_root, "*.txt")))
    assert len(images_jpg) == len(bbox_txt)

    # In wgisd dataset not all masks are paired with bbox and imgs.
    image_names = [os.path.splitext(os.path.basename(fp))[0] for fp in images_jpg]
    mask_names = [os.path.splitext(os.path.basename(fp))[0] for fp in self.mask_npz]
    removable_ind = [ii for ii, n in enumerate(image_names) if n not in mask_names]

    self.images_jpg, self.bbox_txt = list(), list()
    for ii in range(len(images_jpg)):
      if ii not in removable_ind:
        self.images_jpg.append(images_jpg[ii])
        self.bbox_txt.append(bbox_txt[ii])

    if mode == "train":
      print("Dataset in training mode")
      self.mask_npz = self.mask_npz[:100]
      self.images_jpg = self.images_jpg[:100]
      self.bbox_txt = self.bbox_txt[:100]
    elif mode == "valid":
      print("Dataset in evaluation mode")
      self.mask_npz = self.mask_npz[:34]
      self.images_jpg = self.images_jpg[:34]
      self.bbox_txt = self.bbox_txt[:34]
    else:
      raise ValueError("mode is an invalid value")

    # Checks on dataset
    assert len(self.images_jpg) == len(self.mask_npz) == len(self.bbox_txt)
    for ii in range(len(self.images_jpg)):
      assert os.path.splitext(os.path.basename(self.images_jpg[ii]))[0] == \
      os.path.splitext(os.path.basename(self.bbox_txt[ii]))[0] == \
      os.path.splitext(os.path.basename(self.mask_npz[ii]))[0]

    print("Dataset Passed Assertions")


  def __getitem__(self, idx):

    # Handle the img
    img = cv2.imread(self.images_jpg[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = F.to_tensor(img)

    # Handle the mask
    masks = np.load(self.mask_npz[idx])["arr_0"].astype(np.uint8)
    masks = np.moveaxis(masks, -1, 0)
    num_objs = masks.shape[0]
    masks = torch.as_tensor(masks, dtype=torch.uint8)

    # Handle bboxes
    np_bbox_text = np.loadtxt(self.bbox_txt[idx], delimiter = " ", dtype = np.float32)
    bboxes = np_bbox_text[:, 1:]

    assert (bboxes.shape[0] == num_objs)

    _, height, width = img.shape
    scaled_boxes, areas = [], []
    for box in bboxes:
        x1 = box[0] - box[2]/2
        x2 = box[0] + box[2]/2
        y1 = box[1] - box[3]/2
        y2 = box[1] + box[3]/2
        scaled_boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
        areas.append((x2 * width - x1 * width) * (y2 * height - y1 * height))


    scaled_boxes = torch.as_tensor(scaled_boxes, dtype=torch.float32)
    areas = torch.as_tensor(areas, dtype = torch.float32)

    #Create labels from masks
    labels = torch.ones((num_objs,), dtype=torch.int64)
    image_id = torch.tensor([idx])
    iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

    target = {
            "boxes": scaled_boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd
        }

    return img, target

  def __len__(self):
    return len(self.images_jpg)
    
