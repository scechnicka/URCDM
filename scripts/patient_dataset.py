from collections import Counter

import time
import h5py
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F
import slideio
from tqdm import tqdm
from skimage import color
import numpy as np

import glob 
import os

NUM_FLIPS_ROTATIONS = 8
NUM_TRANSLATIONS =  4
MAG_LEVEL_SIZES = [40000, 6500, 1024]
FILL_COLOR = (242, 243, 242)


class PatientDataset(Dataset):
    def __init__(self, svs_dir, magnification_level, verbose=False, center_cond=False, annotated_dataset=False):
        super().__init__()

        self.patch_size = 1024
        self.center_cond = center_cond
        self.magnification_level = magnification_level


        if verbose:
            print(f"Test slide names: {test_slides}")
            print(f"{len(self.train_h5_ids)} annotated patches in train set.")
            print(f"{len(self.test_h5_ids)} annotated patches in test set.")
        
        
        self.train_patch_positions = []
        self.train_slide_ids = []

        self.svs_dir = svs_dir
        self.num_train_patches = 0
        self.num_test_patches = 0
        self.slide_name_to_index = {}


        svs_files = self.find_files_with_extension(svs_dir, '.svs')
        for index, slide_id in enumerate(tqdm(svs_files, desc="Processing slides") if verbose else svs_files):
            slide = slideio.open_slide(slide_id, "SVS")
            metadata = slide.raw_metadata.split("|")
            for prop in metadata:
                if prop.startswith("Filename = "):
                    slide_name = prop.replace("Filename = ", "").split(" ")[0]
                    self.slide_name_to_index[slide_name] = index

            image = slide.get_scene(0)

            # Resize the image to blocks of the patch size
            small_img = image.read_block(image.rect,
                                         size=(image.size[0] // self.patch_size, image.size[1] // self.patch_size))

            # Mask out the background
            img_hs = color.rgb2hsv(small_img)
            #img_hs = np.logical_and(img_hs[:, :, 0] > 0.5, img_hs[:, :, 1] > 0.02)
            # Adjust these thresholds based on visual inspection of your images
            hue_threshold = 0.5  # Lower if cold white is being included
            saturation_threshold = 0.1 # Increase if impurities are present
            value_threshold = 0.9  # Decrease if tissues are darker

            #img_hs = np.logical_and(img_hs[:, :, 0] > 0.5, img_hs[:, :, 1] > 0.02)
            img_hs = np.logical_and.reduce((img_hs[:, :, 0] > hue_threshold, img_hs[:, :, 1] > saturation_threshold, img_hs[:, :, 2] < value_threshold))

            # Get the positions of the patches that are not background
            patch_positions = np.argwhere(img_hs)

            # Scale the positions to the original image size
            patch_positions = patch_positions * self.patch_size
                
            
            self.train_slide_ids.append(slide_id)
            self.train_patch_positions.append(patch_positions)
            self.num_train_patches += len(patch_positions)

        if verbose:
            print(f"{self.num_test_patches} patches in unannotated test set.")
            print(f"{self.num_train_patches} patches in unannotated train set.")
            print(f"Test slide ids: {self.test_slide_ids}")
            print(self.slide_name_to_index)


    def __len__(self):
        if self.magnification_level == 0:
            return NUM_FLIPS_ROTATIONS * len(self.train_slide_ids)
        else:
            return NUM_FLIPS_ROTATIONS * NUM_TRANSLATIONS * self.num_train_patches


    def index_to_slide(self, index):
        for i in range(len(self.train_slide_ids)):
            if index < len(self.train_patch_positions[i]):
                patch_position = self.train_patch_positions[i][index]
                return i, (patch_position[1], patch_position[0])
            else:
                index -= len(self.train_patch_positions[i])

    def find_files_with_extension(self, directory, extension):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
        return file_paths


    def read_block_mag_zero(self, index):
        slide_index = index // NUM_FLIPS_ROTATIONS
        slide = slideio.open_slide(self.train_slide_ids[slide_index], "SVS").get_scene(0)
        width, height = slide.size

        center_x = width // 2
        center_y = height // 2
        zoomed_size = MAG_LEVEL_SIZES[self.magnification_level]
        x = center_x - zoomed_size // 2
        y = center_y - zoomed_size // 2

        return self.read_block(index // NUM_FLIPS_ROTATIONS, 0, x, y, slide=slide)

    # x y is the coordinate of the top-left corner of the patch to read in the overall image
    # mag_level controls the magnification of the patch
    def read_block(self, slide_index, mag_level, x, y, slide=None):
        if slide == None:
            slide = slideio.open_slide(self.train_slide_ids[slide_index], "SVS").get_scene(0)

        width, height = slide.size

        image_size = MAG_LEVEL_SIZES[mag_level]

        patch = np.full((self.patch_size, self.patch_size, 3), FILL_COLOR)

        # if coords are negative, cap to 0
        cropped_x = max(x, 0)
        cropped_y = max(y, 0)

        # if coords are negative, then the section that is out of bounds
        # should count towards the image_size so we should trim this off
        x_trim = max(-x, 0)
        y_trim = max(-y, 0)

        cropped_width = min(width - cropped_x, image_size - x_trim)
        cropped_height = min(height - cropped_y, image_size - y_trim)

        patch_width = int(cropped_width * (self.patch_size / image_size))
        patch_height = int(cropped_height * (self.patch_size / image_size))

        cropped_patch = slide.read_block((cropped_x, cropped_y, cropped_width, cropped_height), size=(patch_width, patch_height))

        # x and y are relative to the actual kidney image, and we need coordinates
        # relative to the patch we are returning. x and y define the top-left corner
        # of the patch, which is coordinate [0,0] so by subtracting x and y from a set
        # of coordinates, it now is relative to the patch. So we subtract x and y from
        # cropped_x and cropped_y to get the right coordinates.

        patch_x = cropped_x - x
        patch_y = cropped_y - y

        # need to multiply by (self.patch_size / image_size) to change coordinates into
        # the same magnification as the patch, rather than the whole slide.
        patch_x = int(patch_x * (self.patch_size / image_size))
        patch_y = int(patch_y * (self.patch_size / image_size))

        patch[patch_y:patch_y+patch_height, patch_x:patch_x+patch_width] = cropped_patch

        # Convert the patch to a tensor
        patch = torch.from_numpy(patch / 255).permute((2, 0, 1)).float()

        return patch


    def read_block_and_zoomed(self, index):
        slide_index, patch_position = self.index_to_slide(index // (NUM_FLIPS_ROTATIONS * NUM_TRANSLATIONS))

        translation_index = index // NUM_FLIPS_ROTATIONS
        if translation_index % NUM_TRANSLATIONS == 0:
            x, y = (patch_position[0], patch_position[1])
        elif translation_index % NUM_TRANSLATIONS == 1:
            x, y = (patch_position[0] + self.patch_size // 2, patch_position[1])
        elif translation_index % NUM_TRANSLATIONS == 2:
            x, y = (patch_position[0] + self.patch_size // 2, patch_position[1] + self.patch_size // 2)
        else:
            x, y = (patch_position[0], patch_position[1] + self.patch_size // 2)


        image_size = MAG_LEVEL_SIZES[self.magnification_level]
        center_x = x + image_size // 2
        center_y = y + image_size // 2
        zoomed_size = MAG_LEVEL_SIZES[self.magnification_level - 1]
        zoomed_x = center_x - zoomed_size // 2
        zoomed_y = center_y - zoomed_size // 2

        patch = self.read_block(slide_index, self.magnification_level, x, y) 
        zoomed_patch = self.read_block(slide_index, self.magnification_level - 1, zoomed_x, zoomed_y)

        return patch, zoomed_patch


    def flip_rotate_patch(self, index, patch):
        if index % NUM_FLIPS_ROTATIONS == 0:
            return patch
        elif index % NUM_FLIPS_ROTATIONS == 1:
            return patch.flip(2)
        elif index % NUM_FLIPS_ROTATIONS == 2:
            return patch.flip(1)
        elif index % NUM_FLIPS_ROTATIONS == 3:
            return patch.flip(1).flip(2)
        elif index % NUM_FLIPS_ROTATIONS == 4:
            return patch.transpose(1, 2)
        elif index % NUM_FLIPS_ROTATIONS == 5:
            return patch.transpose(1, 2).flip(2)
        elif index % NUM_FLIPS_ROTATIONS == 6:
            return patch.transpose(1, 2).flip(1)
        else:
            return patch.transpose(1, 2).flip(1).flip(2)


    def __getitem__(self, index):
        # size of higher mag patch within the zoomed_patch (once a center crop is made)
        patch_width = int(MAG_LEVEL_SIZES[self.magnification_level] * self.patch_size / MAG_LEVEL_SIZES[self.magnification_level - 1])
        
        if self.magnification_level > 0:
            patch, zoomed_patch = self.read_block_and_zoomed(index)
            patch = self.flip_rotate_patch(index, patch)
            zoomed_patch = self.flip_rotate_patch(index, zoomed_patch)
            if self.center_cond:
                center_patch = T.CenterCrop(patch_width)(zoomed_patch)
                center_patch = F.interpolate(center_patch.unsqueeze(0), zoomed_patch.shape[-1], mode='nearest').squeeze(0)
                cond_image = torch.cat((zoomed_patch, center_patch), 0)
                return patch, cond_image
            else:
                return patch, zoomed_patch
        else:
            patch = self.read_block_mag_zero(index)
            return self.flip_rotate_patch(index, patch)

