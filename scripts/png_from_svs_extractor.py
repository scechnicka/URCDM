import argparse
import random
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from collections import Counter
import slideio
from skimage import color

# Constants
MAG_LEVEL_SIZES = [40000, 6500, 1024]  # Corresponding sizes for magnification levels
FILL_COLOR = (242, 243, 242)  # Fill color for empty areas

# PatientDataset Class
class PatientDataset:
    def __init__(self, svs_dir, magnification_level, patch_size=1024):
        self.patch_size = patch_size
        self.magnification_level = magnification_level
        self.svs_dir = svs_dir
        self.train_patch_positions = []
        self.train_slide_ids = []
        self.slide_name_to_index = {}
        self.num_train_patches = 0

        # Process .svs files
        svs_files = self.find_files_with_extension(svs_dir, '.svs')
        for index, slide_id in enumerate(tqdm(svs_files, desc="Processing slides")):
            slide = slideio.open_slide(slide_id, "SVS")
            image = slide.get_scene(0)

            # Resize the image to blocks of the patch size
            small_img = image.read_block(image.rect, size=(image.size[0] // self.patch_size, image.size[1] // self.patch_size))

            # Mask out the background
            img_hs = color.rgb2hsv(small_img)
            img_hs = np.logical_and(img_hs[:, :, 0] > 0.5, img_hs[:, :, 1] > 0.02)

            # Get the positions of the patches that are not background
            patch_positions = np.argwhere(img_hs)

            # Scale the positions to the original image size
            patch_positions = patch_positions * self.patch_size
                
            self.train_slide_ids.append(slide_id)
            self.train_patch_positions.append(patch_positions)
            self.num_train_patches += len(patch_positions)

    def find_files_with_extension(self, directory, extension):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
        return file_paths

    def __len__(self):
        if self.magnification_level == 0:
            return len(self.train_slide_ids)
        else:
            return self.num_train_patches

    def index_to_slide(self, index):
        for i in range(len(self.train_slide_ids)):
            if index < len(self.train_patch_positions[i]):
                patch_position = self.train_patch_positions[i][index]
                return i, (patch_position[1], patch_position[0])
            else:
                index -= len(self.train_patch_positions[i])

    def read_block(self, slide_index, mag_level, x, y, slide=None):
        slide = slide or slideio.open_slide(self.train_slide_ids[slide_index], "SVS").get_scene(0)
        width, height = slide.size
        image_size = MAG_LEVEL_SIZES[mag_level]

        patch = np.full((self.patch_size, self.patch_size, 3), FILL_COLOR)

        # Cropping calculations
        cropped_x = max(x, 0)
        cropped_y = max(y, 0)
        x_trim = max(-x, 0)
        y_trim = max(-y, 0)

        cropped_width = min(width - cropped_x, image_size - x_trim)
        cropped_height = min(height - cropped_y, image_size - y_trim)

        patch_width = int(cropped_width * (self.patch_size / image_size))
        patch_height = int(cropped_height * (self.patch_size / image_size))

        cropped_patch = slide.read_block((cropped_x, cropped_y, cropped_width, cropped_height), size=(patch_width, patch_height))

        patch_x = cropped_x - x
        patch_y = cropped_y - y
        patch_x = int(patch_x * (self.patch_size / image_size))
        patch_y = int(patch_y * (self.patch_size / image_size))

        patch[patch_y:patch_y+patch_height, patch_x:patch_x+patch_width] = cropped_patch

        return torch.from_numpy(patch / 255).permute((2, 0, 1)).float()

    def __getitem__(self, index):
        if self.magnification_level == 0:
            slide_index = index
            patch = self.read_block(slide_index, 0, MAG_LEVEL_SIZES[0] // 2, MAG_LEVEL_SIZES[0] // 2)
        else:
            slide_index, patch_position = self.index_to_slide(index)
            patch = self.read_block(slide_index, self.magnification_level, patch_position[0], patch_position[1])
        return patch


# Main function to handle arguments
def main(args):
    dataset = PatientDataset(args.svs_dir, args.magnification_level)
    os.makedirs(args.output_dir, exist_ok=True)

    num_samples = min(args.num_patches, len(dataset))
    selected_indexes = random.sample(range(len(dataset)), num_samples)

    for i in tqdm(selected_indexes, desc="Saving patches"):
        numpy_array = dataset[i].cpu().numpy()
        image_array = np.transpose(numpy_array, (1, 2, 0))
        image = Image.fromarray((image_array * 255).astype(np.uint8))
        image.save(os.path.join(args.output_dir, f"{i}.png"))

    print(f"Saved {num_samples} patches to {args.output_dir}")

# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess .svs files to extract patches.")
    parser.add_argument('--svs_dir', type=str, required=True, help="Directory containing .svs files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save extracted patches.")
    parser.add_argument('--num_patches', type=int, default=10000, help="Number of patches to extract.")
    parser.add_argument('--magnification_level', type=int, choices=[0, 1, 2], required=True, help="Magnification level (0, 1, 2).")

    args = parser.parse_args()
    main(args)
