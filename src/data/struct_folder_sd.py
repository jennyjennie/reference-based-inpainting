"""
This script is for structuring the folder to match the folder structure of 
of stable diffusion modelss
"""

import os
from PIL import Image, ImageDraw, ImageOps
import shutil

IMAGE_SIZE = (1280, 720)
REF_FILENAME = '1.png'
TAR_FILENAME = '121.png'

def get_scene_ids(path):
    scene_ids = []
    for scene_id in os.listdir(path):
        if scene_id == '.DS_Store':
            continue
        scene_ids.append(scene_id)

    return scene_ids

def apply_mask_to_image(image_path, mask_path):
    mask = Image.open(mask_path).convert("L")
    inverted_mask = ImageOps.invert(mask)  # Invert the mask
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    masked_image = Image.composite(image, Image.new("RGB", image.size, 'white'), inverted_mask)

    return masked_image


def get_scenes_ref_target_mask(src_dataset_dir, dst_dataset_dir, mask_dir, scene_ids):
    for i, scene_id in enumerate(scene_ids):
        src_path = os.path.join(src_dataset_dir, scene_id)
        dst_path = os.path.join(dst_dataset_dir, scene_id)

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        # Copy images and mask from source to target
        shutil.copy(os.path.join(src_path, REF_FILENAME), \
                    os.path.join(dst_path, REF_FILENAME))
        shutil.copy(os.path.join(src_path, TAR_FILENAME), \
                    os.path.join(dst_path, TAR_FILENAME))
        shutil.copy(os.path.join(mask_dir, f'mask_{i}.png'), \
                    os.path.join(dst_path, f'mask.png'))

        # Generate masked target image given image and mask paths
        target_path = os.path.join(dst_path, TAR_FILENAME)
        mask_path = os.path.join(dst_path, f'mask.png')
        masked_image_path = os.path.join(dst_path, 'masked_121.png')
        masked_image = apply_mask_to_image(target_path, mask_path)
        masked_image.save(masked_image_path)





photocon_path = 'Apr_12/photocon'
src_dataset_dir = '../out/train_200'
dst_dataset_dir = 'Apr_12/stable_diffusion'
mask_dir = 'Apr_12/masks'
scene_ids = get_scene_ids(photocon_path)
print(scene_ids)
get_scenes_ref_target_mask(src_dataset_dir, dst_dataset_dir, mask_dir, scene_ids)