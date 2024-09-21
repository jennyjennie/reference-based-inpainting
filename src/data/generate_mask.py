import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps

IMAGE_SIZE = (1280, 720)
MASKED_AREA_PERCENTAGE = 0.2


def calculate_coverage(mask):
    return np.sum(mask) / mask.size


def brush_stroke_mask(shapes):
    """
    Generate brush stroke mask
    Returns:
        numpy.ndarray: output with shape [H, W]
    """
    min_num_vertex = 4
    max_num_vertex = 12
    min_width = 80
    max_width = 120
    mean_angle = 2 * np.pi / 5
    angle_range = 2 * np.pi / 15
    W, H = shapes
    average_radius = np.sqrt(H * H + W * W) / 8
    mask = Image.new('L', (W, H), 0)

    for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = [np.random.uniform(angle_min, angle_max) for _ in range(num_vertex)]

        # Random starting point
        vertex = [(np.random.randint(0, W), np.random.randint(0, H))]

        for angle in angles:
            r = np.clip(np.random.normal(average_radius, average_radius // 2), 0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * np.cos(angle), 0, W)
            new_y = np.clip(vertex[-1][1] + r * np.sin(angle), 0, H)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2, v[1] - width // 2, v[0] + width // 2, v[1] + width // 2), fill=1)

    return np.array(mask, dtype=np.float32)

def add_bounding_box(mask, desired_coverage):
    H, W = mask.shape
    current_coverage = np.mean(mask)

    while current_coverage < desired_coverage:
        box_width = np.random.randint(W // 10, W // 2)
        box_height = np.random.randint(H // 10, H // 2)
        x1 = np.random.randint(0, W - box_width)
        y1 = np.random.randint(0, H - box_height)
        x2, y2 = x1 + box_width, y1 + box_height
        mask[y1:y2, x1:x2] = 1
        current_coverage = np.mean(mask)

    return mask

def generate_mask_with_percentage(path, shapes, desired_coverage):
    irregular_mask = brush_stroke_mask(shapes)
    coverage = np.mean(irregular_mask)

    if coverage < desired_coverage:
        mask = add_bounding_box(irregular_mask, desired_coverage)
    else:
        mask = irregular_mask

    print(calculate_coverage(mask))
    im = Image.fromarray(mask * 255)  # Scale mask to 0-255 range
    im = im.convert("L")  # Convert to grayscale
    im.save(path)



def apply_mask_to_largest_id_image(parent_folder, mask_path):
    mask = Image.open(mask_path).convert("L")
    inverted_mask = ImageOps.invert(mask)  # Invert the mask

    for subdir, dirs, files in os.walk(parent_folder):
        png_files = [file for file in files if file.lower().endswith('.png') and not file.lower().startswith('mask')]
        file_ids = [(int(''.join(filter(str.isdigit, file))), file) for file in png_files]

        if not file_ids:
            continue

        largest_id_file = sorted(file_ids, key=lambda x: x[0], reverse=True)[0][1]
        image_path = os.path.join(subdir, largest_id_file)
        image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)

        # Apply the mask
        masked_image = Image.composite(image, Image.new("RGB", image.size, 'white'), inverted_mask)

        # Save the masked image
        masked_image_path = os.path.join(subdir, f"masked_{largest_id_file}")
        masked_image.save(masked_image_path)
        print(f"Mask applied to {image_path} and saved as {masked_image_path}")

for i in range(79):
    generate_mask_with_percentage(f'experiments/Apr_12/masks/mask_{i}.png', IMAGE_SIZE, MASKED_AREA_PERCENTAGE)

#apply_mask_to_largest_id_image('experiments/Jan_26/orbit_scenes', 'experiments/Jan_5/mask_far_30.png')