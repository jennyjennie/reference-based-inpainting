import os
import json
import shutil

TARGET_IMAGE_FRAME_NUM = 13

def resample_poses(old_file_path, new_file_path):
    with open(old_file_path, 'r') as file:
        data = json.load(file)
    print(data) # DBG
    # Resampling poses, dependencies, and generation_order
    resampled_poses = [data['poses'][i] for i in range(0, TARGET_IMAGE_FRAME_NUM, 3)]
    dependencies = [None] + [[i] for i in range(len(resampled_poses) - 1)]
    generation_order = list(range(1, len(resampled_poses)))
    
    data['focal_y'] = data['focal_y']
    data['poses'] = resampled_poses
    data['dependencies'] = dependencies
    data['generation_order'] = generation_order

    with open(new_file_path, 'w') as file:
        json.dump(data, file, indent=4)

def create_structure(parent_folder, target_folder):
    for subdir, dirs, files in os.walk(parent_folder):
        # Skip the root directory
        if subdir == parent_folder:
            continue
        print(subdir)
        image_files = [file for file in os.listdir(subdir) if file.endswith('.png')]
        
        # Check the number of image files
        if len(image_files) < TARGET_IMAGE_FRAME_NUM+1:
            # Delete the folder if it contains only two images
            print(f"{subdir} insufficient images...")
            continue

        # Extract the ID (subfolder name)
        folder_id = os.path.basename(subdir)

        # Create corresponding directories in the target folder
        target_subfolder = os.path.join(target_folder, folder_id)
        init_ims_folder = os.path.join(target_subfolder, 'init-ims')
        samples_folder = os.path.join(target_subfolder, 'samples')

        os.makedirs(init_ims_folder, exist_ok=True)
        os.makedirs(samples_folder, exist_ok=True)

        # Copy the image and json file
        image_path = os.path.join(subdir, '0000.png')
        if os.path.exists(image_path):
            shutil.copy2(image_path, init_ims_folder)
        
        ''' # Sample sample-spec.json directly from dataset.
        old_json_path = os.path.join(subdir, 'sampling-spec.json')
        new_json_path = os.path.join(target_folder, folder_id, 'sampling-spec.json')
        print(old_json_path, new_json_path) # DBG
        
        # Check if sampling-spec.json exists in the directory
        if os.path.exists(old_json_path):
            print(f"Resampling poses for {subdir}...")
            resample_poses(old_json_path, new_json_path)
            print(f"Finished resampling for {subdir}.")
        '''
        print(f"Processed {folder_id}")

parent_folder = '../out/train_200'  # Replace with the path to your parent folder
target_folder = 'Apr_12/photocon'  # Replace with the path to your target folder

create_structure(parent_folder, target_folder)
