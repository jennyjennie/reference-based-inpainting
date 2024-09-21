import os
import numpy as np
import json
import shutil

# Path to the samples directory
dataset_folder = '../data/train_200'
samples_dir_path = "Apr_12/photocon"


def orb_to_blender(orb_t):
    # blender start with camera looking down, -z forward
    # orb starts with camera looking forward +z forward
    pre_conversion = np.array([ # orb starts with +z forward, +y down
        [1,0,0,0],
        [0,-1,0,0],
        [0,0,-1,0],
        [0,0,0,1],
    ])
    conversion = np.array([ # converts +y down world to z+ up world
        [1,0,0,0],
        [0,0,1,0],
        [0,-1,0,0],
        [0,0,0,1],
    ])
    camera_local = np.linalg.inv(orb_t)
    orb_world = np.matmul(camera_local,pre_conversion)
    blender_world = np.matmul(conversion,orb_world)

    return blender_world


def extract_poses(seq_fn):
    with open(seq_fn) as f:
        url = f.readline()
        video_code = url[url.index('=')+1:-1]

        # read all lines
        lines = []
        for line in f:
            lines.append(line)
        n_poses = len(lines)

        # reject sequences with less than 2 poses
        if n_poses < 2:
            print('too view frames, skipping')
            return

        out_dict = {
            'video_code': video_code,
            'timestamp':np.zeros(shape=[n_poses],dtype=np.int32),
            'focal_x':np.zeros(shape=[n_poses],dtype=np.float32),
            'focal_y':np.zeros(shape=[n_poses],dtype=np.float32),
            'princ_x':np.zeros(shape=[n_poses],dtype=np.float32),
            'princ_y':np.zeros(shape=[n_poses],dtype=np.float32),
            'pose':np.zeros(shape=[n_poses,3,4],dtype=np.float32),
        }

        # extract poses
        for n,line in enumerate(lines):
            fields = line.split(' ')
            timestamp_micro = int(fields[0])
            intrin = fields[1:5]
            extrin = fields[7:]
            extrin = [float(x) for x in extrin]+[0,0,0,1]
            extrin = np.array(extrin).reshape(4,4)
            extrin = orb_to_blender(extrin)
            out_dict['timestamp'][n] = timestamp_micro
            out_dict['focal_x'][n] = intrin[0]
            out_dict['focal_y'][n] = intrin[1]
            out_dict['princ_x'][n] = intrin[2]
            out_dict['princ_y'][n] = intrin[3]
            out_dict['pose'][n] = extrin[:-1,:]

        return out_dict

def select_frame(out_dict, selected_ids):
    formatted_data = {"focal_y": None, "poses": [], "dependencies": [None], "generation_order": []}
    formatted_data["focal_y"] = out_dict['focal_y'][0].astype(float)  # focal_length_y

    for idx in selected_ids:
        pose_matrix = out_dict['pose'][idx-1]
        pose_matrix = np.vstack([pose_matrix, [0, 0, 0, 1]])
        formatted_data["poses"].append(pose_matrix.astype(float).tolist())

    # Modify dependencies and generation_order as per your logic
    for i in range(len(selected_ids)-1):
        formatted_data["dependencies"].append([i])
        formatted_data["generation_order"].append(i+1)  # Assuming index starts from 1

    return formatted_data


# Loop through each sample_id directory
max_frame_id, stride = 151, 30
for sample_id in os.listdir(samples_dir_path):
    if sample_id == '.DS_Store':
        continue
    txt_file_path = os.path.join(dataset_folder, f'{sample_id}.txt')
    print(txt_file_path)
    out_dict = extract_poses(txt_file_path)
    extracted_frames_id = [i for i in range(1, max_frame_id, stride)]
    selected_posed = select_frame(out_dict, extracted_frames_id)

    # Write the formatted data to a JSON file
    output_file = f'{samples_dir_path}/{sample_id}/sampling-spec.json'
    with open(output_file, 'w') as json_file:
        json.dump(selected_posed, json_file, indent=4)
    print(f"Data written to {output_file}")

print("Finished processing all sample_ids.")
