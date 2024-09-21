from __future__ import unicode_literals
import os
import glob
import cv2
import json
import yt_dlp
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw

'''
Download the dataset and extract the tragectory of each scene
'''

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

def download_dataset(txt_dir, out_dir, stride=1, remove_video=True):
    '''
    Download all frames from each video
    '''
    all_files = sorted(glob.glob(os.path.join(txt_dir, '*.txt'))) 
    for i in tqdm(range(len(all_files))):
        f = all_files[i]
        print(f)
        file_name = os.path.basename(f).split('.')[0]
        out_f = os.path.join(out_dir, file_name)
        
        if os.path.exists(out_f): 
            print('the file exists. skip....')
            continue

        with open(f) as video_txt:
            content = video_txt.readlines()
            if len(content) < 101:
                print('File to short, skip...')
                continue
            url = content[0]

        try:
            ydl_opts = {'writesubtitles': True, 'skip-download': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                info = ydl.extract_info(url, download=True)
                output_file = ydl.prepare_filename(info)
        except:
            print("An exception occurred, maybe because of the downloading limits of youtube.")
            continue
        
        # if video is already downloaded, start extracting frames
        os.makedirs(out_f, exist_ok=True)
        if not os.path.exists(output_file): output_file = output_file.replace('.mp4','.mkv','.webm')
        os.rename(output_file, os.path.join(out_f, file_name + '.mp4'))
        line = url
        vidcap = cv2.VideoCapture(os.path.join(out_f, file_name + '.mp4'))
        frame_ind = 1
        for num in range(1, len(content), stride):
            line = content[num]
            if line == '\n': break
            ts = line.split(' ')[0][:-3]  #extract the time stamp
            if ts == '': break
            vidcap.set(cv2.CAP_PROP_POS_MSEC,int(ts))      # just cue to 20 sec. position
            success,image = vidcap.read()
            if success:
                cv2.imwrite(out_f + '/' + str(frame_ind) + '.png', image)     # save frame as JPEG file
                if num == 1: cv2.imwrite(out_f + '/0000.png', cv2.resize(image, (256, 256))) # save and resize reference frame (1st frame)
                frame_ind += stride
        
        if remove_video:
            os.remove(os.path.join(out_f, file_name + '.mp4'))

        out_dict = extract_poses(f)
        extracted_frames_id = [i for i in range(1, frame_ind, args.stride)]
        selected_posed = select_frame(out_dict, extracted_frames_id)

        # Write the formatted data to a JSON file
        output_file = f'{out_f}/sampling-spec.json'
        with open(output_file, 'w') as json_file:
            json.dump(selected_posed, json_file, indent=4)
        print(f"Data written to {output_file}")

def download_specified_samples(txt_dir, out_dir, stride=10, max_stride=51, remove_video=True):
    '''
    Download all frames from each video
    '''
    all_files = [f'{txt_dir}/0a0b550ac417ed7f.txt', f'{txt_dir}/0a1abaa6ea233a54.txt', f'{txt_dir}/0a1c6a4cd53b9fff.txt']
    for i in tqdm(range(len(all_files))):
        f = all_files[i]
        print(f)
        file_name = os.path.basename(f).split('.')[0]
        out_f = os.path.join(out_dir, file_name)
        
        if os.path.exists(out_f): 
            print('the file exists. skip....')
            continue

        with open(f) as video_txt:
            content = video_txt.readlines()
            if len(content) < 50:
                print('File to short, skip...')
                continue
            url = content[0]

        try:
            ydl_opts = {'writesubtitles': True, 'skip-download': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                info = ydl.extract_info(url, download=True)
                output_file = ydl.prepare_filename(info)
        except:
            print("An exception occurred, maybe because of the downloading limits of youtube.")
            continue
        
        # if video is already downloaded, start extracting frames
        os.makedirs(out_f, exist_ok=True)
        if not os.path.exists(output_file): output_file = output_file.replace('.mp4','.mkv','.webm')
        os.rename(output_file, os.path.join(out_f, file_name + '.mp4'))
        line = url
        vidcap = cv2.VideoCapture(os.path.join(out_f, file_name + '.mp4'))
        frame_ind = 1
        for num in range(1, min(max_stride, len(content)), stride):
            line = content[num]
            if line == '\n': break
            ts = line.split(' ')[0][:-3]  #extract the time stamp
            if ts == '': break
            vidcap.set(cv2.CAP_PROP_POS_MSEC,int(ts))      # just cue to 20 sec. position
            success,image = vidcap.read()
            if success:
                cv2.imwrite(out_f + '/' + str(frame_ind) + '.png', image)     # save frame as JPEG file
                if num == 1: cv2.imwrite(out_f + '/0000.png', cv2.resize(image, (256, 256))) # save and resize reference frame (1st frame)
                frame_ind += stride
        
        if remove_video:
            os.remove(os.path.join(out_f, file_name + '.mp4'))

        out_dict = extract_poses(f)
        extracted_frames_id = [i for i in range(1, frame_ind, stride)]
        selected_posed = select_frame(out_dict, extracted_frames_id)

        # Write the formatted data to a JSON file
        output_file = f'{out_f}/sampling-spec.json'
        with open(output_file, 'w') as json_file:
            json.dump(selected_posed, json_file, indent=4)
        print(f"Data written to {output_file}")


if __name__ == "__main__": 
    #using the script to prepare the dataset
    parser = argparse.ArgumentParser(description='Download RealEstate10K Dataset')
    parser.add_argument('--txt_dir', metavar='path', default = './RealEstate10K', required=False,
                        help='path to the original dataset txt files downloaded online')
    parser.add_argument('--out_dir', metavar='path', default = './RealEstate10K_frames', required=False,
                        help='path to store output frames')
    parser.add_argument('--stride', default=50, type=int, required=False,
                        help='the stride between extracted frames')
    args = parser.parse_args()

    #download_dataset(args.txt_dir, args.out_dir, args.stride)
    download_specified_samples(args.txt_dir, args.out_dir, stride=30, max_stride=151)