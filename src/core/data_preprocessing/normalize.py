# Adapted code from AmitMY siqn-vq repo

import functools
from pathlib import Path
import io
import zipfile
from tqdm import tqdm
import json
import numpy as np

from pose_format import Pose
from pose_format.utils.generic import pose_normalization_info, correct_wrists, hands_components, reduce_holistic


def shift_hand(pose: Pose, hand_component: str, wrist_name: str):
    # pylint: disable=protected-access
    wrist_index = pose.header._get_point_index(hand_component, wrist_name)
    hand = pose.body.data[:, :, wrist_index: wrist_index + 21]
    wrist = hand[:, :, 0:1]
    pose.body.data[:, :, wrist_index: wrist_index + 21] = hand - wrist


def pre_process_mediapipe(pose: Pose):
    # Remove legs, simplify face
    pose = reduce_holistic(pose)
    pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])

    # Align hand wrists with body wrists
    correct_wrists(pose)
    
    # Adjust pose based on shoulder positions
    pose = pose.normalize(pose_normalization_info(pose.header))

    # Shift hands to origin
    (left_hand_component, right_hand_component), _, (wrist, _) = hands_components(pose.header)
    shift_hand(pose, left_hand_component, wrist)
    shift_hand(pose, right_hand_component, wrist)

    return pose


def get_mean_and_std(directory: str):
    cumulative_sum, squared_sum, frames_count = None, None, None

    for file in tqdm(list(Path(directory).glob("*.pose"))):
        # Get the pose
        with open(file, 'rb') as pose_file:
            pose = Pose.read(pose_file.read())
            pose = pre_process_mediapipe(pose)
        tensor = pose.body.data.filled(0)

        # Get relevant values
        frames_sum = np.sum(tensor, axis=(0, 1))
        frames_squared_sum = np.sum(np.square(tensor), axis=(0, 1))
        # pylint: disable=singleton-comparison
        unmasked_frames = pose.body.data[:, :, :, 0:1].mask == False    # what is even mask for?
        num_unmasked_frames = np.sum(unmasked_frames, axis=(0, 1))     

        # Update cumulative values
        cumulative_sum = frames_sum if cumulative_sum is None else cumulative_sum + frames_sum
        squared_sum = frames_squared_sum if squared_sum is None else squared_sum + frames_squared_sum
        frames_count = num_unmasked_frames if frames_count is None else frames_count + num_unmasked_frames

    mean = cumulative_sum / frames_count
    std = np.sqrt((squared_sum / frames_count) - np.square(mean))

    return mean, std


def comp_mean_std(pose_dir: str, pose_norm_dir: str):
    mean, std = get_mean_and_std(pose_dir)

    # get a single random pose
    random_pose_path = Path(pose_dir).glob("*.pose").__next__()
    with open(random_pose_path, 'rb') as pose_file:
        pose = Pose.read(pose_file.read())
        pose = pre_process_mediapipe(pose)

    # store header          WHAT IS HEADER?
    with open(Path(pose_norm_dir) / "header.poseheader", "wb") as f:
        pose.header.write(f)

    i = 0
    mean_std_info = {}
    for component in pose.header.components:
        component_info = {}
        for point in component.points:
            component_info[point] = {
                "mean": mean[i].tolist(),
                "std": std[i].tolist()
            }
            i += 1
        mean_std_info[component.name] = component_info

    with open(Path(pose_norm_dir) / "pose_normalization.json", "w", encoding="utf-8") as f:
        json.dump(mean_std_info, f, indent=2)

# ----------------------------------------------
# normalize data using pre-computed mean and std
# ----------------------------------------------

@functools.lru_cache(maxsize=1)
def load_mean_and_std(pose_norm_dir: Path):
    with open(pose_norm_dir/ "pose_normalization.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    mean, std = [], []
    for component in data.values():
        for point in component.values():
            mean.append(point["mean"])
            std.append(point["std"])

    # when std is 0, set std to 1
    std = np.array(std)
    std[std == 0] = 1

    return np.array(mean), std

def normalize_mean_std(pose: Pose, pose_norm_dir: str):
    mean, std = load_mean_and_std(Path(pose_norm_dir))
    pose.body.data = (pose.body.data - mean) / std
    return pose

def normalize_poses(pose_dir: str, zip_output_file: str):
    pose_files = list(Path(pose_dir).glob("**.pose"))
    with zipfile.ZipFile(zip_output_file, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in tqdm(pose_files):
            with open(file, 'rb') as pose_file:
                pose = Pose.read(pose_file.read())
                pose = pre_process_mediapipe(pose)
                pose = normalize_mean_std(pose)

                npz_filename = file.stem + '.npz'            

                with io.BytesIO() as buf:
                    data = pose.body.data[:,0,:,:]
                    float16_data = data.filled(0).astype(np.float16)
                    np.savez_compressed(buf, data=float16_data, mask=data.mask)
                    zip_file.writestr(npz_filename, buf.getvalue())

