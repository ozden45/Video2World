# --------------------------------------------------------------
#   sfm_point_generator.py
#
#   Description:
#       This script generates 3D points from video data using
#       Structure from Motion (SfM).
#
#   Author: Özden Özel
#   Created: 2026-01-28
#
# --------------------------------------------------------------

import os
import cv2 as cv
import torch
import numpy as np
from ..geometry.points import SFMPoints, ImagePoints
from ..geometry.projection import project_sfm_to_img, project_img_to_sfm


def generate_sfm_points(video_path: str, cam_traj_path: str, output_path: str):
    """
    Generates 3D points from video data using Structure from Motion (SfM).

    Args:
        video_path (str): Path to the input video file.
        cam_traj_path (str): Path to the camera trajectory file.
        output_path (str): Path to save the generated 3D points.
    """

    # Check is the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check if the camera trajectory file exists
    if not os.path.exists(cam_traj_path):
        raise FileNotFoundError(f"Camera trajectory file not found: {cam_traj_path}")

    # Read the video file
    cap = cv.VideoCapture(video_path)


def frame_to_sfm_points(frame: torch.Tensor, depth: torch.Tensor, W: torch.Tensor, cam_params: dict) -> SFMPoints:
    """
    Converts a video frame to 3D points using SfM techniques.

    Args:
        frame (torch.Tensor): Input video frame.
        depth (torch.Tensor): Depth map corresponding to the frame.
        R (torch.Tensor): Rotation matrix of the camera.
        t (torch.Tensor): Translation matrix of the camera.
        cam_specs (dict): Camera specifications including intrinsic parameters.

    Returns:
        SFMPoints: The frame index and its generated 3D points in the format [μ, color].
    """

    # The intrinsic parameters of the camera
    fx = cam_params['fx']
    fy = cam_params['fy']
    cx = cam_params['cx']
    cy = cam_params['cy']

    W = W.unsqueeze(0).repeat(N, 1, 1)
    R = W[:, :3, :3]
    t = W[:, :3, 3:].reshape(N, 3, 1)

    R_inv = torch.inverse(R)

    sfm_points = SFMPoints()
    for y in range(0, frame.shape[0]):
        for x in range(0, frame.shape[1]):
            if y % 100 == 0 and x % 100 == 0:
                print(f"Processing pixel ({x}, {y})")

            # Calculate the camera coordinates
            Zc = depth[y, x]
            if Zc == 0:
                print(f"Invalid depth at pixel ({x}, {y}), skipping.")
                continue 
            Xc = (x - cx) * Zc / fx
            Yc = (y - cy) * Zc / fy

            # Form the camera coordinates
            cam_coords = torch.tensor([[Xc], [Yc], [Zc]])
            color = frame[y, x] / 255

            # Convert to world coordinates
            world_coords = R_inv @ (cam_coords - t)

            # Create a sfM point
            sfm_point = SFMPoint(
                coords = world_coords,
                covariance = torch.rand((3, 3)),
                color = color,
                alpha = torch.rand(0, 1)
                )
            
            # Add to the collection
            sfm_points +=  sfm_point


    sfm_pts = SFMPoints()
    Zc = depth[y, x]
    Xc = (x - cx) * Zc / fx
    Yc = (y - cy) * Zc / fy
    
    
def frame_to_sfm_pts(frame: torch.Tensor, depth: torch.Tensor, W: torch.Tensor, K: torch.Tensor) -> SFMPoints:
    """
    Converts a video frame to 3D points using SfM techniques.

    Args:
        frame (torch.Tensor): Input video frame.
        depth (torch.Tensor): Depth map corresponding to the frame.
        R (torch.Tensor): Rotation matrix of the camera.
        t (torch.Tensor): Translation matrix of the camera.
        cam_specs (dict): Camera specifications including intrinsic parameters.

    Returns:
        SFMPoints: The frame index and its generated 3D points in the format [μ, color].
    """
    
    img_pts = ImagePoints(frame, depth)
    sfm_pts = project_img_to_sfm(img_pts, W, K)
    
    return sfm_pts
    
    


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    frame_path = os.path.join(script_dir, 'dog.jpg')
    depth_path = os.path.join(script_dir, 'depth.npy')

    frame = cv.imread(frame_path)  # returns None if file unreadable
    if frame is None:
        raise FileNotFoundError(f"Image not found or unreadable: {frame_path}")
    frame = torch.Tensor(frame)

    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    
    depth = np.load(depth_path)
    depth = torch.Tensor(depth)

    print(f"Frame shape: {frame.shape}, Depth shape: {depth.shape}")
    print(f"Frame type: {type(frame)}, Depth type: {type(depth)}")

    R = torch.Tensor([
        [1, 2, 4],
        [0, 4, 2],
        [0, 0, 2]
    ]).unsqueeze(0)
    t = torch.Tensor([[1], [2], [1]])


    f_x = (f_mm * width_px) / sensor_width_mm
    f_y = (f_mm * height_px) / sensor_height_mm

    f_mm = 2e-3
    width_px = frame.shape[0]
    height_px = frame.shape[1]
    
    sensor_width_mm = 6.784
    sensor_height_mm = 5.427
    
    W = torch.rand((1,3,4))
    cam_params = {
        'fx': int((f_mm * width_px) / sensor_width_mm),
        'fy': int((f_mm * height_px) / sensor_height_mm),
        'cx': frame.shape[1] * .5,
        'cy': frame.shape[0] * .5
    }



    sfm_points = frame_to_sfm_points(frame, depth, R, t, cam_params)
    sfm_point_volume = SFMPointVolume(x_max=10, y_max=10, z_max=10, dx=1, dy=1, dz=1)
    sfm_point_volume.append(sfm_points)


    sfm_point_volume.show()




    

