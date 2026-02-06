#!/usr/bin/env python3
import cv2
import numpy as np
import h5py
import time
import os
from termcolor import cprint
import sys
from diffusion_policy_3d.common.multi_realsense import MultiRealSense
from piper_sdk import *
NUM_POINTS = 10000 
RECORD_FPS_TARGET = 100 
piper = C_PiperInterface_V2("can0")
# Replace these with your actual robot interface functions
def get_action():
    """
    Return current action vector (e.g. delta xyz + rotation + gripper).
    Shape: (7,) float32 or similar.
    """
    text = piper.GetArmJointCtrl().joint_ctrl.angles
    gripper = piper.GetArmGripperCtrl().gripper_ctrl.angle
    text = np.array(list(text) + [gripper])
    print("action:", text)
    return text

def get_env_qpos():
    """
    Return current joint positions / proprioception.
    Shape: (DoF,) float32
    """
    text = piper.GetArmJointMsgs().joint_state.angles
    gripper = piper.GetArmGripperMsgs().gripper_state.angle
    text = np.array(list(text) + [gripper])
    print("env_qpos:", text)
    return text

# ==================== Dataset Collection Class ====================
class DatasetCollector:
    def __init__(self):
        # Start background multi-process camera streaming
        self.camera = MultiRealSense(
            use_front_cam=True,
            use_right_cam=False,
            front_num_points=NUM_POINTS,
            use_grid_sampling=True
        )
        cprint("Camera initialized and running in background.", "green")

        self.recording = False
        self.color_list = []
        self.depth_list = []
        self.cloud_list = []
        self.action_list = []
        self.qpos_list = []
        self.start_time = None

    def run(self):
        """ Main loop for interactive data collection """
        print("Keyboard controls:")
        print("  's' : Start / Stop recording current episode")
        print("  'q' : Quit program (saves any ongoing episode)")
        print("Focus on the OpenCV preview window to use keys.")

        episode_count = 0

        try:
            while True:
                # Get the latest camera frames (always newest available)
                cam_dict = self.camera()

                pcd = (cam_dict['point_cloud'])
                color = (cam_dict['color'])
                depth = (cam_dict['depth'])

                # Show preview with recording indicator
                display_img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                if self.recording:
                    cv2.putText(display_img, f"RECORDING - {len(self.action_list)} frames",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("RealSense L515 - RGB Preview", display_img)

                key = cv2.waitKey(1) & 0xFF

                # Read robot state synchronously with camera frame
                current_action = get_action()
                current_qpos   = get_env_qpos()

                if self.recording:
                    # Accumulate data for current episode
                    self.color_list.append(color)
                    if depth is not None:
                        self.depth_list.append(depth)
                    if pcd is not None:
                        self.cloud_list.append(pcd)
                    self.action_list.append(current_action)
                    self.qpos_list.append(current_qpos)

                    elapsed = time.time() - self.start_time
                    sys.stdout.write(f"\rRecording... {len(self.action_list)} frames | {elapsed:.1f}s")
                    sys.stdout.flush()

                # Handle key presses
                if key == ord('x'):
                    if not self.recording:
                        cprint("\n=== Started recording new episode ===", "green")
                        self.recording = True
                        self.start_time = time.time()
                        # Clear buffers for fresh episode
                        self.color_list.clear()
                        self.depth_list.clear()
                        self.cloud_list.clear()
                        self.action_list.clear()
                        self.qpos_list.clear()
                    else:
                        cprint("\n=== Stopped recording — saving episode ===", "yellow")
                        self.recording = False
                        self.save_episode(episode_count)
                        episode_count += 1

                if key == ord('l'):
                    cprint("\nQuit requested.", "red")
                    if self.recording:
                        cprint("Saving current unfinished episode before exit...", "yellow")
                        self.save_episode(episode_count)
                    break

                # Control approximate frame rate
                time.sleep(1.0 / RECORD_FPS_TARGET)

        finally:
            cv2.destroyAllWindows()
            self.camera.finalize()
            cprint("Camera processes terminated. Program finished.", "green")

    def save_episode(self, episode_id):
        """ Save accumulated episode data to HDF5 file """
        if len(self.action_list) == 0:
            cprint("No data collected in this episode — skipping save.", "yellow")
            return

        seq_length = len(self.action_list)



        # Generate filename with timestamp and episode index
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        record_file_name = f"episode_{episode_id:03d}_{timestamp}.h5"

        discard_end_length = 1  # Optional: discard last N frames (e.g. hand shake)

        with h5py.File(record_file_name, "w") as f:
            # Stack lists into arrays
            color_array = np.stack(self.color_list)      
            depth_array = np.stack(self.depth_list) 
            action_array = np.stack(self.action_list)
            env_qpos_array = np.stack(self.qpos_list)
            cloud_array = np.array(self.cloud_list)
            f.create_dataset("color", data=color_array[:-discard_end_length])
            f.create_dataset("depth", data=depth_array[:-discard_end_length])
            f.create_dataset("cloud", data=cloud_array[:-discard_end_length])
            f.create_dataset("env_qpos_proprioception", data=env_qpos_array[:-discard_end_length])
            f.create_dataset("action", data=action_array[:-discard_end_length])

        cprint(f"color shape: {color_array.shape}", "yellow")
        cprint(f"depth shape: {depth_array.shape if len(depth_array)>0 else 'None'}", "yellow")
        cprint(f"cloud: {len(cloud_array)} variable-length point clouds", "yellow")
        cprint(f"action shape: {action_array.shape}", "yellow")
        cprint(f"env_qpos shape: {env_qpos_array.shape}", "yellow")
        cprint(f"Saved {seq_length} steps to {record_file_name}", "yellow")

        # Optional rename
        # choice = input("Rename file? (y/n): ").strip().lower()
        choice = 'n'  # For automated testing, disable rename prompt
        if choice == 'y':
            new_name_input = input("New filename (without .h5): ").strip()
            new_path = record_file_name.replace("episode_", f"{new_name_input}_")
            try:
                os.rename(record_file_name, new_path)
                cprint(f"Renamed to: {new_path}", "green")
            except Exception as e:
                cprint(f"Rename failed: {e}", "red")


if __name__ == "__main__":
    piper.ConnectPort(True)
    collector = DatasetCollector()
    collector.run()