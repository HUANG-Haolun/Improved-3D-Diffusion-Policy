import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import time
from omegaconf import OmegaConf
import pathlib
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import diffusion_policy_3d.common.gr1_action_util as action_util
import diffusion_policy_3d.common.rotation_util as rotation_util
import tqdm
import torch
import os 
import cv2
from piper_sdk import *
os.environ['WANDB_SILENT'] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


from diffusion_policy_3d.common.multi_realsense import MultiRealSense



import numpy as np
import torch
from termcolor import cprint

class PiPerEnvInference:
    """
    The deployment is running on the local computer of the robot.
    """
    def __init__(self, obs_horizon=2, action_horizon=8, device="gpu",
                use_point_cloud=True, use_image=True, img_size=224,
                 num_points=4096,):
        
        # obs/action
        self.use_point_cloud = use_point_cloud
        self.use_image = use_image
        
        
        # camera
        self.camera = MultiRealSense(use_front_cam=True, # by default we use single cam. but we also support multi-cam
                            front_num_points=num_points,
                            img_size=img_size)

        # horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # inference device
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        # robot comm
        self.piper1 = C_PiperInterface_V2("can0")
        self.piper1.ConnectPort(True)

        while( not self.piper1.EnablePiper()):
            time.sleep(0.01)
        print("Enabled Piper1!")
        self.piper1.GripperCtrl(0,1000,0x01, 0)
        self.init = [90, 0, 0, 0, 0, 0, 0]
        self.init = [round(self.init[i] * 1000) for i in range(len(self.init))]


    
    def step(self, action_list):
        
        for action_id in range(self.action_horizon):
            act = action_list[action_id]
            self.action_array.append(act)
            
            # act = action_util.joint25_to_joint32(act)
            # print("raw action:", act)
            # act = action_util.joint_TODO_to_joint12(act)
            
            filtered_act = act.copy()
            filtered_pos = filtered_act[:-1]
            filtered_handpos = filtered_act[-1]
            self.piper1.MotionCtrl_2(0x01, 0x01, 100, 0x00)
            #send command
            # print([filtered_act[i] for i in range(len(filtered_act))])

            self.piper1.JointCtrl(*[round(filtered_pos[i]*1000) for i in range(len(filtered_pos))])
            self.piper1.GripperCtrl(round(filtered_handpos*1000), 1000, 0x01, 0)


            cam_dict = self.camera()
            self.cloud_array.append(cam_dict['point_cloud'])
            self.color_array.append(cam_dict['color'])
            self.depth_array.append(cam_dict['depth'])
            # display_img = cv2.cvtColor(cam_dict['color'], cv2.COLOR_RGB2BGR)
            # cv2.imshow("RealSense L515 - RGB Preview", display_img)
            # cv2.waitKey(33)

            try:
                arm_pose = self.piper1.GetArmJointMsgs().joint_state.angles
                gripper = self.piper1.GetArmGripperMsgs().gripper_state.angle
                arm_pose = np.array(list(arm_pose) + [gripper])
            except:
                cprint("fail to fetch hand qpos. use default.", "red")
                arm_pose = np.ones(7)
            env_qpos = arm_pose
            self.env_qpos_array.append(env_qpos)
            
        
        agent_pos = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)
        obs_cloud = np.stack(self.cloud_array[-self.obs_horizon:], axis=0)
        obs_img = np.stack(self.color_array[-self.obs_horizon:], axis=0)

        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_point_cloud:
            obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)

        return obs_dict
    
    def reset(self, first_init=True):
        # init buffer
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.action_array = []
    
    
        # # pos init
        # qpos_init1 = np.array([-np.pi / 12, 0, 0, -1.6, 0, 0, 0, 
        #     -np.pi / 12, 0, 0, -1.6, 0, 0, 0])
        # qpos_init2 = np.array([-np.pi / 12, 0, 1.5, -1.6, 0, 0, 0, 
        #         -np.pi / 12, 0, -1.5, -1.6, 0, 0, 0])
        # hand_init = np.ones(12)
        # # hand_init = np.ones(12) * 0

        # if first_init:
        #     # ======== INIT ==========
        #     upbody_initpos = np.concatenate([qpos_init2])
        #     # self.arm_comm.init_set_pos(upbody_initpos)
        #     # self.arm_comm.send_hand_cmd(hand_init[6:], hand_init[:6])
        self.piper1.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.piper1.JointCtrl(*self.init[:-1])
        print("init pos:", self.init)
        time.sleep(0.2)
        self.piper1.GripperCtrl(abs(self.init[-1]), 1000, 0x01, 0)


        time.sleep(2)
        print(self.piper1.GetArmGripperMsgs())
        print("Robot ready!")
        # ======== INIT ==========
        # camera.start()
        cam_dict = self.camera()
        self.color_array.append(cam_dict['color'])
        self.depth_array.append(cam_dict['depth'])
        self.cloud_array.append(cam_dict['point_cloud'])

        try:
            arm_pose = self.piper1.GetArmJointMsgs().joint_state.angles
            gripper = self.piper1.GetArmGripperMsgs().gripper_state.angle
            arm_pose = np.array(list(arm_pose) + [gripper])
            arm_pose = np.ones(7)
        except:
            cprint("fail to fetch hand qpos. use default.", "red")
            arm_pose = np.ones(7)
        env_qpos = arm_pose
        self.env_qpos_array.append(env_qpos)
                        
    

        agent_pos = np.stack([self.env_qpos_array[-1]]*self.obs_horizon, axis=0)
        
        obs_cloud = np.stack([self.cloud_array[-1]]*self.obs_horizon, axis=0)
        obs_img = np.stack([self.color_array[-1]]*self.obs_horizon, axis=0)
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_point_cloud:
            obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)
        return obs_dict


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d','config'))
)
def main(cfg: OmegaConf):
    torch.manual_seed(42)
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    if workspace.__class__.__name__ == 'DPWorkspace':
        use_image = True
        use_point_cloud = False
    else:
        use_image = False
        use_point_cloud = True
        
    # fetch policy model
    policy = workspace.get_model()
    action_horizon = policy.horizon - policy.n_obs_steps + 1

    # pour
    roll_out_length_dict = {
        "pour": 300,
        "grasp": 1000,
        "wipe": 300,
    }
    # task = "wipe"
    task = "grasp"
    # task = "pour"
    roll_out_length = roll_out_length_dict[task]
    
    img_size = 384
    num_points = 4096
    first_init = True

    env = PiPerEnvInference(obs_horizon=2, action_horizon=action_horizon, device="cpu",
                             use_point_cloud=use_point_cloud,
                             use_image=use_image,
                             img_size=img_size,
                             num_points=num_points)

    
    obs_dict = env.reset(first_init=first_init)

    step_count = 0
    time_now = time.time()
    while step_count < roll_out_length:
        with torch.no_grad():
            # display_image = obs_dict['image'][0,-1].permute(1,2,0).cpu().numpy().astype(np.uint8)
            # cv2.imshow("input image", cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(1)
            
            action = policy(obs_dict)[0]
            action_list = [act.numpy() for act in action]
        
        obs_dict = env.step(action_list)
        step_count += action_horizon
        # print(f"step: {step_count}")
        # print(f"fps: {step_count / (time.time() - time_now)}")

    # if record_data:
    #     import h5py
    #     root_dir = "/home/mihawk/piper-learning-real/"
    #     save_dir = root_dir + "deploy_dir"
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     record_file_name = f"{save_dir}/demo.h5"
    #     color_array = np.array(env.color_array)
    #     depth_array = np.array(env.depth_array)
    #     cloud_array = np.array(env.cloud_array)
    #     qpos_array = np.array(env.qpos_array)
    #     with h5py.File(record_file_name, "w") as f:
    #         f.create_dataset("color", data=np.array(color_array))
    #         f.create_dataset("depth", data=np.array(depth_array))
    #         f.create_dataset("cloud", data=np.array(cloud_array))
    #         f.create_dataset("qpos", data=np.array(qpos_array))
        
    #     choice = input("whether to rename: y/n")
    #     if choice == "y":
    #         renamed = input("file rename:")
    #         os.rename(src=record_file_name, dst=record_file_name.replace("demo.h5", renamed+'.h5'))
    #         new_name = record_file_name.replace("demo.h5", renamed+'.h5')
    #         cprint(f"save data at step: {roll_out_length} in {new_name}", "yellow")
    #     else:
    #         cprint(f"save data at step: {roll_out_length} in {record_file_name}", "yellow")


if __name__ == "__main__":
    main()
