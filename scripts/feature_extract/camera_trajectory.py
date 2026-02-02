import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

def generate_all_3d_paths(base_path="./", output_folder="video_learning_samples"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Folders to process
    tasks = ['add_remove_lid', 'assemble_disassemble_furniture_bench_stool', 'basic_pick_and_place', 'open_close']

    for task in tasks:
        task_path = os.path.join(base_path, task)
        if not os.path.exists(task_path): continue
        
        print(f"Generating 3D paths for: {task}")
        h5_files = glob.glob(os.path.join(task_path, "*.hdf5"))
        
        # We also create a single plot for the whole task to show consistency
        fig_task = plt.figure(figsize=(10, 8))
        ax_task = fig_task.add_subplot(111, projection='3d')
        
        for h5_file in h5_files:
            file_id = os.path.splitext(os.path.basename(h5_file))[0]
            with h5py.File(h5_file, 'r') as h5:
                # Extract the 4x4 matrix and get the translation (x,y,z)
                # Note: transforms/camera is the (N, 4, 4) pose matrix
                cam_poses = h5['transforms/camera'][()]
                pos = cam_poses[:, :3, 3] 

                # 1. Plot Individual Trial
                fig_ind = plt.figure(figsize=(8, 6))
                ax_ind = fig_ind.add_subplot(111, projection='3d')
                ax_ind.plot(pos[:, 0], pos[:, 2], pos[:, 1], color='blue', lw=2)
                ax_ind.scatter(pos[0,0], pos[0,2], pos[0,1], color='green', s=100, label='Start')
                ax_ind.scatter(pos[-1,0], pos[-1,2], pos[-1,1], color='red', s=100, label='End')
                ax_ind.set_title(f"3D Path: {task} (Video {file_id})")
                ax_ind.set_xlabel('X (meters)')
                ax_ind.set_ylabel('Z (meters)')
                ax_ind.set_zlabel('Y (meters)')
                ax_ind.legend()
                
                plt.savefig(os.path.join(output_folder, f"{task}_{file_id}_3d_path.png"))
                plt.close()

                # 2. Add to Task-Wide Overlay
                ax_task.plot(pos[:, 0], pos[:, 2], pos[:, 1], alpha=0.5, label=f"Trial {file_id}")

        # Finalize and save the comparison plot
        ax_task.set_title(f"Generalized Trajectory: {task} (All Trials)")
        ax_task.set_xlabel('X (m)')
        ax_task.set_ylabel('Z (m)')
        ax_task.set_zlabel('Y (m)')
        plt.savefig(os.path.join(output_folder, f"{task}_comparison_overlay.png"))
        plt.close()

# Execute
generate_all_3d_paths("video_learning_samples")