import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load the file
filename = '/Users/utx/Desktop/code/video_learning_samples/add_remove_lid/0.hdf5'


with h5py.File(filename, 'r') as f:
    
    # 1. Get the list of all body parts
    body_parts = list(f['transforms'].keys())
    
    xs, ys, zs = [], [], []
    
    print(f"Plotting Frame 0 for {len(body_parts)} body parts...")

    # 2. Extract X, Y, Z for Frame 0
    for part in body_parts:
        # Get the 4x4 matrix for frame 0
        matrix = f['transforms'][part][0] 
        
        # The position is in the last column (index 3), first 3 rows
        x = matrix[0, 3]
        y = matrix[1, 3]
        z = matrix[2, 3]
        
        xs.append(x)
        ys.append(y)
        zs.append(z)

# 3. Plot in 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the joints
ax.scatter(xs, ys, zs, c='r', marker='o')

# Label a few key joints so you know orientation
for i, part in enumerate(body_parts):
    if part in ['hip', 'head', 'rightHand', 'leftHand']: # Only label key parts to avoid clutter
        ax.text(xs[i], ys[i], zs[i], part, fontsize=9)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Skeleton Pose (Frame 0)')

plt.show()