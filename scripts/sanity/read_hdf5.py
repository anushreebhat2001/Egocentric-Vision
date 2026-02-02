import h5py


file = '/Users/utx/Desktop/code/egodexrobotics/video_learning_samples/add_remove_lid/0.hdf5'
def print_structure(name, obj):
    # This function checks if an item is a Group (folder) or Dataset (file)
    if isinstance(obj, h5py.Group):
        print(f"📁 FOLDER: {name}")
    elif isinstance(obj, h5py.Dataset):
        # If it's data, print its size (Shape) and type (int, float, etc)
        print(f"📄 DATA:   {name}  -->  Shape: {obj.shape} | Type: {obj.dtype}")

# Open the file (Make sure 'filename.h5' matches your actual file path)
# Based on your error, the file path in your code seems correct, just the key was wrong.
with h5py.File(file, 'r') as f:
    print(f"\n--- FILE CONTENTS ---")
    # 'visititems' walks through the whole file tree automatically
    f.visititems(print_structure)
    print("---------------------\n")