import h5py
import matplotlib.pyplot as plt
import numpy as np

def visualize_sample(idx):
    # Open the HDF5 dataset file containing the triple pendulum simulation data in read-only mode
    with h5py.File('triple_pendulum_dataset.hdf5', 'r') as f:
        img = f['images'][idx]
        coords = f['coords'][idx]
        
    # Reshape the flat coordinate array into a (4, 2) matrix representing 4 joints/points (X, Y)
    points = coords.reshape((4, 2))
    
    # Denormalize coordinates: scale normalized [0, 1] values back to original 448x448 pixel resolution
    pixel_points = points * 448
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img.squeeze(), cmap='gray')
    plt.scatter(pixel_points[:, 0], pixel_points[:, 1], c='red', s=50, label='Joints')
    plt.plot(pixel_points[:, 0], pixel_points[:, 1], c='yellow', linewidth=2, label='Pendulum')
    plt.title(f"Sample {idx} Projection Check")
    plt.legend()
    plt.show()

visualize_sample(50000)