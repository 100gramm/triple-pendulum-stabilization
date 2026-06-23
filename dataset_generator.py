import multiprocessing as mp
import cv2
import h5py
import mujoco
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Global Configuration & Dataset Constants
# -----------------------------------------------------------------------------
XML_PATH = "triple pendulum.xml"
IMG_SIZE = 448
TOTAL_SAMPLES = 100_000_000
CHUNK_SIZE = 1000


def worker_task(num_samples):
    """
    Worker task executed by each process in parallel.
    Generates a batch (chunk) of synthetic dataset samples.
    Each sample contains a randomized visual environment, physics state,
    rendered grayscale image, projected 2D coordinates, and joint angles.
    """
    # Initialize local instances of MuJoCo model and data for thread safety
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    # Setup the offscreen renderer and look up camera ID
    renderer = mujoco.Renderer(model, height=IMG_SIZE, width=IMG_SIZE)
    cam_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "agent_cam"
    )
    
    # Store original camera values to apply bounded domain randomization
    base_cam_pos = model.cam_pos[0].copy()
    base_cam_fovy = model.cam_fovy[0]

    # Pre-allocate local arrays for the generated chunk data
    images = np.zeros((num_samples, IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
    coords_list = np.zeros((num_samples, 8), dtype=np.float32)
    angles_list = np.zeros((num_samples, 3), dtype=np.float32)

    for i in range(num_samples):
        # 1. Visual Randomizer: Randomize lighting positions and directions
        model.light_pos[0] = np.array([0, 0, 2]) + np.random.uniform(
            -0.5, 0.5, 3
        )
        model.light_dir[0] = np.array([0, 0, -1]) + np.random.uniform(
            -0.1, 0.1, 3
        )
        
        # Uniformly randomize RGB color across all environment geometries
        color = np.random.uniform(0.2, 0.8, size=3)
        for g in range(model.geom_rgba.shape[0]):
            model.geom_rgba[g, :3] = color
            
        # Randomize camera extrinsic parameters (position) and intrinsic (FOV)
        model.cam_pos[0] = base_cam_pos + np.random.uniform(
            -0.05, 0.05, size=3
        )
        model.cam_fovy[0] = base_cam_fovy + np.random.uniform(-3.0, 3.0)

        # 2. State Sampling: Set random positions for the cart and 3 joint links
        thetas = np.random.uniform(-np.pi, np.pi, size=3)
        x = np.random.uniform(-0.8, 0.8)
        data.qpos[:] = np.concatenate(([x], thetas))
        
        # Run forward kinematics to update positions of all bodies and anchors
        mujoco.mj_forward(model, data)

        # 3. Image Rendering and Post-processing
        renderer.update_scene(data, camera="agent_cam")
        img = renderer.render()
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply data augmentation (Gaussian Blur) with 20% probability
        if np.random.rand() > 0.8:
            img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
        images[i] = img_gray[..., np.newaxis]

        # 4. Camera Projection: Compute View and Projection Matrices
        cam_pos = data.cam_xpos[cam_id]
        cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
        cam_rot_inv = cam_rot.T
        
        # Construct the 4x4 View Matrix (world-to-camera transformation space)
        view_mat = np.eye(4)
        view_mat[:3, :3] = cam_rot_inv
        view_mat[:3, 3] = -cam_rot_inv @ cam_pos
        
        # Construct the 4x4 Perspective Projection Matrix
        fovy = np.deg2rad(model.cam_fovy[cam_id])
        f = 1.0 / np.tan(fovy / 2.0)
        proj_mat = np.array([
            [f, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, -1, -0.1],
            [0, 0, -1, 0]
        ])
        
        # Get 3D global coordinates of the cart and the 3 link pins
        points = [data.body("cart").xpos] + [
            data.joint(name).xanchor for name in ["pin1", "pin2", "pin3"]
        ]
        
        # Project 3D points to 2D Normalized Device Coordinates (NDC)
        c_idx = 0
        for p in points:
            p_clip = proj_mat @ (view_mat @ np.append(p, 1.0))
            if p_clip[3] != 0:
                ndc = p_clip[:3] / p_clip[3]
                # Map NDC coordinates [-1, 1] to normalized image space [0, 1]
                coords_list[i, c_idx : c_idx + 2] = [
                    (ndc[0] + 1.0) * 0.5,
                    (1.0 - ndc[1]) * 0.5
                ]
                c_idx += 2
        
        # Store ground truth joint angles
        angles_list[i] = thetas

    return images, coords_list, angles_list


# -----------------------------------------------------------------------------
# Main Dataset Generation Loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    num_cpus = 10 
    num_chunks = TOTAL_SAMPLES // CHUNK_SIZE
    dataset_path = "triple_pendulum_dataset.hdf5"
    
    # Initialize an empty HDF5 container file with gzip compression
    with h5py.File(dataset_path, "w") as f:
        f.create_dataset(
            "images",
            shape=(TOTAL_SAMPLES, IMG_SIZE, IMG_SIZE, 1),
            dtype=np.uint8,
            compression="gzip"
        )
        f.create_dataset(
            "coords", shape=(TOTAL_SAMPLES, 8), dtype=np.float32
        )
        f.create_dataset(
            "angles", shape=(TOTAL_SAMPLES, 3), dtype=np.float32
        )

    print(
        f"Launching: {num_chunks} chunks of {CHUNK_SIZE} "
        f"samples across {num_cpus} CPU cores."
    )
    
    # Parallelize data generation utilizing a Multiprocessing Pool
    with mp.Pool(processes=num_cpus) as pool:
        pbar = tqdm(total=TOTAL_SAMPLES, desc="Generating")
        current_idx = 0
        
        # Stream computed chunks from background workers directly onto disk
        for imgs, coords, angles in pool.imap(
            worker_task, [CHUNK_SIZE] * num_chunks
        ):
            with h5py.File(dataset_path, "a") as f:
                num_received = imgs.shape[0]
                end_idx = current_idx + num_received
                
                # Write arrays into allocated block slices
                f["images"][current_idx:end_idx] = imgs
                f["coords"][current_idx:end_idx] = coords
                f["angles"][current_idx:end_idx] = angles
                
                current_idx = end_idx
            
            pbar.update(num_received)
            
        pbar.close()

    print("Finished. Dataset saved successfully.")