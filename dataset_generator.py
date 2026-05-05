import mujoco
import numpy as np
import cv2
import h5py
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import multiprocessing as mp

XML_PATH = 'triple pendulum.xml'
IMG_SIZE = 448
TOTAL_SAMPLES = 100_000
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

class VisualRandomizer:
    def __init__(self, model):
        self.model = model
        self.base_cam_pos = model.cam_pos[0].copy()
        self.base_cam_fovy = model.cam_fovy[0]
    
    def randomize_light(self):
        self.model.light_pos[0] = np.array([0, 0, 2]) + np.random.uniform(-0.5, 0.5, 3)
        self.model.light_dir[0] = np.array([0, 0, -1]) + np.random.uniform(-0.1, 0.1, 3)

    def randomize_geometry(self):
        color = np.random.uniform(0.2, 0.8, size=3)
        for i in range(self.model.geom_rgba.shape[0]):
            self.model.geom_rgba[i, :3] = color

    def randomize_camera(self):
        cam_noise = np.random.uniform(-0.05, 0.05, size=3)
        self.model.cam_pos[0] = self.base_cam_pos + cam_noise
        fov_noise = np.random.uniform(-3.0, 3.0)
        self.model.cam_fovy[0] = self.base_cam_fovy + fov_noise

def sample_state(data):
    thetas = np.random.uniform(-np.pi, np.pi, size=3)
    x = np.random.uniform(-0.8, 0.8)
    qpos = np.concatenate(([x], thetas))
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return qpos

class PointProjector:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'agent_cam')
        self.joint_names = ['pin1', 'pin2', 'pin3']
    
    def project(self) -> np.ndarray:
        cam_pos = self.data.cam_xpos[self.cam_id]
        cam_rot = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        
        cam_rot_inv = cam_rot.T
        view_mat = np.eye(4)
        view_mat[:3, :3] = cam_rot_inv
        view_mat[:3, 3] = -cam_rot_inv @ cam_pos
        
        fovy = np.deg2rad(self.model.cam_fovy[self.cam_id])
        f = 1.0 / np.tan(fovy / 2.0)
        proj_mat = np.array([
            [f/1.0, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, -1, -0.1],
            [0, 0, -1, 0]
        ])
        
        coords = []
        points = [self.data.body('cart').xpos] + [self.data.joint(name).xanchor for name in self.joint_names]
        
        for p in points:
            p_homo = np.append(p, 1.0)
            p_cam = view_mat @ p_homo
            p_clip = proj_mat @ p_cam
            
            if p_clip[3] == 0: continue
            ndc = p_clip[:3] / p_clip[3]
            
            u = (ndc[0] + 1.0) * 0.5
            v = (1.0 - ndc[1]) * 0.5
            coords.extend([u, v])
            
        return np.array(coords, dtype=np.float32)

class DatasetWriter:
    def __init__(self, hdf5_path):
        self.hdf5_file = h5py.File(hdf5_path, 'w')
        self.hdf5_file.create_dataset('images', shape=(100_000, IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8, compression='gzip')
        self.hdf5_file.create_dataset('coords', shape=(100_000, 8), dtype=np.float32)
        self.hdf5_file.create_dataset('angles', shape=(100_000, 3), dtype=np.float32)
        self.index = 0

    def write(self, img, coords, angles):
        self.hdf5_file['images'][self.index] = img[..., np.newaxis]
        self.hdf5_file['coords'][self.index] = coords
        self.hdf5_file['angles'][self.index] = angles
        self.index += 1



def worker_task(num_samples):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_SIZE, width=IMG_SIZE)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'agent_cam')
    
    base_cam_pos = model.cam_pos[0].copy()
    base_cam_fovy = model.cam_fovy[0]

    images = np.zeros((num_samples, IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
    coords_list = np.zeros((num_samples, 8), dtype=np.float32)
    angles_list = np.zeros((num_samples, 3), dtype=np.float32)

    for i in range(num_samples):
        model.light_pos[0] = np.array([0, 0, 2]) + np.random.uniform(-0.5, 0.5, 3)
        model.light_dir[0] = np.array([0, 0, -1]) + np.random.uniform(-0.1, 0.1, 3)
        
        color = np.random.uniform(0.2, 0.8, size=3)
        for g in range(model.geom_rgba.shape[0]):
            model.geom_rgba[g, :3] = color
            
        model.cam_pos[0] = base_cam_pos + np.random.uniform(-0.05, 0.05, size=3)
        model.cam_fovy[0] = base_cam_fovy + np.random.uniform(-3.0, 3.0)

        thetas = np.random.uniform(-np.pi, np.pi, size=3)
        x = np.random.uniform(-0.8, 0.8)
        data.qpos[:] = np.concatenate(([x], thetas))
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera='agent_cam')
        img = renderer.render()
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if np.random.rand() > 0.8:
            img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
        images[i] = img_gray[..., np.newaxis]

        cam_pos = data.cam_xpos[cam_id]
        cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
        cam_rot_inv = cam_rot.T
        view_mat = np.eye(4)
        view_mat[:3, :3] = cam_rot_inv
        view_mat[:3, 3] = -cam_rot_inv @ cam_pos
        
        fovy = np.deg2rad(model.cam_fovy[cam_id])
        f = 1.0 / np.tan(fovy / 2.0)
        proj_mat = np.array([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, -1, -0.1], [0, 0, -1, 0]])
        
        points = [data.body('cart').xpos] + [data.joint(name).xanchor for name in ['pin1', 'pin2', 'pin3']]
        c_idx = 0
        for p in points:
            p_clip = proj_mat @ (view_mat @ np.append(p, 1.0))
            if p_clip[3] != 0:
                ndc = p_clip[:3] / p_clip[3]
                coords_list[i, c_idx:c_idx+2] = [(ndc[0] + 1.0) * 0.5, (1.0 - ndc[1]) * 0.5]
                c_idx += 2
        
        angles_list[i] = thetas

    return images, coords_list, angles_list

if __name__ == "__main__":
    num_cpus = 10
    samples_per_worker = TOTAL_SAMPLES // num_cpus
    
    print(f"Запуск на {num_cpus} ядрах...")
    with mp.Pool(processes=num_cpus) as pool:
        results = list(tqdm(pool.imap(worker_task, [samples_per_worker] * num_cpus), total=num_cpus))

    print("Запись в файл...")
    with h5py.File('triple_pendulum_dataset.hdf5', 'w') as f:
        d_img = f.create_dataset('images', shape=(TOTAL_SAMPLES, IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8, compression='gzip')
        d_coord = f.create_dataset('coords', shape=(TOTAL_SAMPLES, 8), dtype=np.float32)
        d_angle = f.create_dataset('angles', shape=(TOTAL_SAMPLES, 3), dtype=np.float32)
        
        for i, (imgs, coords, angles) in enumerate(results):
            start = i * samples_per_worker
            end = start + samples_per_worker
            d_img[start:end] = imgs
            d_coord[start:end] = coords
            d_angle[start:end] = angles

    print("Готово.")