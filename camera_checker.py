import mujoco
import cv2
import numpy as np

XML_PATH = 'triple pendulum.xml'
IMG_SIZE = 448

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=IMG_SIZE, width=IMG_SIZE)

def project_manually(model, data, pos_3d, cam_id, width, height):
    cam_pos = data.cam_xpos[cam_id]
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
    
    cam_rot_inv = cam_rot.T
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = cam_rot_inv
    view_matrix[:3, 3] = -cam_rot_inv @ cam_pos
    
    fovy = np.deg2rad(model.cam_fovy[cam_id])
    aspect = width / height
    f = 1.0 / np.tan(fovy / 2.0)
    
    proj_matrix = np.zeros((4, 4))
    proj_matrix[0, 0] = f / aspect
    proj_matrix[1, 1] = f
    proj_matrix[2, 2] = -1.0
    proj_matrix[2, 3] = -0.1
    proj_matrix[3, 2] = -1.0
    
    p_world = np.append(pos_3d, 1.0)
    p_cam = view_matrix @ p_world
    p_clip = proj_matrix @ p_cam
    
    if p_clip[3] != 0:
        ndc = p_clip[:3] / p_clip[3]
    else:
        return None
        
    u = int((ndc[0] + 1.0) * 0.5 * width)
    v = int((1.0 - ndc[1]) * 0.5 * height)
    return (u, v)

def get_visual_check():
    thetas = np.random.uniform(-np.pi, np.pi, size=3)

    x = np.random.uniform(-0.5, 0.5)
    
    data.qpos[:] = np.concatenate(([x], thetas))
    mujoco.mj_forward(model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'agent_cam')
    renderer.update_scene(data, camera='agent_cam')
    img = renderer.render()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    joint_names = ['pin1', 'pin2', 'pin3']
    points = [data.body('cart').xpos] + [data.joint(name).xanchor for name in joint_names]
    
    for pos_3d in points:
        pixel = project_manually(model, data, pos_3d, cam_id, IMG_SIZE, IMG_SIZE)
        if pixel:
            cv2.circle(img_bgr, pixel, 6, (0, 0, 255), -1)

    return img_bgr

if __name__ == "__main__":
    print("Нажмите любую клавишу для обновления кадра, 'q' для выхода.")
    while True:
        frame = get_visual_check()
        cv2.imshow('Manual Projection Check', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()