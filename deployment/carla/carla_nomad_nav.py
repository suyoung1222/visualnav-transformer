#!/usr/bin/env python3
"""
NoMaD goal-conditioned navigation in CARLA.

Loads a pre-recorded topological map and navigates along it using the NoMaD diffusion
policy.  No ROS required — pure Python bridge between the CARLA Python API and the
visualnav-transformer inference code.

Architecture:
  CARLA sensor  →  context_queue  →  NoMaD inference  →  PD controller  →  kinematic update
  (RGB camera)     (PIL images)       (diffusion)          (v, w)            (set_transform)
"""

import os
import sys
import math
import time
import queue

import numpy as np
import torch
import yaml
import cv2
from PIL import Image as PILImage
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import carla

# ── Path setup: add deployment/src and train/ to import path ──────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
REPO_ROOT  = os.path.abspath(os.path.join(DEPLOY_DIR, '..'))
sys.path.insert(0, os.path.join(DEPLOY_DIR, 'src'))
sys.path.insert(0, os.path.join(REPO_ROOT,  'train'))

from utils import load_model, transform_images, to_numpy         # deployment/src/utils.py
from vint_train.training.train_utils import get_action           # train/vint_train/

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_CONFIG  = os.path.join(DEPLOY_DIR, 'config', 'models.yaml')
ROBOT_CONFIG  = os.path.join(DEPLOY_DIR, 'config', 'robot.yaml')
TOPOMAP_DIR   = os.path.join(DEPLOY_DIR, 'topomaps', 'images', 'carla_map')

with open(ROBOT_CONFIG) as f:
    _robot = yaml.safe_load(f)
MAX_V    = _robot['max_v']      # 0.2 m/s
MAX_W    = _robot['max_w']      # 0.4 rad/s
FRAME_HZ = _robot['frame_rate'] # 4 Hz
DT       = 1.0 / FRAME_HZ       # 0.25 s per control step

# ── CARLA parameters ──────────────────────────────────────────────────────────
CARLA_HOST   = 'localhost'
CARLA_PORT   = 2000
PHYS_HZ      = 20                          # simulation ticks per second
PHYS_DT      = 1.0 / PHYS_HZ              # 0.05 s
POLICY_EVERY = round(PHYS_HZ / FRAME_HZ)  # run policy every N physics ticks (= 5)
CAM_W, CAM_H = 320, 240
SPAWN_INDEX  = 0

# ── NoMaD inference parameters ───────────────────────────────────────────────
WAYPOINT_IDX  = 2   # which predicted waypoint to use for control (0–7)
NUM_SAMPLES   = 8   # number of diffusion trajectory samples
CLOSE_THRESH  = 3   # temporal-distance threshold to advance to next topomap node
RADIUS        = 4   # search window (±nodes) around closest node


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def clip_angle(theta: float) -> float:
    """Wrap angle to (-π, π]."""
    theta %= 2 * math.pi
    if -math.pi < theta < math.pi:
        return theta
    return theta - 2 * math.pi


def pd_controller(waypoint: np.ndarray):
    """
    Convert a predicted waypoint (dx, dy) in robot-local frame to (v, w).

    This is a direct port of deployment/src/pd_controller.py's pd_controller(),
    with ROS I/O replaced by plain return values.

    The waypoint represents cumulative displacement after WAYPOINT_IDX steps
    (already scaled by MAX_V / FRAME_HZ so units are meters per control step).

    Returns
    -------
    v : float  linear velocity  [m/s], clipped to [0, MAX_V]
    w : float  angular velocity [rad/s], clipped to [-MAX_W, MAX_W]
    """
    eps = 1e-8
    if len(waypoint) == 2:
        dx, dy = waypoint
    else:
        dx, dy, hx, hy = waypoint

    if len(waypoint) == 4 and abs(dx) < eps and abs(dy) < eps:
        v = 0.0
        w = clip_angle(math.atan2(hy, hx)) / DT
    elif abs(dx) < eps:
        v = 0.0
        w = math.copysign(math.pi / (2.0 * DT), dy)
    else:
        v = dx / DT
        w = math.atan(dy / dx) / DT

    v = float(np.clip(v, 0.0, MAX_V))
    w = float(np.clip(w, -MAX_W, MAX_W))
    return v, w


def carla_to_pil(carla_image) -> PILImage.Image:
    """Convert CARLA's BGRA image to a PIL RGB image."""
    arr = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    arr = arr.reshape((carla_image.height, carla_image.width, 4))
    rgb = arr[:, :, :3][:, :, ::-1].copy()   # BGRA → RGB
    return PILImage.fromarray(rgb)


def apply_kinematic(vehicle, v: float, w: float, dt: float):
    """
    Move vehicle using a differential-drive kinematic model.

    Physics is disabled on the vehicle (set_simulate_physics(False)), so
    set_transform acts as a direct teleport each tick.

    State update:
        x(t+dt)   = x(t) + v·cos(ψ)·dt
        y(t+dt)   = y(t) + v·sin(ψ)·dt
        ψ(t+dt)   = ψ(t) + w·dt

    CARLA convention: yaw=0 → facing +x, positive yaw = counterclockwise (left turn).
    This matches the ROS/NoMaD angular velocity sign convention.

    NOTE: if the simulated robot turns the wrong direction, negate the sign of w
    in this function: `new_yaw = t.rotation.yaw - math.degrees(w * dt)`.
    """
    t       = vehicle.get_transform()
    yaw_rad = math.radians(t.rotation.yaw)
    new_x   = t.location.x + v * math.cos(yaw_rad) * dt
    new_y   = t.location.y + v * math.sin(yaw_rad) * dt
    new_yaw = t.rotation.yaw + math.degrees(w * dt)
    vehicle.set_transform(carla.Transform(
        carla.Location(x=new_x, y=new_y, z=t.location.z),
        carla.Rotation(pitch=t.rotation.pitch, yaw=new_yaw, roll=t.rotation.roll)
    ))


def load_topomap(topomap_dir: str):
    """Load sorted topomap images from directory (filenames must be integers)."""
    fnames = sorted(os.listdir(topomap_dir), key=lambda x: int(os.path.splitext(x)[0]))
    imgs   = [PILImage.open(os.path.join(topomap_dir, f)).convert('RGB') for f in fnames]
    print(f'Loaded {len(imgs)} topomap nodes from {topomap_dir}')
    return imgs


def draw_hud(frame_bgr: np.ndarray,
             waypoint: np.ndarray,
             v: float, w: float,
             node: int, total: int) -> np.ndarray:
    """
    Overlay an arrow showing the predicted waypoint direction and status text.

    The arrow is drawn from the image centre; dx (forward in robot frame) maps to
    the upper part of the image, dy (left in robot frame) maps leftward.
    """
    h, wd = frame_bgr.shape[:2]
    out   = frame_bgr.copy()
    cx, cy = wd // 2, h // 2

    # Render as top-down mini-map arrow:
    #   robot forward (dx > 0) → image up   (y decreases)
    #   robot left    (dy > 0) → image left (x decreases)
    ppm   = 500.0
    end_x = int(np.clip(cx - waypoint[1] * ppm, 0, wd - 1))
    end_y = int(np.clip(cy - waypoint[0] * ppm, 0, h  - 1))

    cv2.arrowedLine(out, (cx, cy), (end_x, end_y), (0, 255, 0), 2, tipLength=0.35)
    cv2.circle(out, (cx, cy), 4, (0, 255, 0), -1)

    # Status text
    cv2.rectangle(out, (0, 0), (wd, 52), (0, 0, 0), -1)
    cv2.putText(out, f'Node {node}/{total-1}   v={v:.3f} m/s   w={w:.3f} rad/s',
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
    cv2.putText(out, f'waypoint  dx={waypoint[0]:.4f}  dy={waypoint[1]:.4f} m',
                (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 255), 1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load NoMaD model ──────────────────────────────────────────────────────
    with open(MODEL_CONFIG) as f:
        model_paths = yaml.safe_load(f)

    # Paths in models.yaml are relative to deployment/config/ — resolve them from there
    config_dir = os.path.join(DEPLOY_DIR, 'config')
    cfg_path   = os.path.normpath(os.path.join(config_dir, model_paths['nomad']['config_path']))
    ckpt_path  = os.path.normpath(os.path.join(config_dir, model_paths['nomad']['ckpt_path']))

    with open(cfg_path) as f:
        model_params = yaml.safe_load(f)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'NoMaD weights not found: {ckpt_path}\n'
            'Download them from: https://github.com/robodhruv/visualnav-transformer#model-weights\n'
            'Then place nomad.pth in deployment/model_weights/'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading NoMaD  ({ckpt_path})  →  {device}')
    model = load_model(ckpt_path, model_params, device)
    model.eval()
    print('Model ready.')

    context_size  = model_params['context_size']         # 3
    image_size    = model_params['image_size']            # [96, 96]
    len_traj      = model_params['len_traj_pred']         # 8
    n_diff_iters  = model_params['num_diffusion_iters']   # 10
    do_normalize  = model_params.get('normalize', True)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=n_diff_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )

    # ── Load topomap ──────────────────────────────────────────────────────────
    if not os.path.isdir(TOPOMAP_DIR) or len(os.listdir(TOPOMAP_DIR)) == 0:
        raise FileNotFoundError(
            f'Topomap directory empty or missing: {TOPOMAP_DIR}\n'
            'Run carla_record_topomap.py first to record a path.'
        )
    topomap   = load_topomap(TOPOMAP_DIR)
    n_nodes   = len(topomap)
    goal_node = n_nodes - 1

    # ── CARLA setup ───────────────────────────────────────────────────────────
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world  = client.get_world()

    orig_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = PHYS_DT
    world.apply_settings(settings)

    bplib = world.get_blueprint_library()

    vehicle_bp = bplib.find('vehicle.micro.microlino')
    vehicle_bp.set_attribute('role_name', 'robot')
    spawn_pts = world.get_map().get_spawn_points()
    vehicle   = world.try_spawn_actor(vehicle_bp, spawn_pts[SPAWN_INDEX])
    if vehicle is None:
        for sp in spawn_pts[1:5]:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                break
    if vehicle is None:
        raise RuntimeError('Could not spawn vehicle.')
    vehicle.set_simulate_physics(False)   # kinematic control via set_transform
    print(f'Spawned {vehicle.type_id}  id={vehicle.id}')

    cam_bp = bplib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(CAM_W))
    cam_bp.set_attribute('image_size_y', str(CAM_H))
    cam_bp.set_attribute('fov', '90')
    cam_bp.set_attribute('sensor_tick', '0.0')
    cam_tf = carla.Transform(
        carla.Location(x=vehicle.bounding_box.extent.x + 0.1, z=1.2),
        carla.Rotation(pitch=-5.0)
    )
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

    img_q = queue.Queue(maxsize=2)

    def on_image(img):
        if img_q.full():
            try:
                img_q.get_nowait()
            except queue.Empty:
                pass
        img_q.put(img)

    camera.listen(on_image)

    # ── Navigation state ──────────────────────────────────────────────────────
    context_queue   = []
    closest_node    = 0
    reached_goal    = False
    chosen_waypoint = np.zeros(2)
    v_cmd, w_cmd    = 0.0, 0.0
    tick_count      = 0

    cv2.namedWindow('NoMaD | CARLA', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('NoMaD | CARLA', CAM_W * 2, CAM_H * 2)
    print('Navigation started.  Press Q in the OpenCV window to stop.')

    try:
        while True:
            world.tick()
            tick_count += 1

            # ── Collect camera frame ──────────────────────────────────────────
            try:
                carla_img = img_q.get_nowait()
            except queue.Empty:
                carla_img = None

            if carla_img is not None:
                pil_img = carla_to_pil(carla_img)
                if len(context_queue) >= context_size + 1:
                    context_queue.pop(0)
                context_queue.append(pil_img)

            # ── Run NoMaD inference at FRAME_HZ ──────────────────────────────
            if (tick_count % POLICY_EVERY == 0
                    and len(context_queue) > context_size
                    and not reached_goal):

                with torch.no_grad():
                    # Encode current observation stack
                    obs_images = transform_images(context_queue, image_size, center_crop=False)
                    obs_images = torch.split(obs_images, 3, dim=1)
                    obs_images = torch.cat(obs_images, dim=1).to(device)
                    # [1, 3*(context_size+1), H, W]

                    mask = torch.zeros(1).long().to(device)  # 0 = use goal conditioning

                    # Collect candidate subgoal images from topomap window
                    start = max(closest_node - RADIUS, 0)
                    end   = min(closest_node + RADIUS + 1, goal_node)
                    goal_imgs = [
                        transform_images(topomap[i], image_size, center_crop=False).to(device)
                        for i in range(start, end + 1)
                    ]
                    goal_imgs = torch.cat(goal_imgs, dim=0)
                    # [N_cands, 3, H, W]

                    # Encode obs+goal pairs to get conditioning embeddings
                    obsgoal_cond = model(
                        'vision_encoder',
                        obs_img=obs_images.repeat(len(goal_imgs), 1, 1, 1),
                        goal_img=goal_imgs,
                        input_goal_mask=mask.repeat(len(goal_imgs)),
                    )
                    # [N_cands, encoding_size=256]

                    # Predict temporal distance to each candidate node
                    dists = to_numpy(
                        model('dist_pred_net', obsgoal_cond=obsgoal_cond).flatten()
                    )
                    min_idx      = int(np.argmin(dists))
                    closest_node = min_idx + start
                    print(f'  closest_node={closest_node}  dist={dists[min_idx]:.2f}')

                    # Select subgoal: advance if close enough
                    sg_idx   = min(
                        min_idx + int(dists[min_idx] < CLOSE_THRESH),
                        len(obsgoal_cond) - 1
                    )
                    obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)
                    # [1, 256]

                    # ── Diffusion action sampling ─────────────────────────────
                    # Repeat conditioning for NUM_SAMPLES parallel trajectories
                    obs_cond = obs_cond.repeat(NUM_SAMPLES, 1)
                    # [NUM_SAMPLES, 256]

                    # Start from pure Gaussian noise in action space
                    naction = torch.randn(
                        (NUM_SAMPLES, len_traj, 2), device=device
                    )
                    noise_scheduler.set_timesteps(n_diff_iters)

                    # DDPM reverse denoising loop  (K=10 steps)
                    for k in noise_scheduler.timesteps:
                        noise_pred = model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond,
                        )
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction,
                        ).prev_sample
                    # naction: [NUM_SAMPLES, len_traj, 2]  (still normalized)

                    # Unnormalize deltas and cumsum to absolute waypoints
                    naction = to_numpy(get_action(naction))
                    # [NUM_SAMPLES, len_traj, 2]  in metric space

                chosen_waypoint = naction[0][WAYPOINT_IDX].copy()
                # Scale from model space to per-step displacement used by pd_controller
                if do_normalize:
                    chosen_waypoint[:2] *= (MAX_V / FRAME_HZ)

                v_cmd, w_cmd = pd_controller(chosen_waypoint)

                reached_goal = (closest_node == goal_node)
                if reached_goal:
                    print('Reached goal!  Stopping.')
                    v_cmd, w_cmd = 0.0, 0.0

            # ── Apply kinematic control ───────────────────────────────────────
            apply_kinematic(vehicle, v_cmd, w_cmd, PHYS_DT)

            # ── OpenCV display ────────────────────────────────────────────────
            if carla_img is not None:
                arr  = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
                bgr  = arr.reshape((carla_img.height, carla_img.width, 4))[:, :, :3].copy()
                hud  = draw_hud(bgr, chosen_waypoint, v_cmd, w_cmd, closest_node, n_nodes)
                cv2.imshow('NoMaD | CARLA', hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

            if reached_goal:
                time.sleep(3.0)
                break

    finally:
        print('Cleaning up...')
        cv2.destroyAllWindows()
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        world.apply_settings(orig_settings)
        print('Done.')


if __name__ == '__main__':
    main()
