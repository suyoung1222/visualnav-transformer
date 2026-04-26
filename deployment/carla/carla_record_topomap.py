#!/usr/bin/env python3
"""
Record a topological map by manually driving a vehicle in CARLA.

Controls: W/Up=throttle  S/Down=brake  A/Left=steer left  D/Right=steer right  Q=quit
Images are saved at a fixed time interval to deployment/topomaps/images/carla_map/.
Run this before carla_nomad_nav.py.
"""

import os
import sys
import time
import queue

import numpy as np
import carla
import pygame
from PIL import Image as PILImage

# ── Output directory ──────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, '..', 'topomaps', 'images', 'carla_map')

# ── Parameters ────────────────────────────────────────────────────────────────
RECORD_INTERVAL_S = 0.5   # seconds between saved frames (≈ 2 images/s, matches 4 Hz policy)
CAM_W, CAM_H     = 320, 240
DISPLAY_W, DISPLAY_H = 640, 480
CARLA_HOST        = 'localhost'
CARLA_PORT        = 2000
SPAWN_INDEX       = 0     # index into world.get_map().get_spawn_points()


def carla_bgra_to_rgb_array(carla_image) -> np.ndarray:
    """Convert carla.Image (BGRA flat bytes) to HxWx3 RGB uint8 numpy array."""
    arr = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    arr = arr.reshape((carla_image.height, carla_image.width, 4))
    return arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Pygame display ────────────────────────────────────────────────────────
    pygame.init()
    display = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption('CARLA Topomap Recorder  |  WASD=drive  Q=quit')
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont(None, 28)

    # ── CARLA connection ──────────────────────────────────────────────────────
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world  = client.get_world()

    # Synchronous mode: we control exactly when the simulation ticks
    orig_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode   = True
    settings.fixed_delta_seconds = 0.05   # 20 Hz physics
    world.apply_settings(settings)

    bplib = world.get_blueprint_library()

    # ── Spawn vehicle ─────────────────────────────────────────────────────────
    vehicle_bp = bplib.find('vehicle.micro.microlino')
    vehicle_bp.set_attribute('role_name', 'recorder')
    spawn_pts = world.get_map().get_spawn_points()
    vehicle   = world.try_spawn_actor(vehicle_bp, spawn_pts[SPAWN_INDEX])
    if vehicle is None:
        # Fall back: try a few nearby spawn points
        for sp in spawn_pts[1:5]:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                break
    if vehicle is None:
        raise RuntimeError('Could not spawn vehicle. Is CARLA running and the map loaded?')
    print(f'Spawned {vehicle.type_id}  id={vehicle.id}')

    # ── Attach front camera ───────────────────────────────────────────────────
    cam_bp = bplib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(CAM_W))
    cam_bp.set_attribute('image_size_y', str(CAM_H))
    cam_bp.set_attribute('fov', '90')
    cam_bp.set_attribute('sensor_tick', '0.0')   # fire every simulation tick
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

    # ── Recording state ───────────────────────────────────────────────────────
    frame_index    = 0
    last_save_time = time.time()
    surface        = None
    recording      = True   # set False only on Q-press; saves happen automatically

    print(f'Output dir: {OUTPUT_DIR}')
    print('Drive around to record topomap frames.  Press Q to finish.')

    try:
        running = True
        while running:
            world.tick()

            # ── Event handling ────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                running = False

            # ── Vehicle control ───────────────────────────────────────────────
            ctrl = carla.VehicleControl()
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                ctrl.throttle = 0.5
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                ctrl.brake = 0.7
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                ctrl.steer = -0.4
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                ctrl.steer = 0.4
            vehicle.apply_control(ctrl)

            # ── Grab latest camera frame ──────────────────────────────────────
            try:
                carla_img = img_q.get_nowait()
            except queue.Empty:
                carla_img = None

            if carla_img is not None:
                rgb = carla_bgra_to_rgb_array(carla_img)

                # Save at regular interval
                now = time.time()
                if recording and (now - last_save_time) >= RECORD_INTERVAL_S:
                    pil = PILImage.fromarray(rgb)
                    path = os.path.join(OUTPUT_DIR, f'{frame_index}.png')
                    pil.save(path)
                    print(f'  saved {frame_index}.png')
                    frame_index    += 1
                    last_save_time  = now

                # Pygame surface for display (note: pygame uses BGR-like swapaxes)
                surf    = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                surface = pygame.transform.scale(surf, (DISPLAY_W, DISPLAY_H))

            # ── Render ────────────────────────────────────────────────────────
            if surface is not None:
                display.blit(surface, (0, 0))

            status = f'Frames saved: {frame_index}   WASD=drive   Q=quit'
            txt = font.render(status, True, (255, 255, 0))
            display.blit(txt, (10, 10))
            pygame.display.flip()
            clock.tick(20)

    finally:
        print(f'\nSaved {frame_index} frames to {OUTPUT_DIR}')
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        world.apply_settings(orig_settings)
        pygame.quit()


if __name__ == '__main__':
    main()
