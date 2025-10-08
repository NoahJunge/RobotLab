import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
from RobotUtils.CalibratedRobot import CalibratedRobot
from scipy.stats import norm
import math
from LocalizationPathing import LocalizationPathing
import random
import cv2
from LandmarkOccupancyGrid import LandmarkOccupancyGrid

# Flags
showGUI = False   # Whether or not to open GUI windows
onRobot = True    # Whether or not we are running on the Arlo robot

def isRunningOnArlo():
    return onRobot

try:
    from RobotUtils.Robot import Robot
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False

# Colors (BGR)
CRED      = (0, 0, 255)
CGREEN    = (0, 255, 0)
CBLUE     = (255, 0, 0)
CCYAN     = (255, 255, 0)
CYELLOW   = (0, 255, 255)
CMAGENTA  = (255, 0, 255)
CWHITE    = (255, 255, 255)
CBLACK    = (0, 0, 0)

# Landmarks (cm)
landmarkIDs = [6, 7]
landmarks = {
    6: (0.0,   0.0),
    7: (300.0, 0.0)
}
center = np.array([(landmarks[6][0] + landmarks[7][0]) / 2,
                   (landmarks[6][1] + landmarks[7][1]) / 2])

landmark_colors = [CRED, CGREEN]

def jet(x):
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)
    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world):
    offsetX, offsetY = 100, 250
    ymax = world.shape[0]
    world[:] = CWHITE

    max_weight = max((p.getWeight() for p in particles), default=1.0)

    for p in particles:
        x = int(p.getX() + offsetX)
        y = ymax - (int(p.getY() + offsetY))
        colour = jet(p.getWeight() / max_weight)
        cv2.circle(world, (x, y), 2, colour, 2)
        b = (int(p.getX() + 15.0*np.cos(p.getTheta())) + offsetX,
             ymax - (int(p.getY() + 15.0*np.sin(p.getTheta())) + offsetY))
        cv2.line(world, (x, y), b, colour, 2)

    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    a = (int(est_pose.getX()) + offsetX, ymax - (int(est_pose.getY()) + offsetY))
    b = (int(est_pose.getX() + 15.0*np.cos(est_pose.getTheta())) + offsetX,
         ymax - (int(est_pose.getX() + 15.0*np.sin(est_pose.getTheta())) + offsetY))
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)

def initialize_particles(num_particles):
    ps = []
    for i in range(num_particles):
        p = particle.Particle(
            600.0*np.random.ranf() - 100.0,
            600.0*np.random.ranf() - 250.0,
            np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi),
            1.0/num_particles
        )
        ps.append(p)
    return ps

def sample_motion_model(particles_list, distance, angle, sigma_d, sigma_theta):
    for p in particles_list:
        delta_x = distance * np.cos(p.getTheta() + angle)
        delta_y = distance * np.sin(p.getTheta() + angle)
        particle.move_particle(p, delta_x, delta_y, angle)
    particle.add_uncertainty_von_mises(particles_list, sigma_d, sigma_theta)

def motion_model_with_map(p, distance, angle, sigma_d, sigma_theta, grid):
    indices, valid = grid.world_to_grid([p.getX(), p.getY()])
    p_map = 0 if (not valid or grid.in_collision(indices)) else 1
    if p_map:
        sample_motion_model([p], distance, angle, sigma_d, sigma_theta)
        return 1
    return 0

def sample_motion_model_with_map(particles_list, distance, angle, sigma_d, sigma_theta, grid, max_tries=10):
    for p in particles_list:
        for attempt in range(max_tries):
            if motion_model_with_map(p, distance, angle, sigma_d, sigma_theta, grid) > 0:
                break
        else:
            p.setWeight(0.01)

def measurement_model(particle_list, landmarkIDs, dists, angles, sigma_d, sigma_theta):
    # Gaussian distance+bearing model to known landmarks
    for p in particle_list:
        x_i, y_i, theta_i = p.getX(), p.getY(), p.getTheta()
        p_obs = 1.0
        for landmarkID, dist, ang in zip(landmarkIDs, dists, angles):
            if landmarkID in landmarks:
                l_x, l_y = landmarks[landmarkID]
                dx, dy = (l_x - x_i), (l_y - y_i)
                d_i = np.hypot(dx, dy)

                # distance likelihood
                p_d_m = norm.pdf(dist, loc=d_i, scale=sigma_d)

                # bearing likelihood
                e_theta = np.array([np.cos(theta_i), np.sin(theta_i)])         # heading
                e_theta_hat = np.array([-np.sin(theta_i), np.cos(theta_i)])    # left-normal
                e_l = np.array([dx, dy]) / max(d_i, 1e-9)

                # clamp for numerical safety
                dot_main = float(np.clip(np.dot(e_l, e_theta), -1.0, 1.0))
                sign_side = np.sign(np.dot(e_l, e_theta_hat))
                phi_i = sign_side * np.arccos(dot_main)
                p_phi_m = norm.pdf(ang, loc=phi_i, scale=sigma_theta)

                p_obs *= p_d_m * p_phi_m

        p.setWeight(p_obs)

def resample_particles(particle_list, weights, w_fast, w_slow):
    cdf = np.cumsum(weights)
    resampled = []
    eps = 1e-12
    for _ in range(len(particle_list)):
        if random.random() < max(0.0, 1.0 - w_fast / max(w_slow, eps)):
            p = initialize_particles(1)[0]
            resampled.append(p)
        else:
            z = np.random.rand()
            idx = np.searchsorted(cdf, z)
            q = particle_list[idx]
            resampled.append(particle.Particle(q.getX(), q.getY(), q.getTheta(), 1.0/len(particle_list)))
    return resampled

def filter_landmarks_by_distance(objectIDs, dists, angles):
    # Keep the closest observation per ID
    min_dist = {}
    for iD, d, a in zip(objectIDs, dists, angles):
        if iD not in min_dist or d < min_dist[iD][0]:
            min_dist[iD] = (d, a)
    filtered_ids = list(min_dist.keys())
    filtered_dists = [min_dist[ID][0] for ID in filtered_ids]
    filtered_angles = [min_dist[ID][1] for ID in filtered_ids]
    return filtered_ids, filtered_dists, filtered_angles

# =========================== MAIN ===========================
try:
    if showGUI:
        WIN_RF1 = "Robot view"; cv2.namedWindow(WIN_RF1); cv2.moveWindow(WIN_RF1, 50, 50)
        WIN_World = "World view"; cv2.namedWindow(WIN_World); cv2.moveWindow(WIN_World, 500, 50)

    # Particles
    num_particles = 1000
    particles = initialize_particles(num_particles)
    est_pose = particle.estimate_pose(particles)
    print(f"estimated pose: {est_pose}")

    # Motion + noise
    distance = 0.0
    angle = 0.0
    sigma_d = 10.0      # cm
    sigma_theta = 0.2   # rad

    # Adaptive resampling
    w_slow = 0.0
    w_fast = 0.0
    alpha_slow = 0.05   # smoother than 1
    alpha_fast = 0.2

    # Robot
    if isRunningOnArlo():
        arlo = CalibratedRobot()

    # World view
    world = np.zeros((500, 500, 3), dtype=np.uint8)
    draw_world(est_pose, particles, world)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        cam = camera.Camera(1, robottype='arlo', useCaptureThread=False)
        pathing = LocalizationPathing(arlo, cam, landmarkIDs)
    else:
        cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=False)

    # ---------- Stabilizers & goal checks ----------
    stabilization_counter = 0
    STABILIZE_N = 3       # need BOTH markers for 3 consecutive frames
    goal_ok_counter = 0
    GOAL_OK_N = 3         # need goal condition true for 3 consecutive frames
    EQUAL_TOL = 8.0       # cm : |d6 - d7| <= this
    RADIAL_TOL = 8.0      # cm : |(d6 + d7)/2 - 150| <= this
    L_BASELINE = 300.0    # cm

    # Optional low-pass on distances (helps with jitter)
    d6_f = None; d7_f = None
    ALPHA = 0.4  # 0..1 (higher = more weight on current reading)

    while True:
        # Quit key (GUI only)
        action = cv2.waitKey(10)
        if action == ord('q'):
            break

        if not isRunningOnArlo():
            if action == ord('w'):
                distance = 10.0
            elif action == ord('x'):
                distance = -10.0
            elif action == ord('a'):
                angle = 0.2
            elif action == ord('d'):
                angle = -0.2
            else:
                distance = 0.0; angle = 0.0

        # ---- FETCH FRAME ----
        colour = cam.get_next_frame()

        # ---- DETECTIONS ----
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        both_seen = False
        d6 = d7 = None

        if not isinstance(objectIDs, type(None)):
            objectIDs, dists, angles = filter_landmarks_by_distance(objectIDs, dists, angles)
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])

            # Update stabilizer from CURRENT detections
            ids_now = list(objectIDs)
            both_seen = (6 in ids_now) and (7 in ids_now)
            if both_seen:
                # get the distances for 6 and 7 (cm from your camera module)
                i6 = ids_now.index(6); i7 = ids_now.index(7)
                d6 = float(dists[i6]); d7 = float(dists[i7])

                # low-pass (optional but helps)
                if d6_f is None: d6_f = d6
                if d7_f is None: d7_f = d7
                d6_f = (1-ALPHA)*d6_f + ALPHA*d6
                d7_f = (1-ALPHA)*d7_f + ALPHA*d7
                d6, d7 = d6_f, d7_f

                stabilization_counter = min(stabilization_counter + 1, STABILIZE_N)
            else:
                stabilization_counter = 0

            # PF measurement update
            measurement_model(particles, objectIDs, dists, angles, sigma_d, sigma_theta)
            weights = np.array([p.getWeight() for p in particles])
            w_avg = float(np.mean(weights))
            w_slow += alpha_slow * (w_avg - w_slow)
            w_fast += alpha_fast * (w_avg - w_fast)
            # normalize
            s = float(np.sum(weights)); s = s if s > 0 else 1.0
            weights /= s
            particles = resample_particles(particles, weights, w_fast, w_slow)

            cam.draw_aruco_objects(colour)
        else:
            stabilization_counter = 0
            # No observation: uniform weights
            for p in particles:
                p.setWeight(1.0/num_particles)

        # ---- CONTROL LOGIC ----
        if isRunningOnArlo():
            if stabilization_counter < STABILIZE_N:
                # Explore until we see both; when both seen, hold for a few frames
                if not pathing.seen_all_landmarks():
                    drive = (random.random() < (1/18))
                    distance, angle = pathing.explore_step(drive)
                else:
                    distance, angle = 0.0, 0.0
                    time.sleep(0.2)  # settle to avoid motion blur
            else:
                # Stable view of both: check if we are at center
                if both_seen and (d6 is not None) and (d7 is not None):
                    mean_d = 0.5*(d6 + d7)  # cm
                    equal_ok = abs(d6 - d7) <= EQUAL_TOL
                    radial_ok = abs(mean_d - (L_BASELINE/2.0)) <= RADIAL_TOL
                    if equal_ok and radial_ok:
                        goal_ok_counter = min(goal_ok_counter + 1, GOAL_OK_N)
                    else:
                        goal_ok_counter = 0
                else:
                    goal_ok_counter = 0

                if goal_ok_counter >= GOAL_OK_N:
                    print("reached center (stable).")
                    distance, angle = 0.0, 0.0
                    try:
                        arlo.stop()
                    except Exception:
                        pass
                    break
                else:
                    distance, angle = pathing.move_towards_goal_step(est_pose, center)

        # ---- PREDICT ----
        sample_motion_model(particles, distance, angle, sigma_d, sigma_theta)
        est_pose = particle.estimate_pose(particles)

        if showGUI:
            draw_world(est_pose, particles, world)
            cv2.imshow("Robot view", colour)
            cv2.imshow("World view", world)

finally:
    cv2.destroyAllWindows()
    try:
        cam.terminateCaptureThread()
    except Exception:
        pass
