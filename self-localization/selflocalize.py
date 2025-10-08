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
showGUI = False  # Whether or not to open GUI windows
onRobot = True # Whether or not we are running on the Arlo robot


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
    You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot

try:
    from RobotUtils.Robot import Robot
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False

# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
landmarkIDs = [6, 7]
landmarks = {
    6: (0.0, 0.0),  # Coordinates for landmark 1
    7: (300.0, 0.0)  # Coordinates for landmark 2
}

center = np.array([(landmarks[6][0] + landmarks[7][0]) / 2,
                   (landmarks[6][1] + landmarks[7][1]) / 2])



landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks

def jet(x):
    """Colour map for drawing particles. This function determines the colour of 
    a particle from its weight."""
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)

    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world):
    """Visualization.
    This functions draws robots position in the world coordinate system."""

    # Fix the origin of the coordinate system
    offsetX = 100
    offsetY = 250

    # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
    ymax = world.shape[0]

    world[:] = CWHITE # Clear background to white

    # Find largest weight
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

    # Draw particles
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x,y), 2, colour, 2)
        b = (int(particle.getX() + 15.0*np.cos(particle.getTheta()))+offsetX, 
                                    ymax - (int(particle.getY() + 15.0*np.sin(particle.getTheta()))+offsetY))
        cv2.line(world, (x,y), b, colour, 2)

    # Draw landmarks
    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    # Draw estimated robot pose
    a = (int(est_pose.getX())+offsetX, ymax-(int(est_pose.getY())+offsetY))
    b = (int(est_pose.getX() + 15.0*np.cos(est_pose.getTheta()))+offsetX, 
        ymax-(int(est_pose.getY() + 15.0*np.sin(est_pose.getTheta()))+offsetY))
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)



def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points. 
        p = particle.Particle(600.0*np.random.ranf() - 100.0, 600.0*np.random.ranf() - 250.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles

def sample_motion_model(particles_list, distance, angle, sigma_d, sigma_theta):
    for p in particles_list:
        delta_x = distance * np.cos(p.getTheta() + angle)
        delta_y = distance * np.sin(p.getTheta() + angle)
    
        particle.move_particle(p, delta_x, delta_y, angle)

    particle.add_uncertainty_von_mises(particles_list, sigma_d, sigma_theta)

def motion_model_with_map(particle, distance, angle, sigma_d, sigma_theta, grid):
    indices, valid = grid.world_to_grid([particle.getX(), particle.getY()])

    p_map = 0 if (not valid or grid.in_collision(indices)) else 1
    if p_map:
        sample_motion_model([particle], distance, angle, sigma_d, sigma_theta)
        return 1
    
    return 0

def sample_motion_model_with_map(particles_list, distance, angle, sigma_d, sigma_theta, grid, max_tries=10):
    for p in particles_list:
        for attempt in range(max_tries):
            pi = motion_model_with_map(p, distance, angle, sigma_d, sigma_theta, grid)
            if pi > 0:
                break
        else:
            # fallback
            p.setWeight(0.01)  # low weight to indicate invalid


def measurement_model(particle_list, landmarkIDs, dists, angles, sigma_d, sigma_theta):
    for particle in particle_list:
        x_i = particle.getX()
        y_i = particle.getY()
        theta_i = particle.getTheta()

        p_observation_given_x = 1.0

        #p(z|x) = sum over the probability for all landmarks
        for landmarkID, dist, angle in zip(landmarkIDs, dists, angles):
            if landmarkID in landmarkIDs:
                l_x, l_y = landmarks[landmarkID]
                d_i = np.sqrt((l_x - x_i)**2 + (l_y - y_i)**2)

                p_d_m = norm.pdf(dist, loc=d_i, scale=sigma_d)

                e_theta = np.array([np.cos(theta_i), np.sin(theta_i)])
                e_theta_hat = np.array([-np.sin(theta_i), np.cos(theta_i)])

                e_l = np.array([l_x - x_i, l_y - y_i]) / d_i

                phi_i = np.sign(np.dot(e_l, e_theta_hat)) * np.arccos(np.dot(e_l, e_theta))
                
                p_phi_m = norm.pdf(angle,loc=phi_i, scale=sigma_theta)


                p_observation_given_x *= p_d_m* p_phi_m

        particle.setWeight(p_observation_given_x)

def resample_particles(particle_list, weights, w_fast, w_slow):
    cdf = np.cumsum(weights)

    resampled = []
    for _ in range(len(particle_list)):
        if random.random() < max(0.0, 1.0 - w_fast /w_slow):
            p = initialize_particles(1)[0]
            resampled.append(p)
        else:
            z = np.random.rand()
            idx = np.searchsorted(cdf, z)
            p_resampled = particle.Particle(particle_list[idx].getX(), particle_list[idx].getY(), particle_list[idx].getTheta(), 1.0/(len(particle_list)))
            resampled.append(p_resampled)

    return resampled


def filter_landmarks_by_distance(objectIDs, dists, angles):
    """
    Keep only the measurement at the smallest distance for each landmark ID.
    """
    min_dist_dict = {}  # dict: landmarkID -> (dist, angle)

    for id, d, a in zip(objectIDs, dists, angles):
        if id not in min_dist_dict or d < min_dist_dict[id][0]:
            min_dist_dict[id] = (d, a)

    filtered_ids = list(min_dist_dict.keys())
    filtered_dists = [min_dist_dict[ID][0] for ID in filtered_ids]
    filtered_angles = [min_dist_dict[ID][1] for ID in filtered_ids]

    return filtered_ids, filtered_dists, filtered_angles

# Main program #
try:
    if showGUI:
        # Open windows
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)

        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)


    # Initialize particles
    num_particles = 1000
    particles = initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose
    print(f"estimated pose: {est_pose}")

    # Driving parameters
    distance = 0.0 # distance driven at this time step
    angle = 0.0 # angle turned at this timestep

    sigma_d = 10
    sigma_theta = 0.2

    w_slow = 0.0
    w_fast = 0.0
    alpha_slow = 1
    alpha_fast = 1

    #Initialize the robot
    if isRunningOnArlo():
        arlo = CalibratedRobot()

    #grid = LandmarkOccupancyGrid([-100, -250], (400, 250))
    #landmark_list = [(x, y, 20.0) for x, y in landmarks.values()]
    #grid.add_landmarks(landmark_list)


    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8)

    # Draw map
    draw_world(est_pose, particles, world)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        #cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
        cam = camera.Camera(1, robottype='arlo', useCaptureThread=False)
        pathing = LocalizationPathing(arlo, cam, landmarkIDs)
    else:
        #cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
        cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=False)

    while True:
        # Move the robot according to user input (only for testing)
        action = cv2.waitKey(10)
        if action == ord('q'): # Quit
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
                # stop if no key pressed
                distance = 0
                angle = 0

        # Use motor controls to update particles
        if isRunningOnArlo():
            if not pathing.seen_all_landmarks():
                drive = random.random() < (1/18)
                distance, angle = pathing.explore_step(drive)
            else:
                if stabilization_counter < 2:
                    stabilization_counter += 1
                    distance, angle = 0, 0
                else:
                    distance, angle = pathing.move_towards_goal_step(est_pose, center)
                    
        sample_motion_model(particles, distance, angle, sigma_d, sigma_theta)
        # Fetch next frame
        colour = cam.get_next_frame()
        
        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            objectIDs, dists, angles = filter_landmarks_by_distance(objectIDs, dists, angles)
            # List detected objects
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])

            # Compute particle weights
            measurement_model(particles, objectIDs, dists, angles, sigma_d, sigma_theta)

            weights = np.array([p.getWeight() for p in particles])

            w_avg = np.mean(weights)
            w_slow += alpha_slow * (w_avg - w_slow)
            w_fast += alpha_fast * (w_avg - w_fast)

            weights /= np.sum(weights)

            # Resampling
            particles = resample_particles(particles, weights, w_fast, w_slow)

            # Draw detected objects
            cam.draw_aruco_objects(colour)
            
        else:
            # No observation - reset weights to uniform distribution
            for p in particles:
                p.setWeight(1.0/num_particles)

    
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

        if showGUI:
            # Draw map
            draw_world(est_pose, particles, world)
    
            # Show frame
            cv2.imshow(WIN_RF1, colour)

            # Show world
            cv2.imshow(WIN_World, world)
    

finally: 
    # Make sure to clean up even if an exception occurred
    
    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()

