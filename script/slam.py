import os, sys, pickle, math
from copy import deepcopy
import cv2
from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *
from scipy.special import logsumexp

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1)) # cell number in x direction
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1)) # cell number in y direction

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX
        # use the upper left corner as the grid cell coordinate frame origin
        # x axis points towards right, y axis points downwards
        cell_indices = np.zeros((2,len(x)),dtype=int)
        # x_cliped = np.clip(x, s.xmin, s.xmax)
        # y_cliped = np.clip(y, s.ymin, s.ymax)
        cell_index_x = np.ceil((x - s.xmin) / s.resolution).astype(np.int)
        cell_index_y = np.ceil((s.ymax - y) / s.resolution).astype(np.int)
        # cell_index_x = ((x + s.szx/2)*s.resolution).astype(np.int)
        # cell_index_y = ((-y+s.szy/2)*s.resolution).astype(np.int)
        cell_indices[0] = np.clip(cell_index_x, 0, s.szx-1)
        cell_indices[1] = np.clip(cell_index_y, 0, s.szy-1)
        return cell_indices

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### TODO: XXXXXXXXXXX

        nums = p.shape[1]
        # normalize weight and get cum sum
        weight_sum = np.cumsum(w)
        weight_sum /= weight_sum[-1]

        # Generate N ordered random numbers
        random = (np.linspace(0, nums-1, nums) + np.random.uniform(size=nums))/nums

        # multinomial distribution
        new_sample = np.zeros(p.shape)
        sample = 0
        index = 0
        while(sample<nums):
            while (weight_sum[index]<random[sample]):
                index += 1
            new_sample[:,sample] = p[:,index]
            sample += 1
        new_p = new_sample

        new_w = np.full(nums,1/nums)
        return new_p, new_w
        

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        INPUT:
            p: (3,), [x,y,yaw]
            d: (n, ), distance of ray
            head_angle: float, the angle of head in body frame
            neck_angle: float, the angle of neck in body frame
            angles: (n, ), s.lidar_angles from -135 to 135
        OUTPUT:
            endpoints: (2,n) ndarray
        """
        #### TODO: XXXXXXXXXXX
        num_readings = len(d)
        x_world = p[0]
        y_world = p[1]
        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data
        # 1. from lidar distances to points in the LiDAR frame
        d = np.clip(d , s.lidar_dmin, s.lidar_dmax)
        x_lidar = d * np.cos(angles)
        y_lidar = d * np.sin(angles)
        coord_lidar = np.vstack((x_lidar, y_lidar,np.zeros(num_readings)))
        coord_lidar_4d = make_homogeneous_coords_3d(coord_lidar) # homogenous 4d coordinate [x,y,0,1]
        
        # 2. from lidar frame to body frame (the frame setup is a little strange here as the body frame
        # is on the top of the robot head)
        T_lidar2Body = np.array([0, 0, s.lidar_height])
        H_lidar2body = euler_to_se3(r = 0, p = head_angle, y = neck_angle, v = T_lidar2Body)
        # H_lidar2body = euler_to_se3(r = 0, p = 0, y = 0, v = T_lidar2Body)

        # 3. from body frame to world frame
        T_body2world = np.array([x_world, y_world, s.head_height])
        # H_body2world = euler_to_se3(r = 0, p = head_angle, y = neck_angle, v = T_body2world)
        H_body2world = euler_to_se3(r = 0, p = 0, y = p[2], v = T_body2world)

        # 4. from lidar to world frame
        H_lidar2world = H_body2world @ H_lidar2body
        coord_world_4d = H_lidar2world @ coord_lidar_4d # (4,4) @ (4, n)
        # coord_world_4d_homo = coord_world_4d / coord_world_4d[-1] # homogenous world coordinate

        # 5. obstacle coordinate
        xy_obstacle = coord_world_4d[0:2]

        # eliminate measurements that hit the floor
        not_floor = coord_world_4d[2] > 0.1 # (s.head_height + s.lidar_height)
        xy_obstacle_calibrated = xy_obstacle[:, not_floor] 
        
        return xy_obstacle_calibrated

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        INPUT:
            t: timestamp
        """
        if t == 0:
            return np.zeros(3)
        #### TODO: XXXXXXXXXXX
        state = s.lidar[t]['xyth']
        state_last = s.lidar[t-1]['xyth']
        return smart_minus_2d(state,state_last)

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        #### TODO: XXXXXXXXXXX
        # particles = s.p # current particles (3,n)
        control = s.get_control(t) # the control used from t-1 to t
        state_noise = np.random.multivariate_normal([0, 0, 0], s.Q, size = s.n).T # (3, n)
        
        # compute the updated x_k+1_k particles
        updated_particles = np.zeros((3,s.n))
        for i in range(s.n):
            updated_particles[:,i] = smart_plus_2d(smart_plus_2d(s.p[:,i],control), state_noise[:,i])
            # updated_particles[:,i] = smart_plus_2d(smart_plus_2d(s.p[:,i],control), state_noise)
        s.p = updated_particles
        

    @staticmethod
    def update_weights(s,w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        log_weights = np.log(w) + obs_logp
        log_weights -= s.log_sum_exp(log_weights)
        s.w = np.exp(log_weights)
        # new_w = np.exp(obs_logp) * w / (np.sum(np.exp(obs_logp) * w))
        return s.w
        

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        #### TODO: XXXXXXXXXXX
        # First find the head, neck angle at t (this is the same for every particle)
        t_joint = (s.find_joint_t_idx_from_lidar(s.lidar[t]['t'])).item()
        # t_joint = s.find_joint_t_idx_from_lidar(t)
        head_angle = s.joint['head_angles'][1][t_joint]
        neck_angle = s.joint['head_angles'][0][t_joint]

        d = s.lidar[t]['scan']
        updated_p = s.p
        obs_logp = np.zeros(s.n)

        # update map
        # particle_toUpdate_map = updated_p[:,np.argmax(s.w).item()]
        particle_toUpdate_map = s.lidar[t]['xyth']
        s.update_map(particle_toUpdate_map,d, head_angle, neck_angle)

        if t != 0:
            for i in range(s.n):
                # Project lidar scan into the world frame (different for different particles)
                xy_obstacle = s.rays2world(updated_p[:,i], d, head_angle, neck_angle, s.lidar_angles)

                # Calculate which cells are obstacles according to this particle for this scan,
                obstacle_indices = s.map.grid_cell_from_xy(xy_obstacle[0],xy_obstacle[1]) # the indices of cells which are observed to have obstacles

                # calculate the observation log-probability
                obs_logp[i] = np.sum(s.map.log_odds[obstacle_indices[1]][obstacle_indices[0]] > s.map.log_odds_thresh) 
            
            obs_logp = obs_logp/10
            # update particle weight to get w_k+1_k+1
            s.w = s.update_weights(s, s.w, obs_logp) 


    def update_map(s, p, d, head_angle, neck_angle):
        # Find the particle with the largest weight
        # largest_weight_particle_index = np.argmax(new_w).item()
        particle_toUpdate_map = p 

        # use its occupied cells to update the map.log_odds and map.cells.
        xy_obstacle_largest_weight = s.rays2world(particle_toUpdate_map, d, head_angle, neck_angle, s.lidar_angles)
        obstacle_indices_largest_weight = s.map.grid_cell_from_xy(xy_obstacle_largest_weight[0],xy_obstacle_largest_weight[1])

        # update the delta log odds of obstacles
        delta_odds = np.zeros(s.map.cells.shape)
        delta_odds[obstacle_indices_largest_weight[1], obstacle_indices_largest_weight[0]] = s.lidar_log_odds_occ - s.lidar_log_odds_free
        
        # update the odds of observed free cells
        # update delta log odds for free grid, using contours to mask region between pose and hit
        mask = np.zeros(s.map.cells.shape)
        largest_weight_particle_cell_postion = s.map.grid_cell_from_xy(particle_toUpdate_map.reshape(-1,1)[0],particle_toUpdate_map.reshape(-1,1)[1])
        contour = np.hstack((largest_weight_particle_cell_postion, obstacle_indices_largest_weight))
        cv2.drawContours(image=mask, contours = [contour.T], contourIdx = -1, color = s.lidar_log_odds_free, thickness=-1)
        delta_odds += mask

        # update the odds, cells with obstacles will increase its log_odds, otherwise decrease its log_odds
        s.map.log_odds = np.clip(a = s.map.log_odds + delta_odds, a_min = -s.map.log_odds_max, a_max = s.map.log_odds_max)
        s.map.cells = (s.map.log_odds >= s.map.log_odds_thresh)


    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
