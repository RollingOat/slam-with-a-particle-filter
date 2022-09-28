import click, tqdm, random
# from cv2 import FILE_STORAGE_FORMAT_MASK
import cv2
import os
import time
import matplotlib.pyplot as plt

from slam import *

def run_dynamics_step(src_dir, log_dir, idx, split, t0=0, draw_fig=False):
    """
    This function is for you to test your dynamics update step. It will create
    two figures after you run it. The first one is the robot location trajectory
    using odometry information obtained form the lidar. The second is the trajectory
    using the PF with a very small dynamics noise. The two figures should look similar.
    """
    slam = slam_t(Q=1e-8*np.eye(3))
    slam.read_data(src_dir, idx, split)

    # trajectory using odometry (xy and yaw) in the lidar data
    d = slam.lidar
    xyth = []
    for p in d:
        xyth.append([p['xyth'][0], p['xyth'][1],p['xyth'][2]])
    xyth = np.array(xyth)

    plt.figure(1); plt.clf();
    plt.title('Trajectory using onboard odometry')
    plt.plot(xyth[:,0], xyth[:,1])
    logging.info('> Saving odometry plot in '+os.path.join(log_dir, 'odometry_%s_%02d.png'%(split, idx)))
    plt.savefig(os.path.join(log_dir, 'odometry_%s_%02d.png'%(split, idx)))

    # dynamics propagation using particle filter
    # n: number of particles, w: weights, p: particles (3 dimensions, n particles)
    # S covariance of the xyth location
    # particles are initialized at the first xyth given by the lidar
    # for checking in this function
    n = 3
    w = np.ones(n)/float(n)
    p = np.zeros((3,n), dtype=np.float64)
    slam.init_particles(n,p,w)
    slam.p[:,0] = deepcopy(slam.lidar[0]['xyth'])

    print('> Running prediction')
    t0 = 0
    T = len(d)
    ps = deepcopy(slam.p)   # maintains all particles across all time steps
    plt.figure(2); plt.clf();
    ax = plt.subplot(111)
    for t in tqdm.tqdm(range(t0+1,T)):
        slam.dynamics_step(t)
        ps = np.hstack((ps, slam.p))

        if draw_fig:
            ax.clear()
            ax.plot(slam.p[0], slam.p[0], '*r')
            plt.title('Particles %03d'%t)
            plt.draw()
            plt.pause(0.01)
            
    plt.plot(ps[0], ps[1], '*c')
    plt.title('Trajectory using PF')
    logging.info('> Saving plot in '+os.path.join(log_dir, 'dynamics_only_%s_%02d.png'%(split, idx)))
    plt.savefig(os.path.join(log_dir, 'dynamics_only_%s_%02d.png'%(split, idx)))

def run_observation_step(src_dir, log_dir, idx, split, is_online=False):
    """
    This function is for you to debug your observation update step
    It will create three particles np.array([[0.2, 2, 3],[0.4, 2, 5],[0.1, 2.7, 4]])
    * Note that the particle array has the shape 3 x num_particles so
    the first particle is at [x=0.2, y=0.4, z=0.1]
    This function will build the first map and update the 3 particles for one time step.
    After running this function, you should get that the weight of the second particle is the largest since it is the closest to the origin [0, 0, 0]
    """
    slam = slam_t(resolution=0.05)
    slam.read_data(src_dir, idx, split)

    # t=0 sets up the map using the yaw of the lidar, do not use yaw for
    # other timestep
    # initialize the particles at the location of the lidar so that we have some
    # occupied cells in the map to calculate the observation update in the next step
    t0 = 0
    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[t0]['rpy'][2]
    # logging.debug('> Initializing 1 particle at: {}'.format(xyth))
    print('Initializing 1 particle at: {}'.format(xyth))
    slam.init_particles(n=1,p=xyth.reshape((3,1)),w=np.array([1]))

    slam.observation_step(t=0)
    print('> Particles\n: {}'.format(slam.p))
    print('> Weights: {}'.format(slam.w))

    # reinitialize particles, this is the real test
    logging.info('\n')
    n = 3
    w = np.ones(n)/float(n)
    p = np.array([[2, 0.2, 3],[2, 0.4, 5],[2.7, 0.1, 4]])
    slam.init_particles(n, p, w)

    slam.observation_step(t=1)
    print('> Particles\n: {}'.format(slam.p))
    print('> Weights: {}'.format(slam.w))

def run_slam(src_dir, log_dir, idx, split):
    """
    This function runs slam. We will initialize the slam just like the observation_step
    before taking dynamics and observation updates one by one. You should initialize
    the slam with n=100 particles, you will also have to change the dynamics noise to
    be something larger than the very small value we picked in run_dynamics_step function
    above.
    OUTPUT:
        particle_xy: (2,T)
        odometry_xy: (2,T)
        cell: (num_cells, num_cells)
    """
    slam = slam_t(resolution=0.05, Q=np.diag([1e-4,1e-4,1e-4]))
    slam.read_data(src_dir, idx, split)
    

    T = len(slam.lidar)
    # again initialize the map to enable calculation of the observation logp in
    # future steps, this time we want to be more careful and initialize with the
    # correct lidar scan. First find the time t0 around which we have both LiDAR
    # data and joint data
    #### TODO: XXXXXXXXXXX
    t0 = 0
    # initialize the occupancy grid using one particle and calling the observation_step
    # function
    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[t0]['rpy'][2]
    slam.init_particles(n=1,p=xyth.reshape((3,1)),w=np.array([1]))
    slam.observation_step(t0)
    # slam, save data to be plotted later
    #### TODO: XXXXXXXXXXX
    slam.init_particles(n=100,p=None,w=None)
    interval = 50
    particle_x = []
    particle_y = []
    odometry_y = []
    odometry_x = []
    for t in tqdm.tqdm(range(1,T,interval)):
        # print('t:',t)
        # dynamic_time = time.time()
        slam.dynamics_step(t)
        # print('dynamic excution time', time.time() - dynamic_time)
        # observation_time = time.time()
        slam.observation_step(t)
        # print('largest weight:',max(slam.w))
        # print('observation excution time', time.time() - observation_time)
        particle_x.append(slam.p[:,np.argmax(slam.w).item()][0])
        particle_y.append(slam.p[:,np.argmax(slam.w).item()][1])
        odometry_x.append(slam.lidar[t]['xyth'][0])
        odometry_y.append(slam.lidar[t]['xyth'][1])
        # resampling_time = time.time()
        slam.resample_particles()
        # print('resampling excution time', time.time() - resampling_time)
    return [particle_x,particle_y], [odometry_x, odometry_y], slam

def plot(particle_xy, odometry_xy, slam):
    '''
    plots of the final binarized version of the map, the (x, y)
    location of the particle in the particle filter with the largest weight at each time-step
    and the odometry trajectory (x, y) (in a different color)
    INPUT:
        particle_xy: (2,T)
        slam: the final version of slam object
    '''
    # occ_mask = np.logical_and(np.exp(slam.map.log_odds) > 0.6, np.exp(slam.map.log_odds) < 1)
    # free_mask = np.exp(slam.map.log_odds) < 0.1
    # nomeasurement_mask = slam.map.log_odds == 0

    # occ_thres = 0.9
    # free_thres = 0.2
    # occ_mask = slam.map.log_odds > np.log(occ_thres / (1 - occ_thres))
    # free_mask = np.exp(slam.map.log_odds) < np.log(free_thres/(1-free_thres))
    # nomeasurement_mask = slam.map.log_odds == 0

    occ_mask = slam.map.cells > 0
    free_mask = slam.map.cells == 0
    nomeasurement_mask = slam.map.log_odds == 0

    map_plot = np.zeros((slam.map.szx, slam.map.szy, 3),np.uint8)
    map_plot[occ_mask] = [0,0,0] # black for occ
    map_plot[free_mask] = [255,255,255] # white for free
    map_plot[nomeasurement_mask] = [128,128,128] # gray for und

    # paint filter_trajectory
    size = len(odometry_xy[0])
    meter2cells_filter = slam.map.grid_cell_from_xy(np.array(odometry_xy[0])+np.random.random(size), np.array(odometry_xy[1]))
    map_plot[meter2cells_filter[1],meter2cells_filter[0]] = [255,0,0] # blue for trajectory

    # paint odometry_trajectory
    meter2cells_lidar = slam.map.grid_cell_from_xy(np.array(odometry_xy[0]), np.array(odometry_xy[1]))
    map_plot[meter2cells_lidar[1], meter2cells_lidar[0]] = [0, 0, 255]  # red for lidar

    # cv2.imshow('SLAM', map_plot)
    # cv2.waitKey(10)
    cv2.imwrite('SLAM.png', map_plot)

def plot_obs(slam):
    occ_mask = slam.map.cells > 0
    map_plot = np.zeros((slam.map.szx, slam.map.szy, 3),np.uint8)
    map_plot[occ_mask] = [255,255,255] # black for occ

    cv2.imshow('obs',map_plot)
    cv2.waitKey(0)

def plot_particle(particle_xy):
    plt.plot(particle_xy[0],particle_xy[1])

def plot_odometry(odometry_xy):
    plt.plot(odometry_xy[0],odometry_xy[1])



@click.command()
@click.option('--src_dir', default='./', help='data directory', type=str)
@click.option('--log_dir', default='logs', help='directory to save logs', type=str)
@click.option('--idx', default='0', help='dataset number', type=int)
@click.option('--split', default='train', help='train/test split', type=str)
@click.option('--mode', default='slam',
              help='choices: dynamics OR observation OR slam', type=str)


def main(src_dir, log_dir, idx, split, mode):
    # Run python main.py --help to see how to provide command line arguments

    if not mode in ['slam', 'dynamics', 'observation']:
        raise ValueError('Unknown argument --mode %s'%mode)
        sys.exit(1)

    np.random.seed(42)
    random.seed(42)

    if mode == 'dynamics':
        run_dynamics_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    elif mode == 'observation':
        run_observation_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    else:
        particle_xy, slam = run_slam(src_dir, log_dir, idx, split)
        plot(particle_xy, slam)
        return particle_xy

if __name__=='__main__':
    main()
