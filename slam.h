# pragma once

# include <iostream>
# include <vector>
# include <math.h>
# include <Eigen/Dense>

class map{
    private:
        float resolution;
        int xmin = -20;
        int xmax = 20;
        int ymin = -20;
        int ymax = 20;
        int szx;
        int szy;
        const double log_odds_max = 5e6;
        const float occupied_prob_thresh = 0.6;
        const float log_odd_thresh = std::log(occupied_prob_thresh/ (1- occupied_prob_thresh));
        // binarized map and log-odds
        std::vector<std::vector<bool>> cells;
        std::vector<std::vector<bool>> log_odds;
    public:
        map(){}
        map(double resolution_){}
        std::pair<int, int> grid_cell_from_xy(double x, double y){}
};

class slam{
    private:
        float resolution;
        grid_map = map(resolution);
        float resampling_threshold;
        Eigen::Matrix<float, 3, 3> Q; // dynamic noise for state (x,y,yaw)
        
        const float head_height = 0.93 + 0.33;
        const float lidar_height = 0.15;
        const float lidar_dmin = 1e-3;
        const float lidar_dmax = 30;
        const float lidar_angular_resolution = 0.25;
        std::vector<float> lidar_angles;
        for(float angle = -135; angle<135+lidar_angular_resolution; angle += lidar_angular_resolution){
            lidar_angles.push_back(angle);
        }
        // sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        // for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        // log_odds for free cells (which are all cells that are not occupied)
        const float lidar_log_odds_occ = std::log(9)
        const float lidar_log_odds_free = std::log(1/9.)
        const int n = 100; // number of particles
        typedef Eigen::Matrix<float, 3, n> particles;
        typedef Eigen::Vector1d<float> weights;
    
    public:
        std::pair<particles, weights> stratified_resampling(particles &p, weights &w){

        }

        weights log_sum_exp(weights &w){

        }

        Eigen::Matrix<float, 2, n> ray2world(){

        }

        void dynamic_step(float t){

        }

        void update_weights(weights &w, float obs_logp){

        }

        void observation_step(float t){

        }

        void update_map(particle &p, float head_angle, float neck_angle){

        }

        void resample_particle(){

        }

}