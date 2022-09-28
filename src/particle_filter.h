# ifndef PF_H
# define PF_H

# include <iostream>
# include <vector>
# include <math.h>
# include <Eigen/Dense>
# include <sensor_msgs/MultiEchoLaserScan.h>
# include <Eigen/Geometry>
# include <geometry_msgs/PoseStamped.h>
# include <cstdlib>
# include <tf/tf.h>
# include <eigen_conversions/eigen_msg.h>
// # include <tf_conversions/tf_eigen.h>
using namespace std;
using namespace Eigen;


class mapt{
    private:
        float resolution;
        int szx;
        int szy;
        // binarized map and log-odds
        
        int xmin = -20;
        int xmax = 20;
        int ymin = -20;
        int ymax = 20;
        const double log_odds_max = 5e6;
        
        
    public:
        vector<vector<bool>> cells;
        vector<vector<float>> log_odds;
        const float occupied_prob_thresh = 0.6;
        const float log_odd_thresh = std::log(occupied_prob_thresh/ (1- occupied_prob_thresh));
        
        mapt(float resolution_){
            resolution = resolution_;
            szx = int((xmax-xmin)/resolution);
            szy = int((ymax-ymin)/resolution);
            cells = vector<vector<bool>> (szy,vector<bool>(szx,false));
            log_odds = vector<vector<float>> (szy,vector<float>(szx, 0));
        }

        std::vector<int> grid_cell_from_xy(float x, float y){
            int x_cell_idx = floor(x/resolution);
            int y_cell_idx = floor(y/resolution);
            if(x_cell_idx >= szx){
                x_cell_idx = szx - 1;
            }
            if(y_cell_idx >= szx){
                y_cell_idx = szy - 1;
            }
            vector<int> pixel_coord = {x_cell_idx, y_cell_idx};
            return pixel_coord;
        }
};


class particle_filter{
    private:
        float resolution_;
        float resampling_threshold_ = 0.3;
        Eigen::Matrix<float, 3, 3> Q_; // dynamic noise for state (x,y,yaw)
        const float lidar_log_odds_occ_ = std::log(9);
        const float lidar_log_odds_free_ = std::log(1/9.);
        const int n_ = 100; // number of particles
        vector<Eigen::Affine3f> particles_;
        vector<float> weights_;
        Eigen::Affine3f last_pose_;
        vector<float> scan_x_; // x coordinate in meters in robot frame
        vector<float> scan_y_; // y coordiante in meters in robot frame
    
    public:
        particle_filter(){
        }

        void readin_scan_data(const sensor_msgs::MultiEchoLaserScanConstPtr &msg)
        {   
            // clear last scan data
            scan_x_.clear();
            scan_y_.clear();
            for (auto i = 0; i < msg->ranges.size(); i++)
            {
                float dist = msg->ranges[i].echoes[0]; //only first echo used for particle_filter2d
                float theta = msg->angle_min + i * msg->angle_increment;
                scan_x_.push_back(dist * cos(theta));
                scan_y_.push_back(dist * sin(theta));
            }
        }

        void init_particles_and_weights(){
            
        }

        void init_pose(){
            last_pose_ = Eigen::Matrix3f::Identity();
        }

        void dynamic_step(const geometry_msgs::PoseStamped::ConstPtr& pose_msg){
            // compute the difference betweeen two poses from odometry
            // convert ros msg pose to eigen
            Eigen::Affine3d pose_eig = Eigen::Affine3d::Identity();
            if (pose_msg){
                tf::poseMsgToEigen(pose_msg->pose, pose_eig);
            }
            Eigen::Affine3f delta_pose = last_pose_.inverse() * pose_eig.cast<float>();
            last_pose_ = pose_eig.cast<float>();
            // use the difference between two poses to update the state
            for(int i = 0; i < n_; i++){
                particles_[i] = delta_pose * particles_[i];
            }
        }

        void observation_step(){
            vector<float> observ_log_prob;
            for(int i = 0; i<n_; ++i){
                int actual_num_obstacle = 0;
                for(int j = 0; j<scan_x_.size(); ++j){
                    // transform the scan using current particle state into map frame 
                    Eigen::Vector3f scan_coord(scan_x_[i], scan_y_[i], 1);
                    // get the scan coordinate in map frame
                    Eigen::Vector3f coord_map = particles_[i]* scan_coord;
                    // convert scan in meters to scan in pixels
                    vector<int>  coord_pixel = map_.grid_cell_from_xy(coord_map(0), coord_map(1));
                    // count the number of cells in the scan that are acutally obstacles in the map 
                    if(map_.cells[coord_pixel[1]][coord_pixel[0]] == true) actual_num_obstacle++;
                }
                // compute the the observation log_probability
                observ_log_prob.push_back(std::log(actual_num_obstacle));
            }
            update_weights(observ_log_prob);
        }

        void update_weights(vector<float> &observ_log_prob){
            for(int i =0; i<n_; i++){
                float log_weight =std::log(weights_[i]) + observ_log_prob[i];
                weights_[i] = std::(log_weight);
            }
        }

        void update_map(){
            
            // use the particle with the largest weight to update the map
            // input: state state[0], state[1], state[2], float[]
            //     LaserScan, &sensor_msg::LaserScan
            
            int maxWeightIndex = std::max_element(weights_.begin(),weights_.end()) - weights_.begin();
            Eigen::Affine3f maxWeightPose = particles_[maxWeightIndex];

            for(int i  = 0; i<scan_x_.size(); ++i){
                Eigen::Vector3f scan_coord(scan_x_[i], scan_y_[i], 1);
                // get the scan coordinate in map frame
                Eigen::Vector3f coord_map = maxWeightPose * scan_coord;
                // convert scan in meters to scan in pixels
                vector<int>  coord_pixel = map_.grid_cell_from_xy(coord_map(0), coord_map(1));
                // get the obstacle coordiante
                // get the free space coordinate
                // increase the odds of obstacles
                // decrease the odds of free space 
                int row_idx = coord_pixel[1];
                int col_idx = coord_pixel[0];
                map_.log_odds[row_idx][col_idx] += lidar_log_odds_occ_;
                if(map_.log_odds[row_idx][col_idx] > map_.log_odd_thresh)
                    map_.cells[row_idx][col_idx] = true;
            }
        }

        void stratified_resampling(){
            // compute the sum of weights
            float sum_weights = 0;
            for(int i = 0; i<n_; i++){
                sum_weights += weights_[i];
            }
            // normalize the weights
            vector<float> normalized_weights;
            for(int i = 0; i<n_; i++){
                normalized_weights.push_back(weights_[i]/sum_weights);
            }
            // generate random numbers


        }

        void resample_particels(){
            // compute the the sum of squares of weights
            float sum_weight_squares = 0;
            for(int i = 0; i<n_; i++){
                sum_weight_squares += weights_[i]*weights_[i];
            } 
            float e = 1/sum_weight_squares;
            ROS_INFO_STREAM("effective number of particles: %f", e);
            if(e/n_ < resampling_threshold_){
                ROS_INFO_STREAM("Resampling.....");
                stratified_resampling();
            }
            
        }
};



#endif