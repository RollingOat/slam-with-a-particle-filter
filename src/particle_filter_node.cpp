// particle_filter_node: subscribe to poses from laser odometry, publish optimized poses and map
# include <ros/ros.h>

ros::Publisher map_pub;
ros::Publisher filtered_pose_pub;

int main(int argc, char **argv){
    // initialize publisher and subscriber
    ros::init(argc,argv,"particle_filter");
    ros::NodeHandle nh;

    map_pub = nh.advertise<>("output_map", 10)
    filtered_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("filtered_pose", 10);

    ros::Subscriber odom_pose_sub = nh.subscribe<geometry_msgs::>

    return 0;
}