#include <ros/ros.h>
#include <sensor_msgs/MultiEchoLaserScan.h>
#include <sensor_msgs/LaserScan.h>
#include "tf/transform_broadcaster.h"
// #include <Eigen/Eigen>


ros::Publisher pub_laserscan;
// convert multiecho to laserscan
void multiecho2laserscan(const sensor_msgs::MultiEchoLaserScanConstPtr &msg)
{
    //publish laserscan
    sensor_msgs::LaserScan laserscan;
    laserscan.header.stamp = msg->header.stamp;
    laserscan.header.frame_id = "robot";
    laserscan.angle_min = msg->angle_min;
    laserscan.angle_max = msg->angle_max;
    laserscan.angle_increment = msg->angle_increment;
    laserscan.time_increment = msg->time_increment;
    laserscan.scan_time = msg->scan_time;
    laserscan.range_min = msg->range_min;
    laserscan.range_max = msg->range_max;
    laserscan.ranges.resize(msg->ranges.size());
    laserscan.intensities.resize(msg->ranges.size());
    for (auto i = 0; i < msg->ranges.size(); i++)
    {
        laserscan.ranges[i] = msg->ranges[i].echoes[0]; // only take the first echo
        laserscan.intensities[i] = msg->intensities[i].echoes[0]; // only take the first echo
        if(laserscan.ranges[i]> msg->range_max || laserscan.ranges[i] < msg->range_min){
            laserscan.ranges[i] = 0;
        }
    }
    pub_laserscan.publish(laserscan);
}

void multiecho_laserscan_callback(const sensor_msgs::MultiEchoLaserScanConstPtr &msg){
    multiecho2laserscan(msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "visualize_laserScan");
    ros::NodeHandle nh;
    pub_laserscan = nh.advertise<sensor_msgs::LaserScan>("/laserscan", 100);
    ros::Subscriber multiecho_laserscan = nh.subscribe<sensor_msgs::MultiEchoLaserScan>("/multiecho_scan", 100, multiecho_laserscan_callback);
    ros::spin();
    return 0;
}