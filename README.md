# Slam with a particle filter
## Result
![image](https://user-images.githubusercontent.com/97129990/206576642-4d0885e0-7e23-41d9-ae05-68845e3bcf4a.png)

## C++ version

### Some info:
1. Reference: https://homes.cs.washington.edu/~todorov/courses/cseP590/16_ParticleFilter.pdf

2. Dataset: 2D lidar(35hz), imu(200hz)

3. Motion Model: 

4. Observation Model:

### Theory:

### TODO:
- [X] visualize the rosbag in robot body frame
- [X] implement the particle fitler - motion model
- [ ] implement the observation model
- [ ] implement the resampling step
- [X] implement the map update step
- [ ] write launch file: launch slam2d, launch pt_slam
- [ ] visualize the generated map and pose
