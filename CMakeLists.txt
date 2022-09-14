cmake_minimum_required(VERSION 3.10)

# set the project name
project(2D_slam)

# add the executable
add_executable(slam slam.cpp)

# include directories
include_directories("/usr/include/eigen3")
# include_directories("/home/jiuzhou/vscode_project/slam-with-a-particle-filter")