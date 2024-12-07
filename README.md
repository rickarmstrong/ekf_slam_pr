Nice discussion of covariance in a geometric context:
https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/

A fellow traveler attempting the same thing:
https://henryomd.blogspot.com/2017/06/understanding-ekf-slam.html

Another fellow traveler, confused by the fact that x_t is not directly updated by measurements.
https://stackoverflow.com/questions/61486107/how-does-covariance-matrix-p-in-kalman-filter-get-updated-in-relation-to-measu

Gazebo simulation issues. Can't build/run the Create 3 Gazebo sim stuff in `jammy`+`humble`+`harmonic`. 
Works on `noble`+`jazzy`+`harmonic`. See my comment here:
https://github.com/iRobotEducation/create3_sim/issues/234, and this page:
https://gazebosim.org/docs/latest/ros_installation/#summary-of-compatible-ros-and-gazebo-combinations
TLDR: Workaround is to start the simulation one part at a time, sim.launch.py, start the simulation by hitting the run 
button in Gazebo, create3_spawn.launch.py, then create3_gz_nodes.launch.py.
