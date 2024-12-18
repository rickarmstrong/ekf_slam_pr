### Resources

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

Excellent discussion of the details. In particular, demonstrates rotating Q_t before adding:
https://www.sush.one/docs/SLAM/EKF.html
https://www.sush.one/docs/SLAM/EKF.html#prediction-step-code


Plot an ellipse: https://stackoverflow.com/questions/10952060/plot-ellipse-with-matplotlib-pyplot


Some discoveries while contemplating angle wrapping in an EKF (searching on "EKF normalize angles"):
A paper specifically about angle wrapping in EKFs: https://arxiv.org/pdf/1708.05551

A filtering library written in Python: https://filterpy.readthedocs.io/en/latest/index.html

A post that makes me think maybe I'm doing the z-z_hat thing wrong:
https://robotics.stackexchange.com/questions/17873/kalman-filter-how-to-solve-angles-near-pi

12/15/24: IT WORKS, or at least, it's _starting_ to look reasonable. We were blowing-up and diverging, 
particularly with more than two landmarks and a range such that we always had all landmarks in-sight. 
Finally peeled the onion down far enough, by resolving these:
* Normalized pose and measurement angles. Tests and refactoring to reduce code duplication were key.
* Non-zero measurement noise. Specifically, I had range variance set to zero. This is apparently Not 
Good. In a hand-waving way, it's not totally surprising that a state estimation scheme based on the 
assumption of perfectly Gaussian noise might go awry with zero-variance 'noise'.

Ugh, my study list gets longer: 
* https://web.mit.edu/2.166/www/handouts/. See lecture 7 for some good stuff about "O-plus and O-minus" notation.