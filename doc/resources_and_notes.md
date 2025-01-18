### Resources and Notes

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

Excellent discussion of the details. In particular, demonstrates rotating `Q_t`before adding:
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

Issues getting a ROS 2 gazebo turtlebot sim working
https://github.com/ros-navigation/navigation2/issues/4765 
This one includes an entry from Smac stating that it works in an docker image.

https://github.com/ros-navigation/navigation2/issues/4722

Attempting to run in a new docker container
```angular2html
docker run -it --net=host --privileged --volume="${XAUTHORITY}:/root/.Xauthority" --env="DISPLAY=$DISPLAY" -v="/tmp/.gazebo/:/root/.gazebo/" -v /tmp/.X11-unix:/tmp/.X11-unix:rw --shm-size=1000mb osrf/ros:jazzy-desktop-full
apt update
apt upgrade
apt install ros-jazzy-navigation2 -y
apt install ros-jazzy-nav2-bringup -y
apt install ros-jazzy-nav2-minimal-tb* -y

[ERROR] [component_container_isolated-10]: process has died [pid 2578, exit code -4, cmd '/opt/ros/jazzy/lib/rclcpp_components/component_container_isolated --ros-args --log-level info --ros-args -r __node:=nav2_container --params-file /tmp/launch_params_vt9mdti0 --params-file /tmp/launch_params_3lq_8574 -r /tf:=tf -r /tf_static:=tf_static'].

# ros2 launch nav2_bringup navigation_launch.py
[ERROR] [controller_server-1]: process has died [pid 2923, exit code -4, cmd '/opt/ros/jazzy/lib/nav2_controller/controller_server --ros-args --log-level info --ros-args -p use_sim_time:=False --params-file /tmp/launch_params_0b47a4b5 -r /tf:=tf -r /tf_static:=tf_static -r cmd_vel:=cmd_vel_nav'].

```

This (closed) issue sounds similar: https://github.com/ros-navigation/navigation2/issues/3767

Stack trace with `-g`:
#0  0x00007fffefaaa0d4 in mppi::critics::ConstraintCritic::initialize (this=0x555555e5f450) at /root/nav2_ws/src/navigation2/nav2_mppi_controller/src/critics/constraint_critic.cpp:38
#1  0x00007ffff04b8527 in mppi::critics::CriticFunction::on_configure (param_handler=<optimized out>, costmap_ros=..., name=..., parent_name=..., parent=..., this=<optimized out>) at /root/nav2_ws/src/navigation2/nav2_mppi_controller/include/nav2_mppi_controller/critic_function.hpp:83
#2  mppi::CriticManager::loadCritics (this=0x555555e6ecf0) at /root/nav2_ws/src/navigation2/nav2_mppi_controller/src/critic_manager.cpp:55
#3  0x00007ffff04b6d57 in mppi::CriticManager::on_configure (this=0x555555e6ecf0, parent=..., name="FollowPath", costmap_ros=warning: RTTI symbol not found for class 'std::_Sp_counted_ptr_inplace<nav2_costmap_2d::Costmap2DROS, std::allocator<void>, (__gnu_cxx::_Lock_policy)2>'
warning: RTTI symbol not found for class 'std::_Sp_counted_ptr_inplace<nav2_costmap_2d::Costmap2DROS, std::allocator<void>, (__gnu_cxx::_Lock_policy)2>'
std::shared_ptr<nav2_costmap_2d::Costmap2DROS> (use count 9, weak count 3) = {...}, param_handler=0x555555e5c3f0) at /root/nav2_ws/src/navigation2/nav2_mppi_controller/src/critic_manager.cpp:32
#4  0x00007ffff048046a in mppi::Optimizer::initialize (this=0x555555e6ec90, parent=..., name="FollowPath", costmap_ros=..., param_handler=<optimized out>) at /root/nav2_ws/src/navigation2/nav2_mppi_controller/src/optimizer.cpp:52
#5  0x00007ffff0453465 in nav2_mppi_controller::MPPIController::configure (this=0x555555e6ec10, parent=..., name=..., tf=..., costmap_ros=...) at /root/nav2_ws/src/navigation2/nav2_mppi_controller/src/controller.cpp:42
#6  0x00007ffff7c41797 in nav2_controller::ControllerServer::on_configure(rclcpp_lifecycle::State const&) () from /root/nav2_ws/install/nav2_controller/lib/libcontroller_server_core.so
#7  0x00007ffff7d2bb9f in rclcpp_lifecycle::LifecycleNode::LifecycleNodeInterfaceImpl::execute_callback(unsigned int, rclcpp_lifecycle::State const&) const () from /opt/ros/jazzy/lib/librclcpp_lifecycle.so
#8  0x00007ffff7d2bf8d in rclcpp_lifecycle::LifecycleNode::LifecycleNodeInterfaceImpl::change_state(unsigned char, rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn&) () from /opt/ros/jazzy/lib/librclcpp_lifecycle.so
#9  0x00007ffff7d2c965 in rclcpp_lifecycle::LifecycleNode::LifecycleNodeInterfaceImpl::on_change_state(std::shared_ptr<rmw_request_id_s>, std::shared_ptr<lifecycle_msgs::srv::ChangeState_Request_<std::allocator<void> > >, std::shared_ptr<lifecycle_msgs::srv::ChangeState_Response_<std::allocator<void> > >) () from /opt/ros/jazzy/lib/librclcpp_lifecycle.so
#10 0x00007ffff7d2d7bc in std::_Function_handler<void (std::shared_ptr<rmw_request_id_s>, std::shared_ptr<lifecycle_msgs::srv::ChangeState_Request_<std::allocator<void> > >, std::shared_ptr<lifecycle_msgs::srv::ChangeState_Response_<std::allocator<void> > >), std::_Bind<void (rclcpp_lifecycle::LifecycleNode::LifecycleNodeInterfaceImpl::*(rclcpp_lifecycle::LifecycleNode::LifecycleNodeInterfaceImpl*, std::_Placeholder<1>, std::_Placeholder<2>, std::_Placeholder<3>))(std::shared_ptr<rmw_request_id_s>, std::shared_ptr<lifecycle_msgs::srv::ChangeState_Request_<std::allocator<void> > >, std::shared_ptr<lifecycle_msgs::srv::ChangeState_Response_<std::allocator<void> > >)> >::_M_invoke(std::_Any_data const&, std::shared_ptr<rmw_request_id_s>&&, std::shared_ptr<lifecycle_msgs::srv::ChangeState_Request_<std::allocator<void> > >&&, std::shared_ptr<lifecycle_msgs::srv::ChangeState_Response_<std::allocator<void> > >&&) () from /opt/ros/jazzy/lib/librclcpp_lifecycle.so
#11 0x00007ffff7d33522 in ?? () from /opt/ros/jazzy/lib/librclcpp_lifecycle.so
#12 0x00007ffff7f1dfca in ?? () from /opt/ros/jazzy/lib/librclcpp.so
#13 0x00007ffff7e46a54 in rclcpp::Executor::execute_service(std::shared_ptr<rclcpp::ServiceBase>) () from /opt/ros/jazzy/lib/librclcpp.so
#14 0x00007ffff7e4ac6a in rclcpp::Executor::execute_any_executable(rclcpp::AnyExecutable&) () from /opt/ros/jazzy/lib/librclcpp.so
#15 0x00007ffff7e5b254 in rclcpp::executors::SingleThreadedExecutor::spin() () from /opt/ros/jazzy/lib/librclcpp.so
#16 0x00007ffff7e535c9 in rclcpp::spin(std::shared_ptr<rclcpp::node_interfaces::NodeBaseInterface>) () from /opt/ros/jazzy/lib/librclcpp.so
#17 0x0000555555556370 in main ()


