# DRL Based Object Detection and Tracking from a UAV

This research conducts a performance analysis of multiple Deep Reinforcement Learning (DRL) algorithms for image based visual servoing (IBVS) controller for unmanned aerial vehicles (UAVs). Our porposed methods compares that performance of each model on the computed bounding-box error and linear-velocity between the UAV’s position and the target
 object. For object detection, Yolov2 algorithm is utilized to identify
 the target. The computer bounding box is then fed to the selected
 DRL algorithm for the visual servoing task. We perform our
 simulation on the Gazebo environment, a robust platform for
 testing and development of robotic applications. Our analysis
 show that Prioritized Experience Replay-Deep Deterministic Policy
 Gradient (PER-DDPG) provides the fastest convergence on the
 errors in the considered environment. Our findings contribute to
 the advancement of DRL-based IBVS systems, highlighting the
 potential for improved real-time control and accuracy in UAV
 operations. The results underscore the importance of selecting
 optimal DRL algorithms to enhance UAV performance in dynamic
 and complex environments.
