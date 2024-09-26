# Params
MODE              = "SIL" # "SIL" or "HIL"
FRAME_MAP         = "map"
FRAME_BASE        = "turtlebot/kobuki/base_footprint"
FRAME_PREDICTED_BASE = "turtlebot/kobuki/predicted_base_footprint"
FRAME_DEPTH_CAMERA= "turtlebot/kobuki/realsense_depth"
FRAME_DEAD_RECKONING_BASE = "turtlebot/kobuki/dead_reckoning"
FRAME_OPTIMIZE = "turtlebot/kobuki/optimize"

PUB_KEYFRAME_DEADRECKONING_TOPIC = "/keyframes_deadReckoning"
PUB_ODOM_TOPIC          = "/odom"

SUB_GROUND_TRUTH_TOPIC  = "/turtlebot/kobuki/odom_ground_truth"
PUB_GROUND_TRUTH_TOPIC  = "/odom_ground_truth"

SUB_IMU_TOPIC           = "/turtlebot/kobuki/sensors/imu_data"
SUB_ODOM_TOPIC          = "/turtlebot/joint_states"

SUB_OPTIMIZED_TOPIC     = "/graphslam/optimizedposes"
PUB_OPTIMIZED_TOPIC     = "/odom_optimized"

PUB_DEAD_RECKONING_TOPIC= "/dead_reckoning"

SERVICE_RESET_FILTER    = "ResetFilter"

ROBOT_WHEEL_BASE        = 0.235
ROBOT_WHEEL_RADIUS      = 0.035

STD_ODOM_X_VELOCITY     = 0.0108          # [m/s]
STD_ODOM_Y_VELOCITY     = 0.00109         # [m/s]
STD_ODOM_ROTATE_VELOCITY= 0.107           # [deg/s]
STD_MAG_HEADING         = 1.08             # [deg]


#