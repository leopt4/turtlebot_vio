#include "ros/ros.h"
#include "graph_vio_node.hpp"


int main(int argc, char **argv) {
    ros::init(argc, argv, "graph_vio");
    ros::NodeHandle nh;

    graph_vio_handler graphVIO(nh);

    ros::spin();
    return 0;
}


