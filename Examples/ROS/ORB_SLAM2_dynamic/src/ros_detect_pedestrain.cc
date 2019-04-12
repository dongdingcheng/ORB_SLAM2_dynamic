#include <ros/ros.h>
#include <stdlib.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cv.h>
#include <iostream>
#include "detectresult.h"
unsigned int *fids, *child, *hfids ,*hchild;
float *thrs, *hs, *hthrs, *hhs;
float* bp[19];float	*sp[19];float *padp[19];float *hp[19];float *hsp[19];float *hpadp[19];
int  h[19] = { 80,73,67,61,56,52,48,44,40,37,33,31,28,25,24,21,20,19,17};
int  w[19] = { 60,55,50,46,42,39,36,33,30,28,25,23,21,19,18,16,15,14,13};
int  hh[19] = {160, 147, 135, 124, 113, 104, 95, 87, 80, 73, 67, 61, 56, 52, 48, 44, 40, 37, 33};
int  wh[19] = {120, 110, 101, 93,  85,  78,  71, 65, 60, 55, 50, 46, 42, 39, 36, 33, 30, 28, 25};
DSResult result;
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	 cv::Mat tempo=cv_bridge::toCvShare(msg,"bgr8")->image;	
	 try
	{
		detectpedestrain(tempo,result);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("couldn't convert from '%s' to 'bgr8'. ",msg->encoding.c_str());	
	}
	
}
int main ( int argc , char**argv )
{
ros::init(argc, argv ,"ros_detect_pedestrain");
ros::NodeHandle  nh;
cv::startWindowThread();
image_transport::ImageTransport itl(nh);
image_transport::Subscriber sub = itl.subscribe("/multisense/left/image_rect_color",1,imageCallback);
ros::spin(); 
}

