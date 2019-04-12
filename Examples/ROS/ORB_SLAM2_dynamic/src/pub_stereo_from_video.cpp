#include <iostream>
#include "cv.h"  
#include "cxcore.h"  
#include "highgui.h"
#include <stdlib.h>
#include <fstream>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/time.h>
unsigned long int i=0;
using namespace std;
ifstream stampfile("/home/lnss/stampfile.txt");
int main ( int argc, char** argv )
	{
		ros::init(argc, argv, "pub_stereo_from_video");
  		ros::NodeHandle nh;
		ros::Rate r(5);
		image_transport::ImageTransport left(nh);
		image_transport::ImageTransport right(nh);
		image_transport::ImageTransport depth(nh);
  		image_transport::Publisher pub_left =  left.advertise ("/multisense/left/image_rect_color", 10);
		image_transport::Publisher pub_right = right.advertise("/multisense/right/image_rect", 10);
		image_transport::Publisher pub_depth = depth.advertise("/multisense/openni_depth", 10);
		cv::VideoCapture captureleftvideo;
		cv::VideoCapture capturerightvideo;
		captureleftvideo.open("/home/lnss/video_left.avi");
		capturerightvideo.open("/home/lnss/video_right.avi");		
		char depthpath[40];
		if((!captureleftvideo.isOpened())||(!captureleftvideo.isOpened()))
		{
			cout<<"fail to load video"<<endl;
			return 0;
		}
		while(ros::ok( ))
		{		
			cv::Mat matleft,matright,matdepth;		
			sprintf(depthpath,"%s%lu%s","/home/lnss/depth/",i,".png");
			captureleftvideo>>matleft;
			capturerightvideo>>matright;
			matdepth=cv::imread(depthpath,CV_LOAD_IMAGE_ANYDEPTH);	
			if(matleft.empty()||matright.empty()||matdepth.empty())
			{
				cout<<"publish data finished"<<endl;
				return 0;			
			}
			char ctime[25];
			ctime[0]='\0';
			if(!stampfile.getline(ctime,25))
			{
				cout<<"fail to get timestamp"<<endl;
				return 0;
			}
			double dtime;
			dtime=atof(ctime);
			sensor_msgs::ImagePtr msgleft  = cv_bridge::CvImage(std_msgs::Header(), "bgr8", matleft).toImageMsg();
			msgleft->header.stamp=ros::Time(dtime);
			pub_left.publish(msgleft);
			sensor_msgs::ImagePtr msgright = cv_bridge::CvImage(std_msgs::Header(), "mono8",matright).toImageMsg();
			msgright->header.stamp=ros::Time(dtime);
			pub_right.publish(msgright);
			sensor_msgs::ImagePtr msgdepth = cv_bridge::CvImage(std_msgs::Header(), "mono16", matdepth).toImageMsg();
			msgdepth->header.stamp=ros::Time(dtime);
			pub_depth.publish(msgdepth);
			i++;
			depthpath[0]='\0';
			r.sleep();
		}	
		return 0;
   }
