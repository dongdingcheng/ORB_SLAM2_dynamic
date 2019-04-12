/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include<opencv2/core/core.hpp>
#include"/home/dongdingcheng/HORB/HORBSLAM1/include/orbslam/System.h"
#include<init.hpp>
unsigned int *hfids,*hchild;
float *hthrs,*hhs;
float *hp[19],*hsp[19],*hpadp[19];
unsigned long int i=0;
int  hh[19] = {160, 147, 135, 124, 113, 104, 95, 87, 80, 73, 67, 61, 56, 52, 48, 44, 40, 37, 33};
int  wh[19] = {120, 110, 101, 93,  85,  78,  71, 65, 60, 55, 50, 46, 42, 39, 36, 33, 30, 28, 25};

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2::System* pSLAM):mpSLAM(pSLAM){}


    ORB_SLAM2::System* mpSLAM;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();
	Init();




    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM2 RGBD path_to_vocabulary path_to_settings" << endl;        
        ros::shutdown();
        return 1;
    }    

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nh;

    cv::VideoCapture capturevideo;

	capturevideo.open("/home/lnss/video_left.avi");

	char depthpath[80];
	if(!capturevideo.isOpened())
	{
		cout<<"fail to load video"<<endl;
		return 0;
	}
	while(ros::ok())
	{
		cv::Mat left,depth;
		sprintf(depthpath,"%s%lu%s","/home/lnss/depth/",i,".png");
		capturevideo.read(left);

		depth=cv::imread(depthpath,CV_LOAD_IMAGE_ANYDEPTH);
		if(left.empty()||depth.empty())
		{	
			if(i!=0)
				cout<<"all frames are processed"<<endl;
			i=0;			
			continue;		
		}
		//cv::resize(left,left,cv::Size(640, 480),0,0,CV_INTER_LINEAR);
		//cv::resize(right,right,cv::Size(640, 480),0,0,CV_INTER_LINEAR);
		//cv::resize(depth,depth,cv::Size(640, 480),0,0,CV_INTER_LINEAR);
		//left=left.rowRange(300,780).colRange(192,832);
		//right=right.rowRange(300,780).colRange(192,832);
		//depth=depth.rowRange(300,780).colRange(192,832);
		cv::Mat pose=SLAM.TrackRGBD(left,depth,0);
		std::cout<<pose<<std::endl;
		i++;	
	}

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    //SLAM.SaveTrajectoryKITTI("/home/lnss/trajectory.txt");

    ros::shutdown();

    return 0;
}

