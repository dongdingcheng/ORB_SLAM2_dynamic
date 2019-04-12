#ifndef DYNAMIC_H
#define DYNAMIC_H
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <random>
#include <stdlib.h>
#include<deque>
#include<utility>
#include<thread>
#include <unistd.h>
#include<atomic>
#include<mutex>
#include "../APF/kd_tree.h"
#include "../APF/APF.h"
#include "../detect/src/ACFDetector.h"
#include"fast.hpp"
#include"../../include/Frame.h"
using namespace std;
using namespace cv;

class detectDynamic{
    
public:
    detectDynamic(int max_num,string &binPath,int width = 640, int height = 480);
    ~detectDynamic(){
        printf("clear detectDynamic");
    }
    void Run();
    void insertData(const cv::Mat& Curim, const cv::Mat& lastIm,cv::Mat& H);
    void insertData(const cv::Mat& Curim, const cv::Mat& CurimR, const cv::Mat& lastIm,const cv::Mat& lastImR,cv::Mat& H);
    
    void insertData(const cv::Mat& Curim, const cv::Mat& lastIm);
    void insertData(const cv::Mat& Curim);
    void computeWrapImage(cv::Mat &preImage,cv::Mat &dstImage,cv::Mat &H,cv::Mat &mask);
    void histNormal(cv::Mat &image);
    void detectMotion(box &bb);
    void computeH();
    void processBB(box& bb,bool isDisplay=false);
    float ShiTomasiScore(const cv::Mat &img, const int &u, const int &v);
    void setFinish()
    {
        isFinish=true;
    }
    bool checkFinish()
    {
        return isFinish;
    }
private:
    shared_ptr<APF> apf;
    shared_ptr<ACF::ACFDetector> detect;
    shared_ptr<ORB_SLAM2::Frame>mCurrFrame;
    shared_ptr<ORB_SLAM2::Frame>mLastFrame;
    
    cv::Mat mCurrIm;
    cv::Mat mCurrImR;
    cv::Mat mLastIm;
    cv::Mat mLastImR;
    cv::Mat mdiffImage;
    cv::Mat mH;
    std::vector<cv::Point2f> mDynamicPoints;
    std::deque<std::pair<cv::Mat,cv::Mat>> vecIm;
    bool isFirst;
    atomic<bool> isFinish;
    bool start;
    std::thread mainloop;
    mutex accept;
    int mNUM;
};



#endif
