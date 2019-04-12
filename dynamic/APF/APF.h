#ifndef APF_H
#define APF_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <random>
#include <stdlib.h>
#include "kd_tree.h"
#include <memory>
#include<queue>
using namespace std;
struct point2f{
    float x;
    float y;
};
struct particle{
    point2f pixel; //current location 
    float deltaX;
    float deltaY;
    float w;//weight
};
struct box{
    int x;
    int y;
    int width;
    int height;
    box()
    {}
    box(int _x,int _y,int _width,int _height):x(_x),y(_y),width(_width),height(_height)
    {}
};

class APF{
private:
    particle* mParticles;
    particle* mParticles_tmp;
    int mN;//the number of  particles;
    const int mWidth;
    const int mHeight;
    float mWeightSum;
    float *mCDF;   //cdf
    float max_w;
    float min_w;
    //kd_tree
    shared_ptr<bj::KDTree> kd_tree;
    const int max_num;
    const int min_num;
    gsl_vector* bin_temp;
    float err_bound;
    float conf_quantile;
    double cluster_th;
    
    std::uniform_real_distribution<float> distributionX;
    std::uniform_real_distribution<float> distributionY;
public:
    APF();
    APF(int maxN, int width = 752, int height = 480);
    ~APF();
    void init_distribution();
    void init_distribution(box &bb);
    int SIR();
    void transition(int p_idx,int new_size);
    float evaluate(cv::Mat& diffImage,int new_size);
    void update(cv::Mat& diffImage);
    int proper_size(int k);
    void display_particle(cv::Mat &image);
    void computeGMM(cv::Mat &curImage,box &bb,bool isDisplay=true);
    void clustering(cv::Mat &curImage,box &bb,bool isDisplay=true);
};
















#endif
