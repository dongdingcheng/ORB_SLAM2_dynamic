
#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "gradientUtil.h"
#include "wrappers.h"
#include "CellArray.h"
#include "ACFDetector.h"

#include <opencv2/opencv.hpp>
using namespace cv;

//void Rectify(Mat &image)
//{
//    Mat k=(Mat_<float>(3,3)<<441.599992, 0.000000, 368.113841, 0.000000, 439.914534, 245.345309, 0.000000, 0.000000, 1.000000);
//    Mat D=(Mat_<float>(1,5)<<-0.299346, 0.079829, 0.000973, 0.000094, 0.000000);
//    Mat R=(Mat_<float>(3,3)<<0.999968, 0.000649, -0.007934, -0.000617, 0.999992, 0.004064, 0.007936, -0.004059, 0.999960);
//    Mat P=(Mat_<float>(3,4)<<408.449822, 0.000000, 389.003597, 0.000000, 0.000000, 408.449822, 242.349159, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000);
//    Mat M1,M2;
//    cv::initUndistortRectifyMap(k,D,R,P,image.size(),CV_32F,M1,M2);
//    cv::remap(image,image, M1, M2, cv::INTER_LINEAR);
////if(image.channels()==3) cvtColor(image,image,CV_BGR2GRAY);
//
//    //cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
////clahe->apply(image,image);
//}

void imageCallback(const sensor_msgs::ImageConstPtr &msgRGB) {
    cv::Mat src=cv_bridge::toCvShare(msgRGB,"rgb8")->image;
    ACF::CellArray I, E, O;
    ACF::ACFDetector acfDetector;
    acfDetector.loadModel("/home/divine/LearnVIORB/dynamic/detect/model/myCal.bin");
    try
    {
//        Rectify(src);
        int h = src.rows, w = src.cols, d = 3;
        uint8_t* I = (uint8_t*)ACF::wrCalloc(h * w * d, sizeof(uint8_t));
        for (int k = 0; k < d; ++k) {
            for (int c = 0; c < w; ++c) {
                for (int r = 0; r < h; ++r) {
                    I[k * w * h + c * h + r] = ((uint8_t*)src.data)[r * w * d + c * d + k];
                }
            }
        }
        ACF::Boxes res = acfDetector.acfDetect(I, h, w, d);
        printf("%d\n", (int)res.size());
        cv::cvtColor(src, src, CV_RGB2BGR);
        for (size_t i = 0; i < res.size(); ++i) {

            if(res[i].s>50.0){
                cv::rectangle(src, cv::Rect(res[i].c, res[i].r, res[i].w, res[i].h),
                              cv::Scalar(0, 0, 255), 1);
                printf("%d %d %d %d %.4f\n", res[i].c + 1, res[i].r + 1, res[i].w,
                       res[i].h, res[i].s);
            }
        }
        cv::imshow("result", src);
        cv::waitKey(1);
        ACF::wrFree(I);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("couldn't convert from '%s' to 'bgr8'. ",msgRGB->encoding.c_str());
    }
}

//int main() {
//    try {
//        ACF::CellArray I, E, O;
//        ACF::ACFDetector acfDetector;
//        acfDetector.loadModel("/home/divine/LearnVIORB/dynamic/detect/model/myCal.bin");
//
////        std::vector<cv::String> filenames;
////        cv::String folder = "/home/divine/data/mynteye/20190322_02/left/";
////        cv::glob(folder, filenames);
//        for (size_t i = 0; i < filenames.size(); ++i) {
//            cv::Mat src = cv::imread(filenames[i],CV_LOAD_IMAGE_COLOR), dst;
//            std::cerr << filenames[i] << std::endl;
//            Rectify(src);
//            //cvtColor(src,src,CV_GRAY2BGR);
//            //	cv::resize(dst, src, cv::Size(640, 480));
//            int h = src.rows, w = src.cols, d = 3;
//            uint8_t* I = (uint8_t*)ACF::wrCalloc(h * w * d, sizeof(uint8_t));
//            for (int k = 0; k < d; ++k) {
//                for (int c = 0; c < w; ++c) {
//                    for (int r = 0; r < h; ++r) {
//                        I[k * w * h + c * h + r] = ((uint8_t*)src.data)[r * w * d + c * d + k];
//                    }
//                }
//            }
//            ACF::Boxes res = acfDetector.acfDetect(I, h, w, d);
//            printf("%d\n", (int)res.size());
//            cv::cvtColor(src, src, CV_RGB2BGR);
//            for (size_t i = 0; i < res.size(); ++i) {
//
//                if(res[i].s>50.0){
//                    cv::rectangle(src, cv::Rect(res[i].c, res[i].r, res[i].w, res[i].h),
//                                  cv::Scalar(0, 0, 255), 1);
//                    printf("%d %d %d %d %.4f\n", res[i].c + 1, res[i].r + 1, res[i].w,
//                           res[i].h, res[i].s);
//                }
//            }
//            cv::imshow("result", src);
//            cv::waitKey(1);
//            ACF::wrFree(I);
//        }
//    }
//    catch (const std::string &e) {
//        std::cerr << e << std::endl;
//    }
//    return 0;
//}

int main(int argc, char **argv){
    ros::init(argc, argv, "ros_detector");
    ros::NodeHandle nh;
    cv::startWindowThread();
    image_transport::ImageTransport itl(nh);
    image_transport::Subscriber sub = itl.subscribe("/multisense/left/image_rect_color",1,imageCallback);
    ros::spin();

}
