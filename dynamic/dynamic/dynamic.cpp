#include"dynamic.h"

detectDynamic::detectDynamic(int max_num, std::string& binPath, int width, int height)
{
     apf=shared_ptr<APF>(new APF( max_num,width,height));
     detect=shared_ptr<ACF::ACFDetector>(new ACF::ACFDetector());
    detect->loadModel(binPath);
    isFirst=true;
    isFinish=false;
    start=true;
    mNUM=0;
 //   mainloop=thread(&detectDynamic::Run,this);
}
void detectDynamic::insertData(const cv::Mat& Curim, const cv::Mat& lastIm,cv::Mat& H)
{
   // std::unique_lock<std::mutex> lck (accept);
  //  vecIm.push_back(make_pair(im,H));
    mCurrIm=Curim.clone();
    mLastIm=lastIm.clone();
    mH=H;
    
}
void detectDynamic::insertData(const cv::Mat& Curim)
{
    mCurrIm=Curim.clone();
    if(isFirst)
        return;
    computeH();
}

void detectDynamic::insertData(const cv::Mat& Curim, const cv::Mat& lastIm)
{
        mCurrIm=Curim;
        mLastIm=lastIm;
        computeH();
}
void detectDynamic::insertData(const cv::Mat& Curim, const cv::Mat& CurimR, const cv::Mat& lastIm, const cv::Mat& lastImR, cv::Mat& H)
{
    mCurrIm=Curim;
    mCurrImR=CurimR;
    mLastIm=lastIm;
    mLastImR=lastImR;
    mH=H;
}

void detectDynamic::computeH()
{
    std::vector<KeyPoint> keypoints_1;keypoints_1.reserve(500);
   cv::Ptr<FeatureDetector> detector=cv::FastFeatureDetector::create(30);
    detector->detect( mLastIm, keypoints_1 );
    std::vector<cv::Point2f> KP1,KP2;
    KP1.reserve(500);KP2.reserve(500);
    for(auto kp:keypoints_1)
    {
        KP1.push_back(kp.pt);
    }
    std::vector<uchar> status;
    std::vector<float>error;
    cv::calcOpticalFlowPyrLK(mLastIm,mCurrIm,KP1,KP2,status,error);
    
    std::vector<cv::Point2f> kp1,kp2;
    kp1.reserve(500);kp2.reserve(500);
    for(size_t i=0;i<KP2.size();i++)
    {
        if(!status.at(i))//each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, it is set to 0.
            continue;
        if(KP2[i].x<10||KP2[i].x>(mLastIm.cols-10)||KP2[i].y<10||KP2[i].y>(mLastIm.rows-10))
            continue;
        kp1.push_back(KP1.at(i));
        kp2.push_back(KP2.at(i));
    }
    mH=cv::findHomography(kp2,kp1,CV_RANSAC,3,noArray());
    
}

void detectDynamic::detectMotion(box &bb)
{
    if(isFirst)
    {
        isFirst=false;
        bb.x=-1;bb.y=-1;
    }
    else
    {
          /*process*/
            //1.person detector
            if(start){
                int h = mCurrIm.rows, w = mCurrIm.cols, d = 3;
                cv::Mat &&src=mCurrIm.clone();
                cvtColor(src,src,CV_GRAY2BGR);
                uchar* im=(uchar*)ACF::wrCalloc(h * w * d, sizeof(uchar));
                {
                    for (int k = 0; k < d; ++k) {
                    for (int c = 0; c < w; ++c) {
                    for (int r = 0; r < h; ++r) {
                        im[k * w * h + c * h + r] = ((uchar*)src.data)[r * w * d + c * d + k];
                            }
                        }
                    }
                }
                ACF::Boxes &&res=detect->acfDetect(im, h, w, d);
                if(res.size()>0&&res[0].s>50.0)
                {
                    box bb(res[0].c,res[0].r,res[0].w,res[0].h);
                apf->init_distribution(bb);
                start=false;
                }
            }
            if(start) return;
        //2. particle filter
            cv::Mat mask;
            computeWrapImage(mLastIm,mdiffImage,mH,mask);
            mdiffImage=(mCurrIm.mul(mask)-mdiffImage)+(mdiffImage-mCurrIm.mul(mask));
   
            apf->update(mdiffImage);
            apf->display_particle(mCurrIm);
            apf->clustering(mCurrIm,bb,true); 
            //processBB(bb,false);
    }
    mLastIm=mCurrIm.clone();
}
/*
void detectDynamic::processBB(box& bb, bool isDisplay)
{
    //extracting FAST corner features
    vector<cv::Point2f>kp;kp.reserve(100);
    std::vector<cv::Point2f> kp2;kp2.reserve(100);
    
    //tracking failed bb.x=-1&bb.y=-1;using LK tracking
    if(bb.x<0&&bb.y<0)
    {
        //using previous feature points 
        if(mDynamicPoints.empty())
            return;
        cout<<"using LKT tracking"<<endl;
        std::vector<uchar> status;
        std::vector<float>error; 
        std::vector<cv::Point2f>kp_now;
        kp_now.assign(mDynamicPoints.begin(),mDynamicPoints.end());
 //        Mat curImage_temp;
 //   cvtColor(curImage,curImage_temp,CV_BGR2GRAY);
    //TODO set inpute images as a const image ,do not change the type of those images
        cv::calcOpticalFlowPyrLK(mLastIm,mCurrIm,mDynamicPoints,kp_now,status,error);
        //update lastPoint
         for(int p=0;p<kp_now.size();p++)
         {
             if(!status[p])
                 continue;
             kp.push_back(kp_now[p]);
             kp2.push_back(kp_now[p]);
         }
    }
    //tracking successfully
    else{
        int  N=8;
        int colNum=bb.width/N;
        int rowNum=bb.height/N;
        cv::Mat roi=mCurrIm(Rect(bb.x,bb.y,bb.width,bb.height));

        cv::Point2f temp;
        //detect some FAST corners
        for(int y=0;y<rowNum;y++){
            for(int x=0;x<colNum;x++){
            const uchar *data=roi.ptr<uchar>(y*N)+x*N;
                vector<fast_xy> fast_corners;
                fast_corner_detect_10(data,N,N,roi.cols,20,fast_corners);
                if(fast_corners.empty())
                {
                fast_corner_detect_10(data,N,N,roi.cols,10,fast_corners);
                }
                if(fast_corners.empty())
                    continue;
                // find the best one and insert as a feature
                int x_start =x * N;
                int y_start = y * N;
                // sort the corners according to shi-tomasi score
                vector<pair<fast_xy, float> > corner_score;
                int idxBest = 0;
                float scoreBest = -1;
                for (int k = 0; k < fast_corners.size(); k++) {
                            fast_xy &xy = fast_corners[k];
                            xy.x += x_start;
                            xy.y += y_start;
                            if (xy.x < 4 || xy.y < 4 ||
                                xy.x >= roi.cols - 4 ||
                                xy.y >= roi.rows - 4) {
                                // 太边缘不便于计划描述子
                                continue;
                            }
                            float score = ShiTomasiScore(roi, xy.x, xy.y);
                            if ( score > scoreBest) {
                                scoreBest = score;
                                idxBest = k;
                            }
                        }
            if (scoreBest < 0)
                continue;      
            fast_xy &best = fast_corners[idxBest];
            temp.x=bb.x+best.x;
            temp.y=bb.y+best.y;
            kp.push_back(temp) ;
            kp2.push_back(temp);
            }
        }
    }
    
    //compute Depth for each feature point
    if(kp.size()<5)
        return;
    //update lastPoint
    mDynamicPoints.assign(kp.begin(),kp.end());
    
    std::vector<uchar> status;
    std::vector<float>error;
//    Mat curImageL;
//    cvtColor(curImage,curImageL,CV_BGR2GRAY);
    //TODO set inpute images as a const image ,do not change the type of those images
  //  double start_time=clock();
    cv::calcOpticalFlowPyrLK(mCurrIm,mCurrImR,kp,kp2,status,error);
  //  double finsh_time=clock();   
   // cout<<"OpenCV LK: "<<(finsh_time-start_time)/CLOCKS_PER_SEC<<"s"<<endl;
    if(kp2.size()<5)
        return;
    std::vector<float> depth_vec;depth_vec.reserve(50);
    std::vector<int> idx;idx.reserve(50);
    for(int i=0;i<kp2.size();i++)
    {
        if(status[i]) 
        {
            cv::Point2f pl(kp[i].x,kp[i].y);
            cv::Point2f pr(kp2[i].x,kp2[i].y);
             if (pl.x < pr.x || (fabs(pl.y - pr.y) > 5)) 
             {
                     depth_vec.push_back(-1.0);
            }
            else{
                 float disparity = pl.x - pr.x;
                 if (disparity > 1){
                     depth_vec.push_back(49.01398/disparity);
                 }
            }
        }
          else
              depth_vec.push_back(-1.0);
    }
   //process depth information
   float sum=0;
   int num=0;
   //compute mean
   for(ushort p=0;p<depth_vec.size();p++)
   {
       if(depth_vec[p]<0.5||depth_vec[p]>25)
           depth_vec[p]=-1.0;
       else{
           sum+=depth_vec[p];
           num++;
       }
   }
   if(num==0||sum==0)
       return;
   float mean_depth=sum/num;
   //compute std
   sum=0;
     for(ushort p=0;p<depth_vec.size();p++)
     {
         if(depth_vec[p]<0)
             continue;
         sum+=(depth_vec[p]-mean_depth)*(depth_vec[p]-mean_depth);
     }
    float std_depth=sqrt(sum/num);
    sum=0;num=0;
    //clear some noise
         for(ushort p=0;p<depth_vec.size();p++)
     {
         if(depth_vec[p]<0)
             continue;
         if(depth_vec[p]>(mean_depth+2*std_depth)||depth_vec[p]<(mean_depth-2*std_depth))
             depth_vec[p]=-1.0;
         else
         {
             sum+=depth_vec[p];
             num++;
             idx.push_back(p);
         }
     }
     //measured depth of moving object 
   mean_depth=sum/num;//TODO return this value as a measured depth of moving object.
   cout<<"mean_depth:"<<mean_depth<<"m"<<endl;
   
   //save dynamic features into Frame
   for(auto i:idx)
   {
      shared_ptr<ygz::dynamicPoint> dp=shared_ptr<ygz::dynamicPoint>(new ygz::dynamicPoint(kp[i].x,kp[i].y,depth_vec[i]));
       mCurrFrame->mDPoints.push_back(dp);
   }
   //display?
    if(isDisplay)
    {
        cv::Mat &&image=mCurrIm.clone();
        if(image.channels()<3) cvtColor(image,image,CV_GRAY2BGR);
        for(int p=0;p<kp.size();p++){
            if(depth_vec[p]<0)
                continue;
            cv::rectangle(image,cv::Rect(kp[p].x,kp[p].y,2,2),cv::Scalar(255,255,0),-1);
        }
        imshow("Moving Object features",image);
        cv::waitKey(1);
    }
}
*/
void detectDynamic::Run()
{
    bool start=true;

    while(1){
        if(checkFinish())
            break;
        if(isFirst)
        {
            if(vecIm.empty()) 
            { 
                usleep(3000); continue;
            }
            mLastIm=vecIm.front().first;
            vecIm.pop_front();
            isFirst=false;
        }
        else
        {
            if(vecIm.empty()) 
            { 
                usleep(3000); continue;
            }
            mCurrIm=vecIm.front().first;
            mH=vecIm.front().second;
            vecIm.pop_front();
            if(mCurrIm.empty()||mLastIm.empty()||mH.empty())
                continue;
    /*process*/

            //1.person detector
            if(start){
                int h = mCurrIm.rows, w = mCurrIm.cols, d = 3;
                cv::Mat &&src=mCurrIm.clone();
                cvtColor(src,src,CV_GRAY2BGR);
                uchar* im=(uchar*)ACF::wrCalloc(h * w * d, sizeof(uchar));
                {
                    for (int k = 0; k < d; ++k) {
                    for (int c = 0; c < w; ++c) {
                    for (int r = 0; r < h; ++r) {
                        im[k * w * h + c * h + r] = ((uchar*)src.data)[r * w * d + c * d + k];
                            }
                        }
                    }
                }
                ACF::Boxes &&res=detect->acfDetect(im, h, w, d);
                if(res.size()>0&&res[0].s>50.0)
                {
                    box bb(res[0].c,res[0].r,res[0].w,res[0].h);
                apf->init_distribution(bb);
                start=false;
                }
            }
            if(start) continue;
        //2. particle filter
            cv::Mat mask;
            computeWrapImage(mLastIm,mdiffImage,mH,mask);
            mdiffImage=(mCurrIm.mul(mask)-mdiffImage)+(mdiffImage-mCurrIm.mul(mask));
            box bb;
            apf->update(mdiffImage);
            apf->display_particle(mCurrIm);
            apf->computeGMM(mCurrIm,bb);
            
        }
        mLastIm=mCurrIm;

    }//while
    mainloop.join();
}

void detectDynamic::computeWrapImage(cv::Mat& preImage, cv::Mat& dstImage, cv::Mat& H, cv::Mat& mask)
{
     int rows=preImage.rows;int cols=preImage.cols;
    dstImage=Mat::zeros(preImage.size(),CV_8U);
    mask=Mat::zeros(preImage.size(),CV_8U);
    int int_x;
    int int_y;
    int int_x_float;
    int int_y_float;
    
    int invFloatX;
    int invFloatY;

    float float_x;
    float float_y;
    float  float_z;
    if(H.type()!=CV_32F) H.convertTo(H,CV_32F);
    float H1[3];H1[0]=H.at<float>(0,0);H1[1]=H.at<float>(0,1);H1[2]=H.at<float>(0,2);
    float H2[3];H2[0]=H.at<float>(1,0);H2[1]=H.at<float>(1,1);H2[2]=H.at<float>(1,2);
    float H3[3];H3[0]=H.at<float>(2,0);H3[1]=H.at<float>(2,1);H3[2]=H.at<float>(2,2);    
    

    for(int y=0;y<rows;y++){
        for(int x=0;x<cols;x++)
        {
            //step 1 compute original coordinates
           float_z=1/(H3[0]*x+H3[1]*y+H3[2]);
            float_x=(H1[0]*x+H1[1]*y+H1[2])*float_z;
            float_y=(H2[0]*x+H2[1]*y+H2[2])*float_z;
            if(float_x<0||float_y<0||float_x>(cols-1)||float_y>(rows-1))
                continue;
            mask.at<uchar>(y,x)=1;
           int_x=int(float_x);
           int_y=int(float_y);
           
            int_x_float=(float_x-int_x)*2048;
            int_y_float=(float_y-int_y)*2048;
            invFloatX=2048-int_x_float;
            invFloatY=2048-int_y_float;
            
            dstImage.at<uchar>(y,x)=uchar(  (invFloatX*invFloatY*preImage.at<uchar>(int_y,int_x)+invFloatY*int_x_float*preImage.at<uchar>(int_y,int_x+1)+invFloatX*int_y_float 
                *preImage.at<uchar>(int_y+1,int_x)+int_x_float*int_y_float*preImage.at<uchar>(int_y+1,int_x+1))>>22);
        }
    }
    
}
float detectDynamic::ShiTomasiScore(const cv::Mat& img, const int& u, const int& v)
{
            float dXX = 0.0;
            float dYY = 0.0;
            float dXY = 0.0;
            const int halfbox_size = 4;
            const int box_size = 2 * halfbox_size;
            const int box_area = box_size * box_size;
            const int x_min = u - halfbox_size;
            const int x_max = u + halfbox_size;
            const int y_min = v - halfbox_size;
            const int y_max = v + halfbox_size;

            if (x_min < 1 || x_max >= img.cols - 1 || y_min < 1 || y_max >= img.rows - 1)
                return 0.0; // patch is too close to the boundary

            const int stride = img.step.p[0];
            for (int y = y_min; y < y_max; ++y) {
                const uint8_t *ptr_left = img.data + stride * y + x_min - 1;
                const uint8_t *ptr_right = img.data + stride * y + x_min + 1;
                const uint8_t *ptr_top = img.data + stride * (y - 1) + x_min;
                const uint8_t *ptr_bottom = img.data + stride * (y + 1) + x_min;
                for (int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
                    float dx = *ptr_right - *ptr_left;
                    float dy = *ptr_bottom - *ptr_top;
                    dXX += dx * dx;
                    dYY += dy * dy;
                    dXY += dx * dy;
                }
            }

            // Find and return smaller eigenvalue:
            dXX = dXX / (2.0 * box_area);
            dYY = dYY / (2.0 * box_area);
            dXY = dXY / (2.0 * box_area);
            return 0.5 * (dXX + dYY - sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
}

void detectDynamic::histNormal(cv::Mat& image)
{   
    if(image.channels()==3)
    cvtColor(image,image,CV_BGR2GRAY);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(0.5, cv::Size(8, 8));
    clahe->apply(image,image);
}
