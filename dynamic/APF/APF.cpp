#include"APF.h"
//define some constants
    const float TRANS_X_STD =15.0;
    const float TRANS_Y_STD=15.0;
    const float TRANS_DELTA=8.0;
    const float MAX_VELOCITY=6.0;
    std::default_random_engine generator;
    std::normal_distribution<float> Gaussian_x(0.0,TRANS_X_STD);//~N(0,TRANS_X_STD)
    std::normal_distribution<float> Gaussian_y(0.0,TRANS_Y_STD);//~N(0,TRANS_Y_STD)
    std::normal_distribution<float> Gaussian_delta(0.0,TRANS_DELTA);//~N(0,TRANS_Y_STD)
    std::uniform_real_distribution<float> uniform(0.0,1.0);
const cv::Mat Gaussian_5x5=(cv::Mat_<float>(5,5)
    <<0.0037,0.0147,0.0256,0.0147,0.0037,
         0.0147,0.0586,0.0952,0.0586,0.0147,
         0.0256,0.0952,0.1502,0.0952,0.0256,
        0.0147,0.0586,0.0952,0.0586,0.0147,
        0.0037,0.0147,0.0256,0.0147,0.0037
    );
const cv::Mat Mean_5x5=(cv::Mat_<float>(5,5)
            <<0.04,0.04,0.04,0.04,0.04,
            0.04,0.04,0.04,0.04,0.04,
            0.04,0.04,0.04,0.04,0.04,
            0.04,0.04,0.04,0.04,0.04,
            0.04,0.04,0.04,0.04,0.04
              );
const float bound=5.0;
//const float A1=2.0;
//const float A2=-1.0;
//const float B0 =1.0;

inline float min(float x,float y){return ( x < y )? x : y;}
inline float max(float x,float y){return ( x > y )? x : y;}
APF::APF(int maxN, int width, int height):mN(maxN),mWidth(width),mHeight(height),max_num(maxN),min_num(1000)
{
    mWeightSum=0.0;
    mCDF=new float[mN+1];
    mParticles=new particle[mN];
    mParticles_tmp=new particle[mN];
    max_w=0.0;
    min_w=1.0;
    bin_temp=gsl_vector_alloc(2);
    gsl_vector_set_all(bin_temp,15);
    kd_tree=shared_ptr<bj::KDTree>(new bj::KDTree(bin_temp));
    err_bound=0.02;
    conf_quantile=2.0537489;
    //update  distributionX and distributionY
    distributionX.param(std::uniform_real_distribution<float>::param_type(bound,mWidth-bound));
    distributionY.param(std::uniform_real_distribution<float>::param_type(bound,mHeight-bound));
}
APF::~APF()
{
    delete [] mParticles;
    delete [] mParticles_tmp;
    gsl_vector_free(bin_temp);
    printf("clear ParticleFilter class");
}
int APF::SIR()
{
    float random_num=uniform(generator);
    int start=0;
    int end = mN;
   while ((start+1) < end)
    {
	int center = (start + end) / 2;
	if (random_num < mCDF[center])
	    end = center;
	else
	    start = center;
    }
       return start;
}
int APF::proper_size(int k)
{
    double t0 = 2.0f / (9 * (k-1));
    double t1 = 1.0 - t0 + sqrt(t0) * conf_quantile;
    return int((k-1)/(2*err_bound) * t1*t1*t1);
}

void APF::init_distribution()
{
    //Initialize to a uniform distribution
    std::uniform_real_distribution<float> initX(bound,mWidth-bound);
    std::uniform_real_distribution<float> initY(bound,mHeight-bound);
    float normW=1.0/mN;
    //init_distribution
    float x=0;float y=0;
    for(int i=0;i<mN;i++)
    {
        x = initX(generator);
        y=initY(generator);
        mParticles[i].pixel.x=x;
        mParticles[i].pixel.y=y;
        
        mParticles[i].deltaX=0;
        mParticles[i].deltaY=0;
        
        mParticles[i].w=normW;
        mCDF[i]=i*normW;
    }
    mCDF[mN]=1.0;
}
void APF::init_distribution(box &bb)
{
    std::uniform_real_distribution<float> initX(bb.x,bb.x+bb.width);
    std::uniform_real_distribution<float> initY(bb.y,bb.y+bb.height);
    float normW=1.0/mN;
    //init_distribution
    float x=0;float y=0;
    for(int i=0;i<mN;i++)
    {
        x=initX(generator);
        y=initY(generator);
        mParticles[i].pixel.x=x;
        mParticles[i].pixel.y=y;
        
        mParticles[i].deltaX=0;
        mParticles[i].deltaY=0;
        
        mParticles[i].w=normW;
        mCDF[i]=i*normW;
    }
    mCDF[mN]=1.0;
}


void APF::transition(int p_idx, int new_size)
{
        float x, y;
    //constant velocity model
    const float thWidth=mWidth-bound;
    const float thHeight=mHeight-bound;
    
    x= mParticles[p_idx].pixel.x+mParticles[p_idx].deltaX+Gaussian_x(generator) ;
    y= mParticles[p_idx].pixel.y+mParticles[p_idx].deltaY+Gaussian_y(generator) ;
    
    if(x<bound||x>thWidth||y<bound||y>thHeight)
    {
        x= distributionX(generator);
        y=distributionY(generator);
      //  x=mParticles[0].pixel.x;
     //   y=mParticles[0].pixel.y;
    }
    mParticles_tmp[new_size].pixel.x=max(bound,min(x,thWidth));
    mParticles_tmp[new_size].pixel.y=max(bound,min(y, thHeight));
    //delta
    x=mParticles[p_idx].deltaX+Gaussian_delta(generator);
    y=mParticles[p_idx].deltaY+Gaussian_delta(generator);
    
    mParticles_tmp[new_size].deltaX=max(-MAX_VELOCITY,min(MAX_VELOCITY,x));
    mParticles_tmp[new_size].deltaY=max(-MAX_VELOCITY,min(MAX_VELOCITY,y));
    gsl_vector_set(bin_temp,0,mParticles_tmp[new_size].pixel.x);
    gsl_vector_set(bin_temp,1,mParticles_tmp[new_size].pixel.y);
    
}
float APF::evaluate(cv::Mat& diffImage, int new_size)
{
    cv::Mat diffImage_5x5;
    
    float  &x=mParticles_tmp[new_size].pixel.x;
    float &y=mParticles_tmp[new_size].pixel.y;
    diffImage_5x5=diffImage.rowRange(int(y+0.5)-2,int(y+0.5)+3).colRange(int(x+0.5)-2,int(x+0.5)+3);
    if( diffImage_5x5.type()!=CV_32FC1)
         diffImage_5x5.convertTo(diffImage_5x5,CV_32FC1);
   //mParticles_tmp[new_size].w=float( Gaussian_5x5.dot( diffImage_5x5 ));
    mParticles_tmp[new_size].w=float(Mean_5x5.dot(diffImage_5x5));
    return mParticles_tmp[new_size].w;
}
void APF::update(cv::Mat& diffImage)
{
    if(diffImage.type()!=CV_32FC1) diffImage.convertTo(diffImage,CV_32FC1);
    cv::normalize(diffImage,diffImage,1.0,0.0,cv::NORM_MINMAX);
    
    int p_idx;
    //Initialize kd-tree
    kd_tree->reset();
    
    int new_size = 0;
   mWeightSum=0;
    do{
        p_idx = SIR();
        transition(p_idx,new_size);
        mWeightSum+=evaluate(diffImage,new_size);
        kd_tree->insert(bin_temp);
        new_size++;
    }while(new_size < min_num ||(new_size < max_num && new_size < proper_size(kd_tree->size())));
    mN=new_size;
    //update CDF
    mCDF[0]=0;
    float invW=1.0/mWeightSum;
    for(int i=0;i<mN;i++)
    {
        mParticles_tmp[i].w=mParticles_tmp[i].w*invW;
        mCDF[i+1]=mCDF[i]+mParticles_tmp[i].w;
    }
    //swap data 
    particle* tmp=mParticles;
    mParticles=mParticles_tmp;
    mParticles_tmp=tmp;
    
}
void APF::display_particle(cv::Mat& image)
{
    cv::Mat &&im1=image.clone();
    if( image.channels()<3) cv::cvtColor(im1,im1,CV_GRAY2BGR);
    cv::Point2f p1;
    cv::Point2f p2;
    for(int i=0;i<mN;i++)
    {
        p1.x=mParticles[i].pixel.x-1;
        p1.y=mParticles[i].pixel.y-1;

        p2.x=mParticles[i].pixel.x+1;
        p2.y=mParticles[i].pixel.y+1;
         
         cv::rectangle(im1,p1,p2,cv::Scalar(0,0,255),-1);
    }
    //display particle number
    p1.x=0.0;
    p1.y=mHeight-11;
    p2.x=p1.x+(mN*100.0)/max_num;
    p2.y=p1.y+10;
    
    cv::rectangle(im1,p1,p2,cv::Scalar(0,0,255),-1);
    p1.x=p2.x;
    p2.x=p1.x+(100-(mN*100.0)/max_num);
    p2.y=p1.y+10;
    cv::rectangle(im1,p1,p2,cv::Scalar(255,255,255),-1);
    cv::namedWindow("Particle Filter");
    cv::imshow("Particle Filter",im1);
    cv::waitKey(1);
}
void APF::computeGMM(cv::Mat& curImage,box &bb,bool isDisplay)
{
    //find mean x y
    float mean_X=0;
    float mean_Y=0;
    float sigma_X=0;
    float sigma_Y=0;
    for(int i=0;i<mN;i++)
    {
        mean_X+=mParticles[i].pixel.x;
        mean_Y+=mParticles[i].pixel.y;
    }
    mean_X=mean_X/mN;
    mean_Y=mean_Y/mN;
    //variance
     for(int i=0;i<mN;i++)
    {
        sigma_X+=(mean_X-mParticles[i].pixel.x)*(mean_X-mParticles[i].pixel.x);
        sigma_Y+=(mean_Y-mParticles[i].pixel.y)*(mean_Y-mParticles[i].pixel.y);
    }
    sigma_X=sigma_X/(mN-1);
    sigma_Y=sigma_Y/(mN-1);
    
    float std_sigma_X=sqrt(sigma_X);
    float std_sigma_Y=sqrt(sigma_Y);
    
    float th_x_left=mean_X-3*std_sigma_X;
    float th_x_right=mean_X+3*std_sigma_X;
    
    float th_y_left=mean_Y-3*std_sigma_Y;    
    float th_y_right=mean_Y+3*std_sigma_Y;
    
    std::vector<int> idx;
    idx.reserve(mN);
    for(int i=0;i<mN;i++)
    {
        if(mParticles[i].pixel.x>th_x_right||mParticles[i].pixel.x<th_x_left||mParticles[i].pixel.y<th_y_left||mParticles[i].pixel.y>th_y_right)
            continue;
        idx.push_back(i);
    }
    
    mean_X=0;
    mean_Y=0;
    sigma_X=0;
    sigma_Y=0;
    //recompute 
    for(short i=0;i<idx.size();i++)
    {
        mean_X+=mParticles[idx[i]].pixel.x;
        mean_Y+=mParticles[idx[i]].pixel.y;
    }
    mean_X=mean_X/idx.size();
    mean_Y=mean_Y/idx.size();
      for(short i=0;i<idx.size();i++)
    {
        sigma_X+=(mean_X-mParticles[idx[i]].pixel.x)*(mean_X-mParticles[idx[i]].pixel.x);
        sigma_Y+=(mean_Y-mParticles[idx[i]].pixel.y)*(mean_Y-mParticles[idx[i]].pixel.y);
    }
    sigma_X= sqrt(sigma_X/(idx.size()-1));
    sigma_Y= sqrt(sigma_Y/(idx.size()-1));
    
    
    //display 
    cv::Point2f pt1(mean_X-1.5*sigma_X,mean_Y-1.5*sigma_Y);
    cv::Point2f pt2(mean_X+1.5*sigma_X,mean_Y+1.5*sigma_Y);

    if(pt1.x>bound&&pt1.x<mWidth-bound&&pt1.y>bound&&pt1.y<mHeight-bound&&pt2.x>bound&&pt2.x<mWidth-bound&&pt2.y>bound&&pt2.y<mHeight-bound)
    {
        if((pt2.x-pt1.x)<(pt2.y-pt1.y)&&((pt2.x-pt1.x)*(pt2.y-pt1.y))<10000)
        {
            bb.x=int(pt1.x);bb.y=int(pt1.y);bb.width=int(pt2.x-pt1.x);bb.height=int(pt2.y-pt1.y);
        }
        else 
        { bb.x=-1;bb.y=-1;bb.height=0;bb.width=0;}
    }
    else
       { bb.x=-1;bb.y=-1;bb.height=0;bb.width=0;}
       //show
     if(isDisplay) 
     {
    cv::Mat &&image=curImage.clone();
    if(image.channels()<3) cv::cvtColor(image,image,CV_GRAY2BGR);
       if(bb.x>0&&bb.y>0)
       {
           cv::rectangle(image,cv::Rect(bb.x,bb.y,bb.width,bb.height),cv::Scalar(255,255,0),2);
       }
        cv::imshow("moving object using Particle Filter",image);
        cv::waitKey(1);
     }
     
}
void APF::clustering(cv::Mat& curImage, box& bb, bool isDisplay)
{
    /***step 1 clear noise data  3delta criterion****/
     float mean_X=0;
    float mean_Y=0;
    float sigma_X=0;
    float sigma_Y=0;
    for(int i=0;i<mN;i++)
    {
        mean_X+=mParticles[i].pixel.x;
        mean_Y+=mParticles[i].pixel.y;
    }
    mean_X=mean_X/mN;
    mean_Y=mean_Y/mN;
    //variance
     for(int i=0;i<mN;i++)
    {
        sigma_X+=(mean_X-mParticles[i].pixel.x)*(mean_X-mParticles[i].pixel.x);
        sigma_Y+=(mean_Y-mParticles[i].pixel.y)*(mean_Y-mParticles[i].pixel.y);
    }
    sigma_X=sigma_X/(mN-1);
    sigma_Y=sigma_Y/(mN-1);
    
    float std_sigma_X=sqrt(sigma_X);
    float std_sigma_Y=sqrt(sigma_Y);
    
    float th_x_left=mean_X-3*std_sigma_X;
    float th_x_right=mean_X+3*std_sigma_X;
    
    float th_y_left=mean_Y-3*std_sigma_Y;    
    float th_y_right=mean_Y+3*std_sigma_Y;
    
    std::vector<int> idx;
    idx.reserve(mN);
    for(int i=0;i<mN;i++)
    {
        if(mParticles[i].pixel.x>th_x_right||mParticles[i].pixel.x<th_x_left||mParticles[i].pixel.y<th_y_left||mParticles[i].pixel.y>th_y_right)
            continue;
        idx.push_back(i);
    }
    /***step 2 re-compute****/
    mean_X=0;
    mean_Y=0;
    sigma_X=0;
    sigma_Y=0;

    for(short i=0;i<idx.size();i++)
    {
        mean_X+=mParticles[idx[i]].pixel.x;
        mean_Y+=mParticles[idx[i]].pixel.y;
    }
    mean_X=mean_X/idx.size();
    mean_Y=mean_Y/idx.size();
      for(short i=0;i<idx.size();i++)
    {
        sigma_X+=(mean_X-mParticles[idx[i]].pixel.x)*(mean_X-mParticles[idx[i]].pixel.x);
        sigma_Y+=(mean_Y-mParticles[idx[i]].pixel.y)*(mean_Y-mParticles[idx[i]].pixel.y);
    }
    sigma_X= sqrt(sigma_X/(idx.size()-1));
    sigma_Y= sqrt(sigma_Y/(idx.size()-1));
    
    std::vector<int>().swap(idx);
    idx.reserve(mN);
    
    cv::Point2f pt1(mean_X-3*sigma_X,mean_Y-3.0*sigma_Y);
    cv::Point2f pt2(mean_X+3*sigma_X,mean_Y+3.0*sigma_Y);
   //check box is ok?
    if(pt1.x>bound&&pt1.x<mWidth-bound&&pt1.y>bound&&pt1.y<mHeight-bound&&pt2.x>bound&&pt2.x<mWidth-bound&&pt2.y>bound&&pt2.y<mHeight-bound)
    {
        if((pt2.x-pt1.x)<(pt2.y-pt1.y))//&&((pt2.x-pt1.x)*(pt2.y-pt1.y))<1.5e4)
        {
            //bb is ok, and then conpute mean x y with weight
            for(int i=0;i<mN;i++){
                point2f &pp=mParticles[i].pixel;
                if(pp.x>pt1.x&&pp.y>pt1.y&&pp.x<pt2.x&&pp.y<pt2.y)
                    idx.push_back(i);
            }
            if(idx.size()<5) { bb.x=-1;bb.y=-1;bb.height=0;bb.width=0; return;}
            float mean_x_w=0;
            float mean_y_w=0;
            float sum_w=0;
            for(auto id:idx){
                point2f &pp=mParticles[id].pixel;
                float &w=mParticles[id].w;
                mean_x_w+=w*pp.x;
                mean_y_w+=w*pp.y;
                sum_w+=w;
            }
            mean_x_w=mean_x_w/sum_w;
            mean_y_w=mean_y_w/sum_w;
            //compute variance with weight
            float sigma_x=0;
            float sigma_y=0;
             for(auto id:idx){
                point2f &pp=mParticles[id].pixel;
                float &w=mParticles[id].w;
                sigma_x+=w*(pp.x-mean_x_w)*(pp.x-mean_x_w);
                sigma_y+=w*(pp.y-mean_y_w)*(pp.y-mean_y_w);
             }
            sigma_x=sigma_x/sum_w;
            sigma_y=sigma_y/sum_w;
            
            bb.x=mean_x_w-sqrt(7.38*sigma_x);
            bb.y=mean_y_w-sqrt(7.38*sigma_y);
            bb.width=2*sqrt(7.38*sigma_x);
            bb.height=2*sqrt(7.38*sigma_y);
        }
        else 
        { bb.x=-1;bb.y=-1;bb.height=0;bb.width=0;}
    }
    else
       { bb.x=-1;bb.y=-1;bb.height=0;bb.width=0;}
       
     if(isDisplay)
    {
        cv::Mat im=curImage.clone();
        if(im.channels()<3) cv::cvtColor(im,im,CV_GRAY2BGR);
        if(bb.x>0&&bb.y>0) {
            //draw ellipse
            cv::ellipse(im,cv::Point2f(bb.x+0.5*bb.width,bb.y+0.5*bb.height),cv::Size(bb.height,bb.width),90,0,360,cv::Scalar(0,255,255),2);
     //       cv::Rect pp(bb.x,bb.y,bb.width,bb.height);
     //       cv::rectangle(im,pp,cv::Scalar(255,255,0),2);
        }
        cv::namedWindow("im");
        cv::imshow("im",im);
        cv::waitKey(1);
    }
}













