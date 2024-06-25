#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
class ban
{
public:
	void SML(Mat& img1, Mat& img2, int flag);
	void s_a(Mat& SML1, Mat& SML2, Mat& IDMA, Mat& IDMB, Mat& MDMA, Mat& MDMB,Mat flag);
	void repairSmallRegions(cv::Mat& img1, cv::Mat& img2, double min_area);
	
	Mat fast_GuidedFilter(const cv::Mat& I, const cv::Mat& p, int r, double eps, int s);
	Mat MST_SML(const cv::Mat& A, int d1, int d2, int d3);
	void moveSmallRegions(cv::Mat& sourceImg, cv::Mat& targetImg, double area_thresh, int flag);

	void FastGuidedFilter(cv::Mat& srcImage, cv::Mat& guidedImage, cv::Mat& outputImage, int filterSize, double eps, int samplingRate);
	void calculate33(Mat& img1, Mat& img2, float& ND);
	void guidedFilter(cv::Mat& guidiance, cv::Mat& src, cv::Mat& dst, const int r, float eps);
};