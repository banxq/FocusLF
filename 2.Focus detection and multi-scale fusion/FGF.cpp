#include <opencv2/opencv.hpp>
#include "ban.h"
using namespace cv;
using namespace std;


#include <opencv2/opencv.hpp>

using namespace cv;


void Fast_integral(cv::Mat& src, cv::Mat& dst) {
	int nr = src.rows;
	int nc = src.cols;
	int sum_r = 0;
	dst = cv::Mat::zeros(nr + 1, nc + 1, CV_32F);
	for (int i = 1; i < dst.rows; ++i) {
		for (int j = 1, sum_r = 0; j < dst.cols; ++j) {
			
			sum_r = src.at<float>(i - 1, j - 1) + sum_r; //行累加
			dst.at<float>(i, j) = dst.at<float>(i - 1, j) + sum_r;
		}
	}
}

//


void boxfilter(cv::Mat & src, cv::Mat & dst, cv::Size wsize, bool normalize) {

	
	if (wsize.height % 2 == 0 || wsize.width % 2 == 0) {
		fprintf(stderr, "Please enter odd size!");
		exit(-1);
	}
	int hh = (wsize.height - 1) / 2;
	int hw = (wsize.width - 1) / 2;
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_REFLECT);
	src.copyTo(dst);

	
	cv::Mat inte;
	Fast_integral(Newsrc, inte);

	//BoxFilter
	double mean = 0;
	for (int i = hh + 1; i < src.rows + hh + 1; ++i) {  
		for (int j = hw + 1; j < src.cols + hw + 1; ++j) {
			double top_left = inte.at<float>(i - hh - 1, j - hw - 1);
			double top_right = inte.at<float>(i - hh - 1, j + hw);
			double buttom_left = inte.at<float>(i + hh, j - hw - 1);
			double buttom_right = inte.at<float>(i + hh, j + hw);
			if (normalize == true)
				mean = (buttom_right - top_right - buttom_left + top_left) / wsize.area();
			else
				mean = buttom_right - top_right - buttom_left + top_left;

			
			if (mean < 0)
				mean = 0;
			else if (mean > 255)
				mean = 255;
			//dst.at<uchar>(i - hh - 1, j - hw - 1) = static_cast<uchar>(mean);
			dst.at<float>(i - hh - 1, j - hw - 1) = static_cast<float>(mean);
		}
	}
}



cv::Mat ban::fast_GuidedFilter(const cv::Mat& I, const cv::Mat& p, int r, double eps, int s) {
    cv::Mat I_sub, p_sub;
    //I.convertTo(I, CV_32F);
    //p.convertTo(p, CV_32F);
    cv::resize(I, I_sub, cv::Size(), 1.0 / s, 1.0 / s, cv::INTER_NEAREST);
    cv::resize(p, p_sub, cv::Size(), 1.0 / s, 1.0 / s, cv::INTER_NEAREST);

    int r_sub = r / s;

	Mat N;
	Mat op1 = Mat::ones(I_sub.size(), CV_32F);
	boxFilter(op1, N, CV_32F,Size(r_sub, r_sub));
    cout << N.type() << endl;
    cout << I_sub.type() << endl;
	Mat mean_I;
	boxFilter(I_sub, mean_I, CV_32F, Size(r_sub, r_sub));
    mean_I = mean_I / N;
	Mat mean_p; 
	boxFilter(p_sub, mean_p, CV_32F, Size(r_sub, r_sub));
    mean_p = mean_p / N;

	Mat mean_Ip, op2;;
	op2 = I_sub.mul(p_sub);
	boxFilter(op2, mean_Ip, CV_32F, Size(r_sub, r_sub));
    mean_Ip = mean_Ip / N;
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);


	Mat mean_II,op3;
	op3 = I_sub.mul(I_sub);
	boxFilter(op3, mean_II, CV_32F, Size(r_sub, r_sub));
    mean_II = mean_II / N;
    Mat var_I = mean_II - mean_I.mul(mean_I);

    Mat a = cov_Ip / (var_I + eps);
    Mat b = mean_p - a.mul(mean_I);

	Mat mean_a;
	boxFilter(a, mean_a, CV_32F, Size(r_sub, r_sub));
    mean_a = mean_a / N;
	Mat mean_b;
	boxFilter(b, mean_b, CV_32F, Size(r_sub, r_sub));
    mean_b = mean_b / N;

    cv::resize(mean_a, mean_a, I.size(), 0, 0, cv::INTER_LINEAR);
    cv::resize(mean_b, mean_b, I.size(), 0, 0, cv::INTER_LINEAR);

    cv::Mat q = mean_a.mul(I) + mean_b;

    return q;
}

