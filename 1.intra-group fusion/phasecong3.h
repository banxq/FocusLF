#pragma once
//头文件
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>


#define PI acos(-1)

//命名空间
using namespace std;
using namespace cv;


class phasecong3
{
public://pha.phasecong(image_2,         4,          5,          3,                 2.5,             0.55,           2.0,       0.4,           10,         -1,    M, m, orient, featType, T, pcSum);
	void phasecong(const Mat image, int scale, int norient, int intWaveLength, double mult, double sigmaonf, double dThetaSigma, float k, float cutoff, float g, int noiseMethod, double& eng_so, Mat& filter, double& pp, Mat& m, Mat& orient, Mat& featType, float T, Mat& pcSum, vector<vector<Mat>>& EO);
	//void getsub(vector<vector<double>>& sub_result);
};

