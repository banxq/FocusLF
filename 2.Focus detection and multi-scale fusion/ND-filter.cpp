#include <opencv2/opencv.hpp>


#include <iostream>
#include <opencv2/opencv.hpp>
#include "ban.h"

using namespace cv;
using namespace std;


/// <summary>
/// 
/// </summary>
/// <param name="img1">£ºinput</param>
/// <param name="ND">£ºoutput</param>
void ban::calculate33(Mat &img1, Mat &img2,float& ND) {

    img1.convertTo(img1, CV_32F);
    img1.copyTo(img2);
    img2.convertTo(img2, CV_32F);
    int row = img1.rows;
    int col = img1.cols;
    Mat iimg;
    copyMakeBorder(img1, iimg, 1, 1, 1, 1, BORDER_REPLICATE);//ÕâÀïµÄiimg£¨514,514£©

    double A1, A2, A3, A4, A5, A6;
    for (int i = 1; i < row-1; i++)
    {
        for (int j = 1; j < col-1; j++)
        {
            A1=img1.at<float>(i - 1, j - 1) + img1.at<float>(i, j - 1) + img1.at<float>(i + 1, j - 1)
                + img1.at<float>(i - 1, j) + img1.at<float>(i, j) + img1.at<float>(i + 1, j)
                + img1.at<float>(i - 1, j + 1) + img1.at<float>(i, j + 1) + img1.at<float>(i + 1, j + 1);

            A2 = -img1.at<float>(i - 1, j - 1) + img1.at<float>(i + 1, j - 1) - img1.at<float>(i - 1, j)
                + img1.at<float>(i + 1, j) - img1.at<float>(i - 1, j + 1) + img1.at<float>(i + 1, j + 1);

            A3 = img1.at<float>(i - 1, j - 1) + img1.at<float>(i, j - 1) + img1.at<float>(i + 1, j - 1)
                - img1.at<float>(i - 1, j + 1) - img1.at<float>(i, j + 1) - img1.at<float>(i + 1, j + 1);

            A4 = img1.at<float>(i - 1, j - 1) + img1.at<float>(i + 1, j - 1) + img1.at<float>(i - 1, j)
                + img1.at<float>(i + 1, j) + img1.at<float>(i - 1, j + 1) + img1.at<float>(i + 1, j + 1);

            A5 = img1.at<float>(i - 1, j - 1) + img1.at<float>(i, j - 1) + img1.at<float>(i + 1, j - 1)
                + img1.at<float>(i - 1, j + 1) + img1.at<float>(i, j + 1) + img1.at<float>(i + 1, j + 1);

            A6 = -img1.at<float>(i - 1, j - 1) + img1.at<float>(i + 1, j - 1) + img1.at<float>(i - 1, j + 1) - img1.at<float>(i + 1, j + 1);
 
            double a00 = 5.0 / 9 * A1 - 1.0 / 3 * A4 - 1.0 / 3 * A5;
            double a10 = 1.0 / 6 * A2;
            double a01 = 1.0 / 6 * A3;
            double a20 = 1.0 / 2 * A4 - 1.0 / 3 * A1;
            double a02 = 1.0 / 2 * A5 - 1.0 / 3 * A1;
            double a11 = 1.0 / 4 * A6;
            double sq = sqrt(1 + a10 * a10 + a01 * a01);
            double L = 2 * a20 / sq;
            double N = 2 * a02 / sq;
            ND = 2 * (L + N);
            img2.at<float>(i - 1, j - 1) = ND;
        }
        
    }
    



    

}

