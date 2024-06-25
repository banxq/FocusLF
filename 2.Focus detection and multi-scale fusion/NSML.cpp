#include <opencv2/opencv.hpp>


#include <iostream>
#include <opencv2/opencv.hpp>
#include "ban.h"
using namespace cv;
using namespace std;

// Reflect function for gray images
void ReflectEdgeGray(const cv::Mat& I, int d, cv::Mat& newI) {
	int m = I.rows;
	int n = I.cols;

	// Initialize newI with zeros
	//newI = cv::Mat::zeros(m + 2 * d, n + 2 * d, I.type());

	// Copy the center region of I to newI
	//I.copyTo(newI(cv::Rect(d, d, n, m)));

	// Reflect the top, bottom, left, and right edges of newI
	copyMakeBorder(I, newI, d, d, d, d, cv::BORDER_REFLECT);
	
}

// Main function
cv::Mat ban::MST_SML(const cv::Mat& A, int d1, int d2, int d3) {
	cv::Mat A1;
	A.convertTo(A1, CV_64F);  // Convert A to double type

	cv::Mat matrix_en1;
	//copyMakeBorder(A1, matrix_en1, 1, 1, 1, 1, BORDER_REPLICATE);
	ReflectEdgeGray(A1, 1, matrix_en1);

	cv::Mat I1(A1.size(), CV_64F);

	for (int i = 1; i < A1.rows+1; i++) {
		for (int j = 1; j < A1.cols+1; j++) {
			I1.at<double>(i - 1, j - 1) = std::abs(2 * matrix_en1.at<double>(i, j) - matrix_en1.at<double>(i - 1, j) - matrix_en1.at<double>(i + 1, j)) +
				std::abs(2 * matrix_en1.at<double>(i, j) - matrix_en1.at<double>(i, j - 1) - matrix_en1.at<double>(i, j + 1)) +
				std::abs(2 * matrix_en1.at<double>(i, j) - matrix_en1.at<double>(i - 1, j - 1) - matrix_en1.at<double>(i + 1, j + 1)) +
				std::abs(2 * matrix_en1.at<double>(i, j) - matrix_en1.at<double>(i + 1, j - 1) - matrix_en1.at<double>(i - 1, j + 1));
		}
	}

	cv::Mat I11, I12, I13;
	/*copyMakeBorder(I1, I11, d1, d1, d1, d1, BORDER_REPLICATE);
	copyMakeBorder(I1, I12, d2, d2, d2, d2, BORDER_REPLICATE);
	copyMakeBorder(I1, I13, d3, d3, d3, d3, BORDER_REPLICATE);*/
	ReflectEdgeGray(I1, d1, I11);
	ReflectEdgeGray(I1, d2, I12);
	ReflectEdgeGray(I1, d3, I13);

	cv::Mat SF1(A1.size(), CV_64F);
	cv::Mat SF3(A1.size(), CV_64F);
	cv::Mat SF5(A1.size(), CV_64F);
	cv::Mat MSF1(A1.size(), CV_64F);

	for (int i = 0; i < A1.rows; ++i) {
		for (int j = 0; j < A1.cols; ++j) {
			cv::Mat window1 = I11(cv::Rect(j, i, 2 * d1 + 1, 2 * d1 + 1));
			cv::Mat window3 = I12(cv::Rect(j, i, 2 * d2 + 1, 2 * d2 + 1));
			cv::Mat window5 = I13(cv::Rect(j, i, 2 * d3 + 1, 2 * d3 + 1));

			cv::Mat win1 = window1.mul(window1);
			cv::Mat win3 = window3.mul(window3);
			cv::Mat win5 = window5.mul(window5);

			SF1.at<double>(i, j) = cv::sum(win1)[0];
			SF3.at<double>(i, j) = cv::sum(win3)[0];
			SF5.at<double>(i, j) = cv::sum(win5)[0];

			MSF1.at<double>(i, j) = std::abs(SF1.at<double>(i, j) - SF3.at<double>(i, j)) + std::abs(SF1.at<double>(i, j) - SF5.at<double>(i, j));
		}
	}

	cv::Mat final_matrix;
	MSF1.convertTo(final_matrix, CV_32F);  // Convert back to float type

	return final_matrix;
}






/// <summary>
/// 
/// </summary>
/// <param name="img1">input img</param>
/// <param name="img2">output </param>
void ban::SML(Mat &img1, Mat &img2,int flag)
{
	if (flag == 1)
	{
		img1.convertTo(img1, CV_32F);
		img2.convertTo(img2, CV_32F);
		
		Mat TE;
		img1.copyTo(TE);
		Mat iimg;
		copyMakeBorder(img1, iimg, 1, 1, 1, 1, BORDER_REPLICATE);//iimg£¨514,514£©
		
		for (int i = 1; i < img1.rows + 1; i++)
		{
			for (int j = 1; j < img1.cols + 1; j++)
			{
				float gSML = abs(2 * iimg.at<float>(i, j) - iimg.at<float>(i - 1, j) - iimg.at<float>(i + 1, j))
					+ abs(2 * iimg.at<float>(i, j) - iimg.at<float>(i, j - 1) - iimg.at<float>(i, j + 1))
					+ abs(2 * iimg.at<float>(i, j) - iimg.at<float>(i - 1, j - 1) - iimg.at<float>(i + 1, j + 1))
					+ abs(2 * iimg.at<float>(i, j) - iimg.at<float>(i - 1, j + 1) - iimg.at<float>(i + 1, j - 1));
				TE.at<float>(i - 1, j - 1) = gSML;
			}
		}
	
		Mat temp;
		copyMakeBorder(TE, temp, 1, 1, 1, 1, BORDER_REPLICATE);
		float js = 0.0;
		for (int i = 1; i < img2.rows + 1; i++)
		{
			for (int j = 1; j < img2.cols + 1; j++)
			{
				js = pow(temp.at<float>(i - 1, j - 1), 2) + pow(temp.at<float>(i - 1, j), 2) + pow(temp.at<float>(i - 1, j + 1), 2)
					+ pow(temp.at<float>(i, j - 1), 2) + pow(temp.at<float>(i, j), 2) + pow(temp.at<float>(i, j + 1), 2)
					+ pow(temp.at<float>(i + 1, j - 1), 2) + pow(temp.at<float>(i + 1, j), 2) + pow(temp.at<float>(i + 1, j + 1), 2);
				img2.at<float>(i - 1, j - 1) = js;
			}
		}

	}
	
	if (flag == 2)
	{
		img1.convertTo(img1, CV_32F);
		img2.convertTo(img2, CV_32F);

		Mat TE;
		img1.copyTo(TE);

		Mat iimg;
		copyMakeBorder(img1, iimg, 2, 2, 2, 2, BORDER_REPLICATE);

		for (int i = 2; i < iimg.rows - 2; i++)
		{
			for (int j = 2; j < iimg.cols - 2; j++)
			{
				float gSML = abs(2 * iimg.at<float>(i, j) - iimg.at<float>(i - 1, j) - iimg.at<float>(i + 1, j))
					+ abs(2 * iimg.at<float>(i, j) - iimg.at<float>(i, j - 1) - iimg.at<float>(i, j + 1))
					+ abs(2 * iimg.at<float>(i, j) - iimg.at<float>(i - 1, j - 1) - iimg.at<float>(i + 1, j + 1))
					+ abs(2 * iimg.at<float>(i, j) - iimg.at<float>(i + 1, j + 1) - iimg.at<float>(i + 1, j - 1));

				TE.at<float>(i - 2, j - 2) = gSML;
			}
		}

		Mat temp;
		copyMakeBorder(TE, temp, 2, 2, 2, 2, BORDER_REPLICATE);

		float js = 0.0;
		for (int i = 2; i < temp.rows - 2; i++)
		{
			for (int j = 2; j < temp.cols - 2; j++)
			{
				js = pow(temp.at<float>(i - 2, j - 2), 2) + pow(temp.at<float>(i - 2, j - 1), 2) + pow(temp.at<float>(i - 2, j), 2) + pow(temp.at<float>(i - 2, j + 1), 2) + pow(temp.at<float>(i - 2, j + 2), 2)
					+ pow(temp.at<float>(i - 1, j - 2), 2) + pow(temp.at<float>(i - 1, j - 1), 2) + pow(temp.at<float>(i - 1, j), 2) + pow(temp.at<float>(i - 1, j + 1), 2) + pow(temp.at<float>(i - 1, j + 2), 2)
					+ pow(temp.at<float>(i, j - 2), 2) + pow(temp.at<float>(i, j - 1), 2) + pow(temp.at<float>(i, j), 2) + pow(temp.at<float>(i, j + 1), 2) + pow(temp.at<float>(i, j + 2), 2)
					+ pow(temp.at<float>(i + 1, j - 2), 2) + pow(temp.at<float>(i + 1, j - 1), 2) + pow(temp.at<float>(i + 1, j), 2) + pow(temp.at<float>(i + 1, j + 1), 2) + pow(temp.at<float>(i + 1, j + 2), 2)
					+ pow(temp.at<float>(i + 2, j - 2), 2) + pow(temp.at<float>(i + 2, j - 1), 2) + pow(temp.at<float>(i + 2, j), 2) + pow(temp.at<float>(i + 2, j + 1), 2) + pow(temp.at<float>(i + 2, j + 2), 2);

				img2.at<float>(i - 2, j - 2) = js;
			}
		}
	}

	if (flag == 3)
	{
		img1.convertTo(img1, CV_32F);
		img2.convertTo(img2, CV_32F);

		Mat TE;
		img1.copyTo(TE);

		Mat iimg;
		copyMakeBorder(img1, iimg, 3, 3, 3, 3, BORDER_REPLICATE);

		for (int i = 3; i < iimg.rows - 3; i++)
		{
			for (int j = 3; j < iimg.cols - 3; j++)
			{
				float gSML = abs(2 * iimg.at<float>(i, j) - iimg.at<float>(i - 1, j) - iimg.at<float>(i + 1, j))
					+ abs(2 * iimg.at<float>(i, j) - iimg.at<float>(i, j - 1) - iimg.at<float>(i, j + 1))
					+ abs(2 * iimg.at<float>(i, j) - iimg.at<float>(i - 1, j - 1) - iimg.at<float>(i + 1, j + 1))
					+ abs(2 * iimg.at<float>(i, j) - iimg.at<float>(i + 1, j + 1) - iimg.at<float>(i + 1, j - 1));

				TE.at<float>(i - 3, j - 3) = gSML;
			}
		}

		Mat temp;
		copyMakeBorder(TE, temp, 3, 3, 3, 3, BORDER_REPLICATE);

		float js = 0.0;
		for (int i = 3; i < img2.rows - 3; i++)
		{
			for (int j = 3; j < img2.cols - 3; j++)
			{
				js = pow(temp.at<float>(i - 3, j - 3), 2) + pow(temp.at<float>(i - 3, j - 2), 2) + pow(temp.at<float>(i - 3, j - 1), 2) + pow(temp.at<float>(i - 3, j), 2) + pow(temp.at<float>(i - 3, j + 1), 2) + pow(temp.at<float>(i - 3, j + 2), 2) + pow(temp.at<float>(i - 3, j + 3), 2)
					+ pow(temp.at<float>(i - 2, j - 3), 2) + pow(temp.at<float>(i - 2, j - 2), 2) + pow(temp.at<float>(i - 2, j - 1), 2) + pow(temp.at<float>(i - 2, j), 2) + pow(temp.at<float>(i - 2, j + 1), 2) + pow(temp.at<float>(i - 2, j + 2), 2) + pow(temp.at<float>(i - 2, j + 3), 2)
					+ pow(temp.at<float>(i - 1, j - 3), 2) + pow(temp.at<float>(i - 1, j - 2), 2) + pow(temp.at<float>(i - 1, j - 1), 2) + pow(temp.at<float>(i - 1, j), 2) + pow(temp.at<float>(i - 1, j + 1), 2) + pow(temp.at<float>(i - 1, j + 2), 2) + pow(temp.at<float>(i - 1, j + 3), 2)
					+ pow(temp.at<float>(i, j - 3), 2) + pow(temp.at<float>(i, j - 2), 2) + pow(temp.at<float>(i, j - 1), 2) + pow(temp.at<float>(i, j), 2) + pow(temp.at<float>(i, j + 1), 2) + pow(temp.at<float>(i, j + 2), 2) + pow(temp.at<float>(i, j + 3), 2)
					+ pow(temp.at<float>(i + 1, j - 3), 2) + pow(temp.at<float>(i + 1, j - 2), 2) + pow(temp.at<float>(i + 1, j - 1), 2) + pow(temp.at<float>(i + 1, j), 2) + pow(temp.at<float>(i + 1, j + 1), 2) + pow(temp.at<float>(i + 1, j + 2), 2) + pow(temp.at<float>(i + 1, j + 3), 2)
					+ pow(temp.at<float>(i + 2, j - 3), 2) + pow(temp.at<float>(i + 2, j - 2), 2) + pow(temp.at<float>(i + 2, j - 1), 2) + pow(temp.at<float>(i + 2, j), 2) + pow(temp.at<float>(i + 2, j + 1), 2) + pow(temp.at<float>(i + 2, j + 2), 2) + pow(temp.at<float>(i + 2, j + 3), 2)
					+ pow(temp.at<float>(i + 3, j - 3), 2) + pow(temp.at<float>(i + 3, j - 2), 2) + pow(temp.at<float>(i + 3, j - 1), 2) + pow(temp.at<float>(i + 3, j), 2) + pow(temp.at<float>(i + 3, j + 1), 2) + pow(temp.at<float>(i + 3, j + 2), 2) + pow(temp.at<float>(i + 3, j + 3), 2);

				img2.at<float>(i - 3, j - 3) = js;
			}
		}
	}

}