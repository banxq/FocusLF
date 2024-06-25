#include <iostream>
#include <opencv2/opencv.hpp>
#include "ban.h"
using namespace cv;
using namespace std;


//ͼ��ƴ��
Mat imageStitching(std::vector<cv::Mat> images, int marginSize) {
	// �������ͼ�������Ƿ��㹻
	if (images.size() < 2) {
		std::cerr << "At least two images are required." << std::endl;
		return cv::Mat();
	}

	// �����������ͼ���Ƿ������ͬ�ĳߴ������
	for (int i = 0; i < images.size() - 1; i++) {
		if (images[i].size() != images[i + 1].size() || images[i].type() != images[i + 1].type()) {
			std::cerr << "All images must have the same size and type." << std::endl;
			return cv::Mat();
		}
	}

	// ����ͼ����ܿ�Ⱥ͸߶�
	int totalWidth = 0;
	int maxHeight = 0;
	for (const auto& image : images) {
		/*totalWidth += image.cols + marginSize;
		maxHeight = std::max(maxHeight, image.rows);*/

		totalWidth = std::max(totalWidth, image.cols);
		maxHeight += image.rows + marginSize;
	}

	// �������ͼ��
	//Mat outputImage(maxHeight, totalWidth - marginSize, images[0].type(), cv::Scalar(0));//��
	Mat outputImage(maxHeight - marginSize, totalWidth, images[0].type(), cv::Scalar(0));
	// ������ͼ��ƴ�ӵ����ͼ����
	int currentX = 0;
	//����ƴ��
	/*for (const auto& image : images) {
		cv::Rect roi(currentX, 0, image.cols, image.rows);
		image.copyTo(outputImage(roi));
		currentX += image.cols + marginSize;
	}*/
	//����ƴ��
	for (const auto& image : images) {
		cv::Rect roi(0, currentX, image.cols, image.rows);
		image.copyTo(outputImage(roi));
		currentX += image.rows + marginSize;
	}

	return outputImage;
}
Mat imageStitching2(std::vector<cv::Mat> images, int marginSize) {
	// �������ͼ�������Ƿ��㹻
	if (images.size() < 2) {
		std::cerr << "At least two images are required." << std::endl;
		return cv::Mat();
	}

	// �����������ͼ���Ƿ������ͬ�ĳߴ������
	for (int i = 0; i < images.size() - 1; i++) {
		if (images[i].size() != images[i + 1].size() || images[i].type() != images[i + 1].type()) {
			std::cerr << "All images must have the same size and type." << std::endl;
			return cv::Mat();
		}
	}

	// ����ͼ����ܿ�Ⱥ͸߶�
	int totalWidth = 0;
	int maxHeight = 0;
	for (const auto& image : images) {
		totalWidth += image.cols + marginSize;
		maxHeight = std::max(maxHeight, image.rows);


	}

	// �������ͼ��
	Mat outputImage(maxHeight, totalWidth - marginSize, images[0].type(), cv::Scalar(0));//��

	// ������ͼ��ƴ�ӵ����ͼ����
	int currentX = 0;
	//����ƴ��
	for (const auto& image : images) {
		cv::Rect roi(currentX, 0, image.cols, image.rows);
		image.copyTo(outputImage(roi));
		currentX += image.cols + marginSize;
	}

	return outputImage;
}


// ��֪�ֿ����������ͼ��ֿ�
void ImageBlock(const Mat& src, int segRow, int seCol, vector<Mat>& result)
{
	int segHeight = src.rows / segRow; // �ֿ�߶�
	int segWidth = src.cols / seCol;  // �ֿ���

	Mat roiImg;
	for (int i = 0; i < segRow; ++i)
	{
		for (int j = 0; j < seCol; ++j)
		{
			cv::Rect rect(j * segWidth, i * segHeight, segWidth, segHeight);
			src(rect).copyTo(roiImg);



			std::string str = std::to_string(i) + std::to_string(j);
			//imwrite(str + "_block.png", roiImg);
			//imshow(str, roiImg);
			result.push_back(roiImg.clone());


		}
	}

}

//�����������Ҫ��uchar����
void bwareaopen(Mat src, Mat& dst, double min_area) {
	dst = src.clone();
	vector<vector<Point> > 	contours;
	vector<Vec4i> 			hierarchy;
	findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
	if (!contours.empty() && !hierarchy.empty()) {
		vector<vector<Point> >::const_iterator itc = contours.begin();
		while (itc != contours.end()) {
			Rect rect = boundingRect(Mat(*itc));
			double area = contourArea(*itc);
			if (area < min_area) {
				for (int i = rect.y; i < rect.y + rect.height; i++) {
					uchar* output_data = dst.ptr<uchar>(i);
					for (int j = rect.x; j < rect.x + rect.width; j++) {
						if (output_data[j] == 255) {
							output_data[j] = 0;
						}
					}
				}
			}
			itc++;
		}
	}
}
//ͼ��ϸ������ʣͼ��Ǽ�;Ŀǰû��ʹ��
void thinImage(const Mat& src, Mat& dst, const int maxIterations = -1) {
	assert(src.type() == CV_8UC1);
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	int count = 0;  //��¼��������  
	while (true) {
		count++;
		if (maxIterations != -1 && count > maxIterations) //���ƴ������ҵ�����������  
			break;
		std::vector<uchar*> mFlag; //���ڱ����Ҫɾ���ĵ�  
		//�Ե���  
		for (int i = 0; i < height; ++i) {
			uchar* p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j) {
				//��������ĸ����������б��  
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6) {
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0) {
						//���  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//����ǵĵ�ɾ��  
		for (std::vector<uchar*>::iterator i = mFlag.begin(); i != mFlag.end(); ++i) {
			**i = 0;
		}

		//ֱ��û�е����㣬�㷨����  
		if (mFlag.empty()) {
			break;
		}
		else {
			mFlag.clear();//��mFlag���  
		}

		//�Ե���  
		for (int i = 0; i < height; ++i) {
			uchar* p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j) {
				//��������ĸ����������б��  
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6) {
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0) {
						//���  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//����ǵĵ�ɾ��  
		for (std::vector<uchar*>::iterator i = mFlag.begin(); i != mFlag.end(); ++i) {
			**i = 0;
		}

		//ֱ��û�е����㣬�㷨����  
		if (mFlag.empty()) {
			break;
		}
		else {
			mFlag.clear();//��mFlag���  
		}
	}
}

cv::Mat filterSmallRegions(cv::Mat& src, cv::Mat& dst, const double area_thresh)
{
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(src, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	std::vector<std::vector<cv::Point>> filtered_contours;
	for (int i = 0; i < contours.size(); i++)
	{
		const double area = cv::moments(contours[i]).m00;
		if (area < area_thresh)
			filtered_contours.push_back(contours[i]);
	}
	cv::Mat filtered_mask = cv::Mat::zeros(src.size(), CV_8UC1);
	cv::fillPoly(filtered_mask, filtered_contours, 255);

	src.copyTo(dst);
	dst.setTo(0, filtered_mask);
	return filtered_mask;
}
void ban::moveSmallRegions(cv::Mat& sourceImg, cv::Mat& targetImg, double area_thresh,int flag)
{
	cv::Mat idmSource, idmSource1;
	idmSource=filterSmallRegions(sourceImg, idmSource1, area_thresh);
	imshow("zj1", idmSource);
	waitKey(0);
	// �������޸����������
	cv::Mat move_mask = cv::Mat::zeros(sourceImg.size(), CV_8UC1);

	// ��ǳ���sourceImg�м�⵽��С����
	for (int i = 0; i < sourceImg.rows; i++) {
		for (int j = 0; j < sourceImg.cols; j++) {
			if (idmSource.at<uchar>(i, j) == 255) {
				move_mask.at<uchar>(i, j) = 255;
			}
		}
	}

	// ����⵽��С�����Ƶ�targetImg��
	if (flag == 0)//��һ�α�֤С�����ԭλ���޸����ִ���
	{
		sourceImg.setTo(255, move_mask);
		targetImg.setTo(0, move_mask);
	}
	if (flag == 1)//�ڶ����ƶ�
	{
		targetImg.setTo(255, move_mask);
		sourceImg.setTo(0, move_mask);
	}
	
}


void ban::repairSmallRegions(cv::Mat& img1, cv::Mat& img2, double min_area) {
	// ʹ��bwareaopen���������������
	cv::Mat idm1, idm2;
	bwareaopen(img1, idm1, min_area);
	bwareaopen(img2, idm2, min_area);

	// �������޸����������
	cv::Mat undecided_mask = cv::Mat::zeros(img1.size(), CV_8UC1);

	// ��ǳ�����ͼ��Ϊ0��λ�ã������޸�����
	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			if (idm1.at<uchar>(i, j) == 0 && idm2.at<uchar>(i, j) == 0) {
				undecided_mask.at<uchar>(i, j) = 255;
			}
		}
	}

	// ��ʼ����С�������
	cv::Mat min_dist(img1.size(), CV_32FC1, cv::Scalar::all(img1.rows));

	// �������޸�����
	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			if (undecided_mask.at<uchar>(i, j) == 255) {
				// ��������ͼ���Ѷ�����ľ���任
				cv::Mat inv_mask, dist, smaller_mask;
				cv::compare(idm1, 0, inv_mask, cv::CMP_EQ);
				cv::distanceTransform(inv_mask, dist, cv::DIST_L1, 3, CV_32FC1);

				// ѡ������С����������޸�
				if (dist.at<float>(i, j) < min_dist.at<float>(i, j)) {
					min_dist.at<float>(i, j) = dist.at<float>(i, j);
					img1.at<uchar>(i, j) = 255;
					img2.at<uchar>(i, j) = 0;  // ����img2��img1�Ĳ���
				}
			}
		}
	}
}





/// <summary>
/// 
/// 
/// </summary>
/// <param name="SML1"> img1 ��SML</param>
/// <param name="SML2"> img2 ��SML</param>

void ban::s_a(Mat& SML1, Mat& SML2,Mat& IDMA,Mat& IDMB, Mat& MDMA, Mat& MDMB,Mat flag)
{

	
	cout << IDMA.type() << endl;

	for (int i = 0; i < SML1.rows; i++)
	{
		for (int j = 0; j < SML1.cols; j++)
		{
			if (SML1.at<float>(i, j) >= SML2.at<float>(i, j))
			{
				IDMA.at<float>(i, j) = 1;
				IDMB.at<float>(i, j) = 0;
			}
			if (SML1.at<float>(i, j) < SML2.at<float>(i, j))
			{
				IDMA.at<float>(i, j) = 0;
				IDMB.at<float>(i, j) = 1;
			}
			
		}
	}
	imshow("zj1", IDMA);
	imshow("zj2", IDMB);
	waitKey(0);

	

	Mat pp1, pp2;
	int num1 = 0;
	int num2 = 0;
	
	pp1 = IDMA.mul(flag);
	pp2 = IDMB.mul(flag);
	for (int i = 0; i < flag.rows - 511; i = i + 512)
	{
		for (int j = 0; j < flag.cols - 511; j = j + 512)
		{
			for (int x = i; x < i + 512; ++x)
			{
				for (int y = j; y < j + 512; ++y)
				{
					
					if (pp1.at<float>(x, y) != 0)
					{
						num1++;
					}
					if (pp2.at<float>(x, y) != 0)
					{
						num2++;
					}
				}
			}
			if (num1 >= num2)
			{
				for (int x = i; x < i + 512; ++x)
				{
					for (int y = j; y < j + 512; ++y)
					{
						if (flag.at<float>(x, y) == 1)
						{
							IDMA.at<float>(x, y) = 1;
							IDMB.at<float>(x, y) = 0;
						}
					}
				}
			}
			else
			{
				for (int x = i; x < i + 512; ++x)
				{
					for (int y = j; y < j + 512; ++y)
					{
						if (flag.at<float>(x, y) == 1)
						{
							IDMB.at<float>(x, y) = 1;
							IDMA.at<float>(x, y) = 0;
						}
					}
				}
			}
			num1 = 0;
			num2 = 0;
		}
	}


	IDMA.convertTo(IDMA, CV_8U, 255);
	IDMB.convertTo(IDMB, CV_8U, 255);
	moveSmallRegions(IDMA, IDMB, 10000, 0);
	moveSmallRegions(IDMA, IDMB, 500, 1);

	moveSmallRegions(IDMB, IDMA, 10000, 0);
	moveSmallRegions(IDMB, IDMA, 3000, 1);

	IDMA.convertTo(IDMA, CV_32F);
	IDMB.convertTo(IDMB, CV_32F);
	normalize(IDMA, IDMA, 0, 1, NORM_MINMAX);
	normalize(IDMB, IDMB, 0, 1, NORM_MINMAX);
	Mat IDMA1, IDMB1;
	Mat tIDMA1, tIDMB1;

	IDMA.copyTo(MDMA);
	IDMB.copyTo(MDMB);




}