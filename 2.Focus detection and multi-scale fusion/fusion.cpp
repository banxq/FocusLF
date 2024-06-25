#include <iostream>
#include <opencv2/opencv.hpp>
#include "ban.h"
using namespace cv;
using namespace std;






int main()
{

	string path1 = "C:/....../Lego1.jpg";//img1 address
	string path2 = "C:/....../Lego2.jpg";//img2 address

	Mat img1 = imread(path1);
	Mat img2 = imread(path2);
	vector<Mat>mv1, mv2,mv;
	split(img1, mv1);
	split(img2, mv2);
	
	for (int k = 0; k < 3; k++)
	{
		
		Mat i1, i2, p1, p2;
		mv1[k].copyTo(i1);
		mv2[k].copyTo(i2);
		mv1[k].copyTo(p1);
		mv2[k].copyTo(p2);
		Mat flag1 = Mat::zeros(mv1[k].size(), CV_32F);
		Mat flag2 = Mat::zeros(mv1[k].size(), CV_32F);


		ban fu;

		//求NSML
		Mat smla1, smlb1, smla2, smlb2, smla3, smlb3;
		mv1[k].copyTo(smla1); mv1[k].copyTo(smla2); mv1[k].copyTo(smla3);
		mv2[k].copyTo(smlb1); mv2[k].copyTo(smlb2); mv2[k].copyTo(smlb3);
		fu.SML(mv1[k], smla1, 1);
		fu.SML(mv2[k], smlb1, 1);
		fu.SML(mv1[k], smla2, 2);
		fu.SML(mv2[k], smlb2, 2);
		fu.SML(mv1[k], smla3, 3);
		fu.SML(mv2[k], smlb3, 3);
		Mat nsmla, nsmlb;
		mv1[k].copyTo(nsmla); mv1[k].copyTo(nsmlb);
		for (int i = 0; i < mv1[k].rows; i++)
		{
			for (int j = 0; j < mv1[k].cols; j++)
			{
				nsmla.at<float>(i, j) = abs(smla2.at<float>(i, j) - smla1.at<float>(i, j))
					+ abs(smla3.at<float>(i, j) - smla1.at<float>(i, j));
				nsmlb.at<float>(i, j) = abs(smlb2.at<float>(i, j) - smlb1.at<float>(i, j))
					+ abs(smlb3.at<float>(i, j) - smlb1.at<float>(i, j));
			}
		}
		Mat ttt1, ttt2;
		nsmla.copyTo(ttt1);
		nsmlb.copyTo(ttt2);
		normalize(ttt1, ttt1, 0, 1, NORM_MINMAX);
		normalize(ttt2, ttt2, 0, 1, NORM_MINMAX);//Weak texture markers


		imshow("zj", ttt1);
		imshow("zj2", ttt2);
		waitKey(0);
		int num1 = 0;
		int num2 = 0;
		for (int i = 0; i < flag1.rows; i++)
		{
			for (int j = 0; j < flag1.cols; j++)
			{
				if (ttt1.at<float>(i, j) < 0.0001)
				{
					flag1.at<float>(i, j) = 1;

				}
				if (ttt2.at<float>(i, j) < 0.0001)
				{
					flag2.at<float>(i, j) = 1;

				}
			}
		}
		imshow("zj", flag1);
		imshow("zj2", flag2);
		waitKey(0);
		Mat flag = Mat::zeros(mv2[k].size(), CV_32F);
		for (int i = 0; i < flag1.rows; i++)
		{
			for (int j = 0; j < flag1.cols; j++)
			{
				if (flag1.at<float>(i, j) == 1 || flag2.at<float>(i, j) == 1)
				{
					flag.at<float>(i, j) = 1;//Weak texture 
				}
			}
		}
		imshow("zj", flag);
		waitKey(0);



		Mat idma, idmb, mdma, mdmb;
		//cout << img1.type() << endl;
		mv1[k].copyTo(idma);
		mv2[k].copyTo(idmb);

		fu.s_a(nsmla, nsmlb, idma, idmb, mdma, mdmb, flag);
		

		Mat fdmah, fdmal, fdmbh, fdmbl;
		


		i1.convertTo(i1, CV_32F);
		i2.convertTo(i2, CV_32F);
		mdma.convertTo(mdma, CV_32F);
		mdmb.convertTo(mdmb, CV_32F);
		fdmah = fu.fast_GuidedFilter(i1, mdma, 5, 0.000001, 5);
		fdmal = fu.fast_GuidedFilter(i1, mdma, 5, 0.3, 5);
		fdmbh = fu.fast_GuidedFilter(i2, mdmb, 5, 0.000001, 5);
		fdmbl = fu.fast_GuidedFilter(i2, mdmb, 5, 0.3, 5);


		Mat iah, ial, ibh, ibl;
		float nd1, nd2;
		nd1 = nd2 = 0;
		//Mat pp1;
		fu.calculate33(p1, iah, nd1);

		ial = p1 - iah;	
		fu.calculate33(p2, ibh, nd2);
	
		ibl = p2 - ibh;
		Mat fh, fl, t1, t2, t3, t4;
		p1.copyTo(fh);
		p2.copyTo(fl);
		
		
		t1 = fdmah.mul(iah);
		t2 = fdmbh.mul(ibh);
		t3 = fdmal.mul(ial);
		t4 = fdmbl.mul(ibl);
		
		for (int i = 0; i < fh.rows; i++)
		{
			for (int j = 0; j < fh.cols; j++)
			{
				fh.at<float>(i, j) = t1.at<float>(i, j) + t2.at<float>(i, j);
				fl.at<float>(i, j) = t3.at<float>(i, j) + t4.at<float>(i, j);
				
			}
		}

		Mat F = fh + fl;

		normalize(F, F, 0, 1, NORM_MINMAX);
		F.convertTo(F, CV_8U, 255);
		
		imshow("zj", F);
		mv.push_back(F);
		waitKey(0);
	}
	Mat fus;
	merge(mv, fus);
	imwrite("", fus);
	imshow("zj", fus);
	waitKey(0);
	
	return 0;
}