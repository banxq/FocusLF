#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <map>
#include <cstdlib>
#include <algorithm>
#include "group.h"



int ROI_Width = 80;//Clear area settings, the size can be adjusted for different images.
int ROI_Height = 80;

class MyStruct
{
public:
    int x;
    int y;
    int width;
    int height;

    MyStruct(int ix = 0, int iy = 0, int iw = 0, int ih = 0)
    {
        x = ix;
        y = iy;
        width = iw;
        height = ih;
    }
    //  friend bool operator< (const MyStruct &ms1, const MyStruct &ms)
    //  {
    //      return ms1.x < ms.x && ms1.y < ms.y && ms1.width < ms.width && ms1.height < ms.height;
    //  }


    friend bool operator< (const MyStruct& ms1, const MyStruct& ms)
    {
        if (ms1.x != ms.x)
        {
            return ms1.x < ms.x;
        }

        if (ms1.y != ms.y)
        {
            return ms1.y < ms.y;
        }

        if (ms1.width != ms.width)
        {
            return ms1.width < ms.width;
        }
        if (ms1.height != ms.height)
        {
            return ms1.height < ms.height;
        }
        return false;
    }


};

Mat RoiImage;
map<MyStruct, float> tMap;
vector<pair<MyStruct, float>> tVector;

float AF_Score(Mat srcImage)
{
    Mat imageGrey;
    //srcImage.copyTo(imageGrey);
    cvtColor(srcImage, imageGrey, COLOR_RGB2GRAY);
    Mat imageSobel;
    Sobel(imageGrey, imageSobel, CV_16U, 1, 1);


    double af_score = 0.0;
    af_score = mean(imageSobel)[0];
    return af_score;
}

bool cmp(const pair<MyStruct, float>& x, const pair<MyStruct, float>& y)
{
    return x.second > y.second;
}

void sortMapByValue(map<MyStruct, float>& tMap, vector<pair<MyStruct, float> >& tVector)
{
    
    for (map<MyStruct, float>::iterator curr = tMap.begin(); curr != tMap.end(); curr++)
        tVector.push_back(make_pair(curr->first, curr->second));

    sort(tVector.begin(), tVector.end(), cmp);
}

void SaveROIValue(Mat sourceImage)
{
    int nr = sourceImage.rows;
    int nc = sourceImage.cols;
    for (int i = 0; i < (nc - ROI_Height); i += 100)
    {
        for (int j = 0; j < (nr - ROI_Width); j += 100)
        {
            Rect rect_roi(i, j, ROI_Width, ROI_Height);
            sourceImage(rect_roi).copyTo(RoiImage);
            float tempvalue = AF_Score(RoiImage);
            tMap.insert(make_pair(MyStruct(i, j, ROI_Width, ROI_Height), tempvalue));
        }
    }
}

Point getCenterPoint(Rect rect)
{
    Point cpt;
    cpt.x = rect.x + cvRound(rect.width / 2.0);
    cpt.y = rect.y + cvRound(rect.height / 2.0);
    return cpt;
}

void DrawAndPutText(Mat sourceImage, Rect rect, float afvalue)
{

    Scalar color = Scalar(0, 0, 255);
    int linewidth = 2;
    rectangle(sourceImage, rect, color, linewidth);
    Point centerPoint = getCenterPoint(rect);
    char tam[100];
    int LeftUpX = rect.x;
    int LeftUpY = rect.y;
    sprintf(tam, "Value%.2f[%d,%d,%d,%d]", afvalue, LeftUpX, LeftUpY, rect.width, rect.height);
    //putText(sourceImage, tam, centerPoint, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 255), 1);
}


//img: input images
//flag: output order
void group::grouptwo(vector<Mat>& img, vector<int>& flag)
{
    vector<Mat>t_img;
    vector<float>tempvalue;
    vector<float> re_value;
    vector<float> value;
    float tt = 0;
    for (int i = 0; i < 10; i++)
    {
        SaveROIValue(img[i]);
        sortMapByValue(tMap, tVector);
        int x = tVector[0].first.x;
        int y = tVector[0].first.y;
        Rect bestrect(x, y, ROI_Width, ROI_Height);
        for (int j = 0; j < 10; j++)
        {
            //t_img = img[0](bestrect);
            t_img.push_back(img[j](bestrect));//The corresponding position is obtained here.
            
            float temp = AF_Score(t_img[j]);
            tempvalue.push_back(temp);
            //Here, ten values corresponding to the same position in the \(i\)-th image will be obtained.
        }
        sort(tempvalue.begin(), tempvalue.end());
        tt = tempvalue[9] - tempvalue[0];
        re_value.push_back(tt);
        float bestvalue = tVector[0].second;
        
        tMap.clear();
        tVector.clear();
        tempvalue.clear();
        t_img.clear();
    }
  
    float max = *max_element(re_value.begin(), re_value.end());
    auto maxap = max_element(re_value.begin(), re_value.end());
    int Pmaxf = maxap - re_value.begin();
    //t_img[Pmaxf];
    SaveROIValue(img[Pmaxf]);
    sortMapByValue(tMap, tVector);
    int x = tVector[0].first.x;
    int y = tVector[0].first.y;
    Rect bestrect(x, y, ROI_Width, ROI_Height);
    for (int j = 0; j < 10; j++)
    {
        //t_img = img[0](bestrect);
        t_img.push_back(img[j](bestrect));
        
        float temp = AF_Score(t_img[j]);
        tempvalue.push_back(temp);
    }
   
    for (int i = 0; i < 10; i++)
    {
        value.push_back(tempvalue[i]);
    }
    sort(tempvalue.begin(), tempvalue.end());
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            if (tempvalue[i] == value[j])
            {
                flag.push_back(j);
            }
        }
        //cout << flag[i] << " " << endl;
    }
}