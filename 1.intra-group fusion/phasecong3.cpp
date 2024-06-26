#include "phasecong3.h"
#include<opencv2/opencv.hpp>











void ifftshift(Mat& data) {
    Mat data_temp;
    data.copyTo(data_temp);
    int rows = data.rows, cols = data.cols;

    for (int i = 0; i < rows - rows / 2; i++)
    {
        for (int j = 0; j < cols - cols / 2; j++)
        {
            data.at<float>(i, j) = data_temp.at<float>(rows / 2 + i, cols / 2 + j);
        }
    }


    for (int i = 0; i < rows - rows / 2; i++)
    {
        for (int j = cols - cols / 2; j < cols; j++)
        {
            data.at<float>(i, j) = data_temp.at<float>(rows / 2 + i, j - cols + cols / 2);
        }
    }

    for (int i = rows - rows / 2; i < rows; i++)
    {
        for (int j = 0; j < cols - cols / 2; j++)
        {
            data.at<float>(i, j) = data_temp.at<float>(i - rows + rows / 2, j + cols / 2);
        }
    }

    for (int i = rows - rows / 2; i < rows; i++)
    {
        for (int j = cols - cols / 2; j < cols; j++)
        {
            data.at<float>(i, j) = data_temp.at<float>(i - rows + rows / 2, j - cols + cols / 2);
        }
    }
}

void lowpassfilter(Size sze, float cutoff, int n, Mat& radius) {
    if (cutoff < 0 || cutoff > 0.5) {
        cout << "cutoff frequency must be between 0 and 0.5";
    }
    if (n % 1 != 0 || n < 1) {
        cout << "n must be greater than or equal to 1.";
    }
    int rows = sze.height, cols = sze.width;
    radius = Mat::zeros(sze.height, sze.width, CV_32FC1);
    for (int i = -rows / 2; i < rows / 2.0; i++)
    {
        for (int j = -cols / 2; j < cols / 2.0; j++)
        {
            int temp_x = rows / 2 * 2;//Odd numbers are rows - 1; even numbers are row£»
            int temp_y = cols / 2 * 2;
            radius.at<float>(i + rows / 2, j + cols / 2) = 1 / (pow((sqrt(pow((i / (1.0 * temp_x)), 2) + pow((j / (1.0 * temp_y)), 2)) / cutoff), 2 * n) + 1);
        }
    }
    ifftshift(radius);
}
/// <summary>

/// </summary>
/// <param name="data">Mat</param>
/// <param name="model">0£ºlog£»1£ºexp£»2£ºsin£»3£ºcos</param>
void math_mat(Mat& data, int model) {
    if (model == 0) {
        for (int i = 0; i < data.rows; i++)
        {
            for (int j = 0; j < data.cols; j++)
            {
                data.at<float>(i, j) = log(data.at<float>(i, j));
            }
        }
    }
    else if (model == 1) {
        for (int i = 0; i < data.rows; i++)
        {
            for (int j = 0; j < data.cols; j++)
            {
                data.at<float>(i, j) = exp(data.at<float>(i, j));
            }
        }
    }
    else if (model == 2) {
        for (int i = 0; i < data.rows; i++)
        {
            for (int j = 0; j < data.cols; j++)
            {
                data.at<float>(i, j) = sin(data.at<float>(i, j));
            }
        }
    }
    else if (model == 3) {
        for (int i = 0; i < data.rows; i++)
        {
            for (int j = 0; j < data.cols; j++)
            {
                data.at<float>(i, j) = cos(data.at<float>(i, j));
            }
        }
    }
}
/// <summary>
/// Matrix operation (complex ).
/// </summary>
/// <param name="data">input</param>
/// <param name="data_out">output</param>
/// <param name="model">0: Get real part; 1: Get imaginary part.</param>
void math_mat(const Mat& data, Mat& data_out, int model) {
    data_out = Mat::zeros(data.size(), data.type());
    if (model == 0) {
        for (int i = 0; i < data.rows; i++)
        {
            for (int j = 0; j < data.cols; j++)
            {
                data_out.at<float>(i, j) = real(data.at<float>(i, j));
            }
        }
    }
    else if (model == 1) {
        for (int i = 0; i < data.rows; i++)
        {
            for (int j = 0; j < data.cols; j++)
            {
                data_out.at<float>(i, j) = imag(data.at<float>(i, j));
            }
        }
    }
}
/// <summary>
/// atan2£¨image_Y,image_X)
/// </summary>
/// <param name="data_y">DataY</param>
/// <param name="data_x">DataX</param>
/// <param name="data_out">atan2  Radian mode£¨PI£©</param>
/// <param name="model">0:atan2;1:sqrt(x^2 + y^2)</param>
void math_mat(const Mat& data_y, const Mat& data_x, Mat& data_out, int model) {
    if (data_y.size() != data_x.size()) {
        cerr << "Data dimensions are inconsistent.";
    }
    data_out = Mat::zeros(data_y.size(), data_y.type());
    if (model == 0) {
        for (int i = 0; i < data_y.rows; i++)
        {
            for (int j = 0; j < data_y.cols; j++)
            {
                data_out.at<float>(i, j) = atan2(data_y.at<float>(i, j), data_x.at<float>(i, j));
            }
        }
    }
    else if (model == 1) {
        for (int i = 0; i < data_y.rows; i++)
        {
            for (int j = 0; j < data_y.cols; j++)
            {
                data_out.at<float>(i, j) = sqrt(pow(data_y.at<float>(i, j), 2) + pow(data_x.at<float>(i, j), 2));
            }
        }
    }

}
void medianMat(Mat& image, float& median) {
    Mat tmp = image.reshape(1, 1);//make matrix new number of channels and new number of rows. here Put data: 1 row, all cols 

    Mat sorted; //after sorted data
    cv::sort(tmp, sorted, SORT_ASCENDING);//Very time-consuming.
    median = sorted.at<float>(sorted.cols / 2);
}




///
void getColorMask(std::vector<double>& colorMask, double colorSigma) {

    for (int i = 0; i < 256; ++i) {
        double colordiff = exp(-(i * i) / (2 * colorSigma * colorSigma));
        colorMask.push_back(colordiff);
    }

}



void getGausssianMask(cv::Mat& Mask, cv::Size wsize, double spaceSigma) {
    Mask.create(wsize, CV_64F);
    int h = wsize.height;
    int w = wsize.width;
    int center_h = (h - 1) / 2;
    int center_w = (w - 1) / 2;
    double sum = 0.0;
    double x, y;

    for (int i = 0; i < h; ++i) {
        y = pow(i - center_h, 2);
        double* Maskdate = Mask.ptr<double>(i);
        for (int j = 0; j < w; ++j) {
            x = pow(j - center_w, 2);
            double g = exp(-(x + y) / (2 * spaceSigma * spaceSigma));
            Maskdate[j] = g;
            sum += g;
        }
    }
}











/****************LG***************/

vector<double>engso;
vector<vector<double>>engre;
double eng_tem[4];
int subscript = 0;
int ss = 0;
int tem = 1;
double value1 = 0;
//int g = 1;

void phasecong3::phasecong(const Mat image, int nscale, int norient, int minWaveLength, double  mult, double sigmaonf, double dThetaSigma, float k, float cutoff, float g, int noiseMethod, double& eng_so,
    Mat& filter, double& pp,
    Mat& m, Mat& orient, Mat& featType, float T, Mat& pcSum, vector<vector<Mat>>& EO) {
    if (image.channels() != 1) {
        cerr << "The input data type is not a grayscale image.";
    }
    int ww = 0;
    //double pp[64 * 64] = { 0 };
    double eng_t = 0;
    float epsilon = 0.0001;
    int rows = image.rows, cols = image.cols;
    Mat an;
    Mat eng_sy = Mat::zeros(image.size(), CV_32F);
    Mat imagefft;
    image.copyTo(imagefft);
    Mat im_temp = image.clone();
    Mat imagefft_real_and_image[] = { Mat_<float>(im_temp),Mat::zeros(image.size(),CV_32F) };
    merge(imagefft_real_and_image, 2, imagefft);
    dft(imagefft, imagefft);
    split(imagefft, imagefft_real_and_image);

    Mat zero = Mat::zeros(rows, cols, CV_32FC1);
    //M = Mat::zeros(image.size(), CV_32FC1);
    m = Mat::zeros(image.size(), CV_32FC1);


    for (int i = 0; i < nscale; i++)
    {
        vector<Mat> temp_EO;
        for (int j = 0; j < norient; j++)
        {
            Mat p = Mat::zeros(image.size(), CV_32FC2);
            temp_EO.push_back(p);
        }
        EO.push_back(temp_EO);
    }
    vector< Mat> PC(norient);
    Mat covx2, covy2, covxy;
    zero.copyTo(covx2);    
    zero.copyTo(covy2);
    zero.copyTo(covxy);
    Mat EnergyV = Mat::zeros(image.size(), CV_32FC3);
    pcSum = Mat::zeros(image.size(), CV_32FC1);

    Mat radius = Mat::zeros(image.size(), CV_32FC1), theta = Mat::zeros(image.size(), CV_32FC1);
    for (int i = -rows / 2; i < rows / 2.0; i++)
    {
        for (int j = -cols / 2; j < cols / 2.0; j++)
        {
            int temp_x = rows / 2 * 2;
            int temp_y = cols / 2 * 2;

            radius.at<float>(i + rows / 2, j + cols / 2) = sqrt(pow((i / (1.0 * temp_x)), 2) + pow((j / (1.0 * temp_y)), 2));
            float pp = -j / (1.0 * temp_y);
            float ppp = i / (1.0 * temp_x);
            float pppp = atan2(ppp, pp);
            theta.at<float>(i + rows / 2, j + cols / 2) = atan2(-i / (1.0 * temp_x), j / (1.0 * temp_y));
        }
    }
    ifftshift(radius);              
    ifftshift(theta);

    radius.at<float>(0, 0) = 1;
    Mat sintheta = Mat::zeros(theta.size(), theta.type()), costheta = Mat::zeros(theta.size(), theta.type());

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            sintheta.at<float>(i, j) = sin(theta.at<float>(i, j));
            costheta.at<float>(i, j) = cos(theta.at<float>(i, j));
        }
    }

    Mat lp;
    lowpassfilter(Size(cols, rows), 0.45, 15, lp);
    vector<Mat> logGabor;

    for (int s = 0; s < nscale; s++)
    {
        Mat temp;
        radius.copyTo(temp);
        float wavelength = minWaveLength * pow(mult, s);
        float f0 = 1.0 / wavelength;
        ////logGabor.push_back()
        temp = temp * wavelength;//r/f0


        math_mat(temp, 0);
        temp = temp.mul(temp) * (-1);
        temp = temp * (1.0 / (2 * pow(log(sigmaonf), 2)));
        math_mat(temp, 1);
        temp = temp.mul(lp);
        temp.at<float>(0, 0) = 0;
        logGabor.push_back(temp);
    }
    //Mat eng = Mat::zeros(rows, cols, CV_32FC1);
    //Mat eng_s = Mat::zeros(rows, cols, CV_32FC1);

    //double eng_so=0;
    //Mat eng_so = Mat::zeros(rows, cols, CV_32FC1);
    for (int o = 0; o < norient; o++)
    {
        float angl = o * PI / norient;
        double waveLength = minWaveLength;
        Mat ds = sintheta * cos(angl) - costheta * sin(angl);
        Mat dc = costheta * cos(angl) + sintheta * sin(angl);
       
        Mat dtheta;
        math_mat(ds, dc, dtheta, 0);
        dtheta = abs(dtheta);
        pow(dtheta, 2, dtheta);
        dtheta = -dtheta;
        //dtheta = min(dtheta * norient / 2, PI);
        Mat spread = Mat::zeros(dtheta.size(), dtheta.type());
   
        double thetaSigma = PI / norient / dThetaSigma;
        exp(dtheta / (2 * pow(thetaSigma, 2)), spread);

        float tau = 0;
        for (int s = 0; s < nscale; s++)
        {
            filter = logGabor[s].mul(spread);
           

            Mat temp_fft[] = { Mat::zeros(imagefft.size(), CV_32FC1),Mat::zeros(imagefft.size(), CV_32FC1) };
            temp_fft[0] = imagefft_real_and_image[0].mul(filter);
            temp_fft[1] = imagefft_real_and_image[1].mul(filter);

            Mat temp_dft = Mat::zeros(imagefft.size(), CV_32FC2);



            merge(temp_fft, 2, temp_dft);
            Mat imh;
            idft(temp_dft, imh);
            idft(temp_dft, EO[s][o], DFT_COMPLEX_OUTPUT + DFT_SCALE, 0);
           
            Mat real_imag_EO[] = { Mat::zeros(imagefft.size(),image.type()),Mat::zeros(imagefft.size(),image.type()) };

            split(EO[s][o], real_imag_EO);

            waveLength = waveLength * mult;
         
         
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (real_imag_EO[0].at<float>(i, j) < 0)
                    {
                        real_imag_EO[0].at<float>(i, j) = 0;
                    }
                }
            }
            double eng_s = 0;

            Mat imager;
            image.convertTo(imager, CV_32F);
            Mat eng;
            eng = real_imag_EO[0];
            eng_s = eng.dot(eng);
            eng_tem[o] = eng_s;
            eng_so = eng_so + eng_s;
         

        }

    }
    eng_so = eng_so / image.rows / image.cols;
}

