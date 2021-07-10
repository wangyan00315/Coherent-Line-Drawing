#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include<cmath>
#include<iostream>
using namespace std;
using namespace cv;

void Grad(Mat src);
void on_tarckbar(int, void*);
void AnisotropicFilter(unsigned char* srcData, int width, int height, int channel, int iter, float k, float lambda, int offset);

#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define CLIP3(x, a, b) MIN2(MAX2(a,x), b)

Mat src, src_gray;      //定义Mat
Mat grad;

int kernel_size = 5;          //内核

int scale = 1;
int delta = 0;
int ddepth = CV_16S;

int g_thresh = 100;

const char* window_name = "Edge Map";

int main(int, char** argv)
{
    src = imread("1.jpg", IMREAD_COLOR); // 载入图片
  //  Grad(src);
	namedWindow("Source", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Source", src); // Show our image inside it.
	
	createTrackbar("Threshold:", "Source", &g_thresh, 255, on_tarckbar);
	on_tarckbar(0, 0);

	waitKey(0); // Wait for a keystroke in the window
    return 0;
}

/*求梯度*/
void Grad(Mat src) {
    
    cvtColor(src, src_gray, COLOR_BGR2GRAY);   //转换为灰度

    namedWindow(window_name, WINDOW_AUTOSIZE);  //创建一个新窗口

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /*在 x 和 y 方向计算“导数”*/

    Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    /*试图接近梯度通过将两个方向的梯度*/
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    imshow(window_name, grad);
    waitKey(0);
}

/*轮廓*/
Mat g_binary;
void on_tarckbar(int, void*) {
	threshold(src, g_binary, g_thresh, 255, THRESH_BINARY);

	const char* binary = "binary";
	namedWindow(binary, WINDOW_AUTOSIZE);
	imshow(binary, g_binary);

	waitKey(0);

	vector < vector < Point > > contours;
	vector<Vec4i> hierarchy;
	findContours(g_binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat drawing = Mat::zeros(g_binary.size(), CV_8UC3);
	//g_binary = Scalar::all(0);
	for (size_t i = 0; i < contours.size(); i++)
	{
		drawContours(drawing, contours, (int)i, Scalar(0, 0, 255), 2, 8, hierarchy, 0, Point());
	}
	namedWindow("Contours", WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
}


int Ws(int x, int y, double r) {
    if (x-y<r)
        return 1;
    else
        return 0;
}
double Wm(double gradnum_x, double gradnum_y, double g) {
    return (1 + tanh(g * (gradnum_y - gradnum_x)))/2;
}

double Wd(double tx, double ty) {
    return fabs(tx * ty);
}

int sign(double tx, double ty) {
    double re = tx * ty;
    if (re >= 0)
        return 1;
    else
        return -1;
}

void AnisotropicFilter(unsigned char* srcData, int width, int height, int channel, int iter, float k, float lambda, int offset)
{
	int i, j, pos1, pos2, pos3, pos4, n, pos_src;
	int NI, SI, EI, WI;
	float cN, cS, cE, cW;
	int stride = width * channel;
	unsigned char* grayData = (unsigned char*)malloc(sizeof(unsigned char) * stride * height);
	unsigned char* pSrc = srcData;
	float MAP[512];
	float kk = 1.0f / (k * k);
	for (i = -255; i <= 255; i++)
	{
		MAP[i + 255] = exp(-i * i * kk) * lambda * i;
	}
	int r, g, b;
	for (n = 0; n < iter; n++)
	{
		//cout << n << endl;
		memcpy(grayData, srcData, sizeof(unsigned char) * height * stride);
		pSrc = srcData;
		for (j = 0; j < height; j++)
		{
			//cout << "j : "<<j << endl;
			for (i = 0; i < width; i++)
			{
				//cout << "j : " << j << " i : " << i << endl;
				pos_src = CLIP3((i * channel), 0, width * channel - 1) + j * stride;
				pos1 = CLIP3((i * channel), 0, width * channel - 1) + CLIP3((j - offset), 0, height - 1) * stride;
				pos2 = CLIP3((i * channel), 0, width * channel - 1) + CLIP3((j + offset), 0, height - 1) * stride;
				pos3 = (CLIP3((i - offset) * channel, 0, width * channel - 1)) + j * stride;
				pos4 = (CLIP3((i + offset) * channel, 0, width * channel - 1)) + j * stride;
				//cout << pos_src << " , " << pos1 << " , " << pos2 << " , " << pos3 << " , " << pos4 << endl;
				b = grayData[pos_src];
				NI = grayData[pos1] - b;
				SI = grayData[pos2] - b;
				EI = grayData[pos3] - b;
				WI = grayData[pos4] - b;
				//cout << b << " , " << NI << " , " << SI << " , " << EI << " , " << WI << endl;
				cN = MAP[NI + 255];// opt:exp(-NI*NI / (k * k));
				cS = MAP[SI + 255];
				cE = MAP[EI + 255];
				cW = MAP[WI + 255];
				int temp = CLIP3((b + (cN + cS + cE + cW)), 0, 255);
				/*cout << temp << endl;
				cout << pSrc[0] << endl;*/
				pSrc[0] = (int)(CLIP3((b + (cN + cS + cE + cW)), 0, 255));
				//cout << pSrc[0] << endl;

				pos_src = pos_src + 1;
				pos1 = pos1 + 1;
				pos2 = pos2 + 1;
				pos3 = pos3 + 1;
				pos4 = pos4 + 1;
				g = grayData[pos_src];
				NI = grayData[pos1] - g;
				SI = grayData[pos2] - g;
				EI = grayData[pos3] - g;
				WI = grayData[pos4] - g;
				cN = MAP[NI + 255];
				cS = MAP[SI + 255];
				cE = MAP[EI + 255];
				cW = MAP[WI + 255];
				pSrc[1] = (int)(CLIP3((g + (cN + cS + cE + cW)), 0, 255));

				pos_src = pos_src + 1;
				pos1 = pos1 + 1;
				pos2 = pos2 + 1;
				pos3 = pos3 + 1;
				pos4 = pos4 + 1;
				r = grayData[pos_src];
				NI = grayData[pos1] - r;
				SI = grayData[pos2] - r;
				EI = grayData[pos3] - r;
				WI = grayData[pos4] - r;
				cN = MAP[NI + 255];
				cS = MAP[SI + 255];
				cE = MAP[EI + 255];
				cW = MAP[WI + 255];
				pSrc[2] = (int)(CLIP3((r + (cN + cS + cE + cW)), 0, 255));
				pSrc += channel;
			}
		}
	}
	free(grayData);
}


