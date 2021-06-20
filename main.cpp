#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
void AnisotropicFilter(unsigned char* srcData, int width, int height, int channel, int iter, float k, float lambda, int offset);

#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define CLIP3(x, a, b) MIN2(MAX2(a,x), b)
int main()
{
	Mat srcImage = imread("20.jpg");
	namedWindow("原图", WINDOW_NORMAL);
	imshow("原图", srcImage);

	unsigned char* strData;
	strData = srcImage.data;

	AnisotropicFilter(strData, srcImage.cols, srcImage.rows, srcImage.channels(), 7, 10, 0.23, 3);

	Mat grayImage = Mat(srcImage.rows, srcImage.cols, srcImage.type(), strData, 0);
	namedWindow("修改图", WINDOW_NORMAL);
	imshow("修改图", grayImage);

	waitKey(0);
}

//width 为图像的cols
//height 为图像的rows
//stride 为图像每行的数据cols*channel()
void AnisotropicFilter(unsigned char* srcData, int width, int height, int channel, int iter,float k, float lambda, int offset)
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