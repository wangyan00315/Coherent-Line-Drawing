#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include<cmath>
#include<iostream>
#include <algorithm>
#include <fstream>

void GVF(const cv::Mat& src, cv::Mat& grad_x, cv::Mat& grad_y, cv::Mat& grad_mag);
void Tangent(const cv::Mat& grad_x, const cv::Mat& grad_y, cv::Mat& tan_x, cv::Mat& tan_y);
cv::Point2f ETF(const cv::Mat& tan_x, const cv::Mat& tan_y, const cv::Mat& grad_mag, const cv::Point& x);
void ETF(const cv::Mat& tan_x, const cv::Mat& tan_y, const cv::Mat& grad_mag, cv::Mat& tnew_x, cv::Mat& tnew_y);
void Output(const std::string& filename, const cv::Mat& vx, const cv::Mat& vy);
void AnisotropicFilter(unsigned char* srcData, int width, int height, int channel, int iter, float k, float lambda, int offset);

#define CLIP3(x, a, b) std::min(std::max(a, x), b)
const double PI = 3.1415926;


int main(int, char** argv)
{
	cv::Mat src, grad_x, grad_y, grad_mag, tan_x, tan_y, tnew_x, tnew_y, tnew_x1, tnew_y1, ngrad_x, ngrad_y;
	src = cv::imread("2.jpg", cv::IMREAD_COLOR); // 载入图片
	GVF(src, grad_x, grad_y, grad_mag);
	Tangent(grad_x, grad_y, tan_x, tan_y);
	ETF(tan_x, tan_y, grad_mag, tnew_x, tnew_y);
	Tangent(tnew_x, tnew_y, ngrad_x, ngrad_y);

	//ETF(tnew_x, tnew_y, grad_mag, tnew_x1, tnew_y1);
	Output("file.vecT", tnew_x, tnew_y);
	cv::waitKey(0);
	return 0;
}

/*求GVF*/
void GVF(const cv::Mat& src, cv::Mat& grad_x, cv::Mat& grad_y,cv::Mat& grad_mag) {
	cv::Mat src_gray;

	cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);   //转换为灰度
	imwrite("gray.jpg", src_gray);

	blur(src_gray, src_gray, cv::Size(1, 1));    //模糊

	/*在 x 和 y 方向计算“导数”*/
	int scale = 1;
	int delta = 0;

	Sobel(src_gray, grad_x, CV_32F, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
	Sobel(src_gray, grad_y, CV_32F, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);

	magnitude(grad_x, grad_y, grad_mag);       //取模
	cv::Mat abs_grad_mag;
	convertScaleAbs(grad_mag, abs_grad_mag);    //转换成单通道
	imwrite("GVF.jpg", grad_mag);


}

/*求垂直向量*/
void Tangent(const cv::Mat& grad_x, const cv::Mat& grad_y, cv::Mat& tan_x, cv::Mat& tan_y) {
	int rows = grad_x.rows;
	int cols = grad_x.cols;
	tan_x = cv::Mat(rows, cols, CV_32F);
	tan_y = cv::Mat(rows, cols, CV_32F);
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			tan_x.at<float>(y, x) = -grad_y.at<float>(y, x);
			tan_y.at<float>(y, x) = grad_x.at<float>(y, x);
		}
	}
}


int Phi(const cv::Mat& tan_x, const cv::Mat& tan_y, const cv::Point & x, const cv::Point & y) {
	float result = cv::Point2f(tan_x.at<float>(x), tan_y.at<float>(x)).dot(cv::Point2f(tan_x.at<float>(y), tan_y.at<float>(y)));
	if (result > 0)
		return 1;
	else
		return -1;
}

int Ws(const cv::Point2f& x, const cv::Point2f& y, float r) {
	if (cv::norm(x - y) < r)
		return 1;
	else
		return 0;
}

float Wd(const cv::Mat& tan_x, const cv::Mat& tan_y, const cv::Point& x, const cv::Point& y) {
	float result = cv::Point2f(tan_x.at<float>(x), tan_y.at<float>(x)).dot(cv::Point2f(tan_x.at<float>(y), tan_y.at<float>(y)));
	return cv::abs(result);
}

float Wm(const cv::Mat& grad_mag, const cv::Point& x, const cv::Point& y) {
	return (1 + tanh(grad_mag.at<float>(y) - grad_mag.at<float>(x))) / 2;
}

cv::Point2f ETF(const cv::Mat& tan_x, const cv::Mat& tan_y, const cv::Mat& grad_mag, const cv::Point& x) {
	int ns = 3;
	cv::Point2f sum(0.0f);
	float k = 0;
	for (int r = x.y - ns; r <= x.y + ns; r++) {
		for (int c = x.x - ns; c <= x.x + ns; c++) {
			if (r < 0 || c < 0 || r >= tan_x.rows || c >= tan_x.cols) continue;	

			cv::Point y(c, r);
			cv::Point2f t(tan_x.at<float>(y), tan_y.at<float>(y));
			int phi = Phi(tan_x, tan_y, x, y);
			int ws = Ws(x, y, ns);
			float wd = Wd(tan_x, tan_y, x, y);
			float wm = Wm(grad_mag, x, y);
			sum += phi * t * ws * wd * wm;
			k += abs(phi * ws * wd * wm);
		}
	}
	if (abs(k) <= 0.0000001)
		return cv::Point2f(0, 0);
	return sum/k;

}

void ETF(const cv::Mat& tan_x, const cv::Mat& tan_y, const cv::Mat& grad_mag, cv::Mat& tnew_x, cv::Mat& tnew_y) {
	int rows = tan_x.rows;
	int cols = tan_x.cols;
	tnew_x = cv::Mat(rows, cols, CV_32F);
	tnew_y = cv::Mat(rows, cols, CV_32F);
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			cv::Point2f t_new = ETF(tan_x, tan_y, grad_mag, cv::Point(x, y));
			tnew_x.at<float>(y, x) = t_new.x;
			tnew_y.at<float>(y, x) = t_new.y;
		}
	}
	cv::Mat des;
	magnitude(tnew_x, tnew_y, des);
	convertScaleAbs(des, des);
	imwrite("des.jpg", des);
}


void Output(const std::string& filename, const cv::Mat& vx, const cv::Mat& vy ) {
	std::ofstream outfile;
	outfile.open(filename);
    outfile << vx.cols << " " << vx.rows << std::endl;
	for (int c = 0; c < vx.cols; c++) {
		for (int r = 0; r < vx.rows; r++) {
			outfile << vx.at<float>(r, c) << " " << vy.at<float>(r, c) << std::endl;
		}
	}
}


float gaussian(float x, float sigma) {
	return (1 / (sqrt(PI * 2) * sigma) * exp(-(x * x)/(2*sigma*sigma)));
}

float function(float t, float sigma_c) {
	float sigma_s = 1.6 * sigma_c;
	float rho = 0.99;
	return gaussian(t, sigma_c) - rho * gaussian(t, sigma_s);
}

void one_dimensional_DOG(const cv::Mat& ngrad_x, const cv::Mat& ngrad_y, cv::Mat& fog_x, const cv::Mat& fog_y){

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
				int temp = CLIP3((b + (cN + cS + cE + cW)), 0.0f, 255.0f);
				/*cout << temp << endl;
				cout << pSrc[0] << endl;*/
				pSrc[0] = (int)(CLIP3((b + (cN + cS + cE + cW)), 0.0f, 255.0f));
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
				pSrc[1] = (int)(CLIP3((g + (cN + cS + cE + cW)), 0.0f, 255.0f));

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
				pSrc[2] = (int)(CLIP3((r + (cN + cS + cE + cW)), 0.0f, 255.0f));
				pSrc += channel;
			}
		}
	}
	free(grayData);
}

