#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

static void gaussian_filter(cv::Mat &img, double sigma)
{
	int filter_size;

	// Reject unreasonable demands
	if (sigma > 300) sigma = 300;

	// get needed filter size (enforce oddness)
	filter_size = (int)floor(sigma * 6) / 2;
	filter_size = filter_size * 2 + 1;

	// If 3 sigma is less than a pixel, why bother (ie sigma < 2/3)
	if (filter_size < 3) return;

	// Filter
	GaussianBlur(img, img, cv::Size(filter_size, filter_size), 0);
}

cv::Mat multi_scale_retinex(const cv::Mat &img, const std::vector<double>& weights, const std::vector<double>& sigmas, int gain, int offset)
{
	cv::Mat fA, fB, fC;

	img.convertTo(fB, CV_32FC3);
	cv::log(fB, fA);

	// Normalize according to given weights
	double weight = 0;
	size_t num = weights.size();
	for (int i = 0; i < num; i++)
		weight += weights[i];

	if (weight != 1.0f)
		fA *= weight;

	// Filter at each scale
	for (int i = 0; i < num; i++)
	{
		cv::Mat blur = fB.clone();
		gaussian_filter(blur, sigmas[i]);
		cv::log(blur, fC);

		// Compute weighted difference
		fC *= weights[i];
		fA -= fC;
	}
#if 0
	// Color restoration
	float restoration_factor = 6;
	float color_gain = 2;
	cv::normalize(fB, fC, restoration_factor, cv::NORM_L1);
	cv::log(fC+1, fC);
	cv::multiply(fA, fC, fA, color_gain);
#endif
	// Restore
	Mat result = (fA * gain) + offset;
	result.convertTo(result, CV_8UC3);
	return result;
}

cv::Mat GuidedFilter(cv::Mat& I, cv::Mat& p, int r, float eps)
{
#define _cv_type_	CV_32FC1
	/*
	% GUIDEDFILTER   O(N) time implementation of guided filter.
	%
	%   - guidance image: I (should be a gray-scale/single channel image)
	%   - filtering input image: p (should be a gray-scale/single channel image)
	%   - local window radius: r
	%   - regularization parameter: eps
	*/

	cv::Mat _I;
	I.convertTo(_I, _cv_type_);
	I = _I;

	cv::Mat _p;
	p.convertTo(_p, _cv_type_);
	p = _p;

	//[hei, wid] = size(I);  
	int hei = I.rows;
	int wid = I.cols;

	r = 2 * r + 1;//因为opencv自带的boxFilter（）中的Size,比如9x9,我们说半径为4 

	//mean_I = boxfilter(I, r) ./ N;  
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, _cv_type_, cv::Size(r, r));

	//mean_p = boxfilter(p, r) ./ N;  
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, _cv_type_, cv::Size(r, r));

	//mean_Ip = boxfilter(I.*p, r) ./ N;  
	cv::Mat mean_Ip;
	cv::boxFilter(I.mul(p), mean_Ip, _cv_type_, cv::Size(r, r));

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.  
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;  
	cv::Mat mean_II;
	cv::boxFilter(I.mul(I), mean_II, _cv_type_, cv::Size(r, r));

	//var_I = mean_II - mean_I .* mean_I;  
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;     
	cv::Mat a = cov_Ip / (var_I + eps);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;  
	cv::Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;  
	cv::Mat mean_a;
	cv::boxFilter(a, mean_a, _cv_type_, cv::Size(r, r));

	//mean_b = boxfilter(b, r) ./ N;  
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, _cv_type_, cv::Size(r, r));

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;  
	cv::Mat q = mean_a.mul(I) + mean_b;

	return q;
}

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

cv::Mat ALTM_retinex(const cv::Mat& img, bool LocalAdaptation = false, bool ContrastCorrect = true)
{
	// Adaptive Local Tone Mapping Based on Retinex for HDR Image
	// https://github.com/IsaacChanghau/OptimizedImageEnhance

	const int cx = img.cols / 2;
	const int cy = img.rows / 2;
	Mat temp, Lw;

	img.convertTo(temp, CV_32FC3);
	cvtColor(temp, Lw, CV_BGR2GRAY);

	// global adaptation 
	double LwMax;
	cv::minMaxLoc(Lw, NULL, &LwMax);

	Mat Lw_;
	const int num = img.rows * img.cols;
	cv::log(Lw + 1e-3f, Lw_);
	float LwAver = exp(cv::sum(Lw_)[0] / num);

	// globally compress the dynamic range of a HDR scene we use the following function in(4) presented in[5].
	Mat Lg;
	cv::log(Lw / LwAver + 1.f, Lg);
	cv::divide(Lg, log(LwMax / LwAver + 1.f), Lg);

	// local adaptation 
	Mat Lout;
	if (LocalAdaptation) {
		int kernelSize = floor(std::max(3, std::max(img.rows / 100, img.cols / 100)));
		Mat Lp, kernel = cv::getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
		cv::dilate(Lg, Lp, kernel);
		Mat Hg = GuidedFilter(Lg, Lp, 10, 0.01f);

		double eta = 36;
		double LgMax;
		cv::minMaxLoc(Lg, NULL, &LgMax);
		Mat alpha = 1.0f + Lg * (eta / LgMax);

		Mat Lg_;
		cv::log(Lg + 1e-3f, Lg_);
		float LgAver = exp(cv::sum(Lg_)[0] / num);
		float lambda = 10;
		float beta = lambda * LgAver;

		cv::log(Lg / Hg + beta, Lout);
		cv::multiply(alpha, Lout, Lout);
		cv::normalize(Lout, Lout, 0, 255, NORM_MINMAX);
	}else {
		cv::normalize(Lg, Lout, 0, 255, NORM_MINMAX);
	}

#if 0
	Mat yuv;
	cvtColor(temp, yuv, CV_BGR2YUV);
	Mat yuv_[3];
	cv::split(yuv, yuv_);
	yuv_[0] = Lout;
	cv::merge(yuv_, 3, yuv);
	cvtColor(yuv, yuv, CV_YUV2BGR);
	yuv.convertTo(yuv, CV_8UC3);
	imshow("yuv",yuv);
	std::cout << "temp:" << temp.at <Vec3f>(cy, cx) << std::endl;
	std::cout << "Lg:" << Lg.at <float>(cy, cx) << std::endl;
	std::cout << "Lw:" << Lw.at <float>(cy, cx) << std::endl;
	std::cout << "Lout:" << Lout.at <float>(cy, cx) << std::endl;
#endif

	Mat gain(img.rows , img.cols, CV_32F);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			float x = Lw.at<float>(i, j);
			float y = Lout.at<float>(i, j);
			if (0 == x) gain.at<float>(i, j) = y;
			else gain.at<float>(i, j) = y / x;
		}
	}
	
	Mat out;
	Mat bgr[3];
	cv::split(temp, bgr);
	if (ContrastCorrect) {
		// Contrast image correction method
		// https://www.researchgate.net/publication/220051147_Contrast_image_correction_method
		bgr[0] = (gain.mul(bgr[0] + Lw) + bgr[0] - Lw) *0.5f;
		bgr[1] = (gain.mul(bgr[1] + Lw) + bgr[1] - Lw) *0.5f;
		bgr[2] = (gain.mul(bgr[2] + Lw) + bgr[2] - Lw) *0.5f;
	} else {
		cv::multiply(bgr[0], gain, bgr[0]);
		cv::multiply(bgr[1], gain, bgr[1]);
		cv::multiply(bgr[2], gain, bgr[2]);
	}

	cv::merge(bgr, 3, out);
	out.convertTo(out, CV_8UC3);
	return out;
}

cv::Mat adaptive_logarithmic_mapping(const cv::Mat& img)
{
	// Adaptive Logarithmic Mapping For Displaying High Contrast Scenes
	// http://resources.mpi-inf.mpg.de/tmo/logmap/
	Mat ldrDrago, result;
	img.convertTo(ldrDrago, CV_32FC3, 1.0f/255);
	cvtColor(ldrDrago, ldrDrago, cv::COLOR_BGR2XYZ);
	Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.f, 1.f, 0.85f);
	tonemapDrago->process(ldrDrago, result);
	cvtColor(result, result, cv::COLOR_XYZ2BGR);
	result.convertTo(result, CV_8UC3, 255);
    return result;
}
