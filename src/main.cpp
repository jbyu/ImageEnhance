
#include "ImageEnhance.h"

using namespace cv;
using namespace std;


int main(int argc, char**argv)
{
	if (2 > argc) return -1;

    vector<double> sigemas;
    vector<double> weights;
	for (int i = 0; i < 3; i++)
	{
		weights.push_back(1.f / 3);
	}
    sigemas.push_back(30);
    sigemas.push_back(150);
    sigemas.push_back(300);

	bool show = (6 > argc);
	
	Mat img = imread(argv[1]);
	if (show) imshow("Source", img);

	Mat out1 = multi_scale_retinex(img, weights, sigemas, 128, 128);
	if (show) imshow("MSR", out1);
	if (2 < argc) imwrite(argv[2], out1);

	Mat out2 = adaptive_logarithmic_mapping(img);
	if (show) imshow("HDR", out2);
	if (3 < argc) imwrite(argv[3], out2);

	Mat out3 = ALTM_retinex(img);
	if (show) imshow("ALTM", out3);
	if (4 < argc) imwrite(argv[4], out3);

	if (show) char key = (char) cvWaitKey(0);

    return 0;
}
