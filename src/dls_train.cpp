#include <cmath>

#include <opencv2/opencv.hpp>

#include <dls.h>

#include <simplenet/Net.h>

void showPoints(dls::Points * pts,
		float xMin = 0, float xMax = 1, 
		float yMin = 0, float yMax = 1, 
		int sizeX = 500, int sizeY = 500 
		)
{
	
	const cv::Vec3b color = cv::Vec3b(0,0,255);
	cv::Mat M(sizeY, sizeX, CV_8UC3);
	cv::floodFill(M, cv::Point(0,0), cv::Vec3b(255,255,255));
	float yPrev;
	float xPrev;
	float y = pts->getY(0);
	float x = pts->getX(0);
	for (int pt = 0; pt < pts->nPts(); pt++)
	{
			xPrev = x;
			x = pts->getX(pt);
			
			yPrev = y;
			y = pts->getY(pt);

			float i = (yMax - y) / (yMax - yMin) * sizeY;
			float i0 = (yMax - yPrev) / (yMax - yMin) * sizeY;
			float j = (x - xMin) / (xMax - xMin) * sizeX;
			float j0 = (xPrev - xMin) / (xMax - xMin) * sizeX;
			cv::Point from = cv::Point(j0, i0);
			cv::Point to = cv::Point(j, i);

			cv::line(M, from, to, color, 2);
	}

	cv::imshow("Plot", M);

	cv::waitKey(100);
}

void showDistribution(sn::Net * net, sn::Dataset * dataset = nullptr,
		float xMin = -1, float xMax = 1, 
		float yMin = -1, float yMax = 1, 
		int nX = 50, int nY = 50,
		int sizeX = 500, int sizeY = 500 
		)
{
	if (net != nullptr)
		if (net->getNIn() != 2 || net->getNOut() != 1)
			return;

	cv::Mat M0(nY, nX, CV_8UC3);
	if (net != nullptr)
		for (int i = 0; i < nY; i++)
			for (int j = 0; j < nX; j++)
			{
				float x = xMin + j * (xMax - xMin) / nX;
				float y = yMin + (nY - i) * (yMax - yMin) / nY;
				float xx[2] = {x, y};
				float p = *(net->process(xx));
				//p = 0.5 * (1 + p);
				M0.at<cv::Vec3b>(i, j) = cv::Vec3b::all(p * 255);
			}

	cv::Mat M(sizeY, sizeX, CV_8UC3);
	cv::resize(M0, M, cv::Size2i(sizeX, sizeY), 0, 0, cv::INTER_NEAREST);

	if (dataset == nullptr)
		goto draw;

	for (int point = 0; point < dataset->getNPoints(); point++)
	{
		float x = dataset->getPointIn(point)[0];
		float y = dataset->getPointIn(point)[1];
		int j =    (x - xMin) / (xMax - xMin)  * sizeX;
		int i = (1-(y - yMin) / (yMax - yMin)) * sizeY;
		const cv::Vec3b color = dataset->getPointOut(point)[0] > 0.5f ? cv::Vec3b(0,0,255) : cv::Vec3b(255,0,0);
		cv::circle(M, cv::Point(j, i), 5, color, 2);
	}

	draw:

	cv::imshow("Distribution", M);

	cv::waitKey(1);
}


float noact(float x)
{
	return x;
}


void genMixedSpectrum(float * spectrum, float * amounts)
{
	float xMin = 0;
	float xMax = 1;
	int nx = 100;

	int nGs = 1;
	//float gammas[15] = {0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.75, 0.80, 0.85, 0.90, 0.95, 1};

	for (int i = 0; i < nx; i++)
	{
		float x = xMin + i * (xMax - xMin) / nx;
		float y = 0;
		for (int j = 0; j < nGs; j++)
		{
			//float G = gammas[j];
			//float n = amounts[j];
			float G = amounts[j];
			float n = 1;
			y += n * dls::LorentzFunc(G, x);
		}
		spectrum[i] = y;
	}
}

float relativeSqrDiff(float * vec1, float * vec2, int n)
{
	float L = 0;
	for (int i = 0; i < n; i++)
	{
		float d = 2 * std::abs(vec1[i] - vec2[i]) / (vec1[i] + vec2[i]);
		L += d * d;
	}

	return L / n;
}

template float sn::customMeanNorm<relativeSqrDiff>(Dataset*, Dataset*);

int main()
{
	sn::Dataset * dataset = new sn::Dataset(genMixedSpectrum, 100, 1, 0.01, 1, 100, false);


	sn::Net * net = new sn::Net(dataset, 1, new int[2]{10, 5}, sn::sigmoid, sn::customMeanNorm<relativeSqrDiff>, sn::optimizeWeightNewton);

	net->train(dataset, 0.0002);

	float spectrum[100];
	int nPatricles = 15;
	//float amounts[15] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	float gammas[1] = {0.5};

	genMixedSpectrum(spectrum, gammas);

	for (float gamma = 0.1; gamma < 1; gamma += 0.1)
	{
		gammas[0] = gamma;
		genMixedSpectrum(spectrum, gammas);
		net->process(spectrum, gammas);
		std::cerr << gamma << "-->" << gammas[0] << "\n";
	}

	std::cout << net;

	return 0;
}
