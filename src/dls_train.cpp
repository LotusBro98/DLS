#include <cmath>

//#include <opencv2/opencv.hpp>

#include <dls.h>

#include <simplenet/Net.h>

float noact(float x)
{
	return x;
}


template <int nx>
void genMixedSpectrum(float * spectrum, float * amounts)
{
	float xMin = 0;
	float xMax = 1;
	//int nx = 100;

	int nGs = 1;
	//float gammas[15] = {0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.75, 0.80, 0.85, 0.90, 0.95, 1};

	float y0 = dls::LorentzFunc(amounts[0], 0);

	for (int i = 0; i < nx; i++)
	{
		float x = xMin + i * (xMax - xMin) / nx;
		float y = 0;
		for (int j = 0; j < nGs; j++)
		{
			//float G = gammas[j];
			//float n = amounts[j];
			float G = amounts[j];
			float n = 1 / y0;
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
	sn::Dataset * dataset = new sn::Dataset(genMixedSpectrum<33>, 33, 1, 0.01, 1, 100, false);


	sn::Net * net = new sn::Net(dataset, 1, new int[2]{20, 5}, sn::sigmoid, sn::customMeanNorm<relativeSqrDiff>, sn::optimizeWeightNewton);

	net->train(dataset, 0.005);

	float spectrum[100];
	int nPatricles = 15;
	//float amounts[15] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	float gammas[1] = {0.5};

	genMixedSpectrum<33>(spectrum, gammas);

	for (float gamma = 0.1; gamma < 1; gamma += 0.1)
	{
		gammas[0] = gamma;
		genMixedSpectrum<33>(spectrum, gammas);
		net->process(spectrum, gammas);
		std::cerr << gamma << "-->" << gammas[0] << "\n";
	}

	std::cout << net;

	return 0;
}
