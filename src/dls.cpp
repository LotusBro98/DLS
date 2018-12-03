#include <cmath>

#include <dls.h>
#include <fstream>

namespace dls
{
//----------------------------------------

	float LorentzFunc(float G, float x)
	{
		return 1 / M_PI * G / (x * x + G * G);
	}

	std::vector<std::pair<float, float>> * genSpectrum(float G)
	{
		auto points = new std::vector<std::pair<float, float>>();

		float xMin = 0;
		float xMax = 1;
		int n = 33;

		for (int i = 0; i < n; i++)
		{
			float x = xMin + i * (xMax - xMin) / n;
			float y = dls::LorentzFunc(G, x);
			points->push_back(std::pair<float, float>(x, y));
		}

		return points;
	}

	void normalizeSpectrum(float * spectrum, int n)
	{
		float y0 = spectrum[0];
		for (int i = 0; i < n; i++)
			spectrum[i] /= y0;
	}

	float recognizeGamma(std::vector<std::pair<float, float>> * spectrumVec)
	{
		float spectrum[33];

		int i = 0;
		for (std::pair<float, float> pt : *spectrumVec)
		{
			spectrum[i] = pt.second;
			i++;
		}

		normalizeSpectrum(spectrum, 33);

		std::ifstream file("net.txt");
		sn::Net * net = new sn::Net(file, sn::sigmoid);

		return *net->process(spectrum);
	}
	
}
