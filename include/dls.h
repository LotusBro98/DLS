#ifndef DLS_H
#define DLS_H

#include <vector>

#include <simplenet/Net.h>

namespace dls
{	
	float LorentzFunc(float G, float x);
	void normalizeSpectrum(float * spectrum, int n);


	std::vector<std::pair<float, float>> * genSpectrum(float G);

	float recognizeGamma(std::vector<std::pair<float, float>> * spectrum);
}

#endif
