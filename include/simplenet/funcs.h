#include "Dataset.h"

#include <cmath>

namespace sn
{
#ifndef FUNCS_H
#define FUNCS_H

	float sigmoid(float x);

    float sqrDiffNorm(Dataset * dataset1, Dataset * dataset2);
	
	template <float (*dist)(float * vec1, float * vec2, int n)> 
	float customMeanNorm(Dataset * dataset1, Dataset * dataset2);

    void optimizeWeightNewton(float & weight, float L, float dw, float dL);

    void optimizeWeightGradient(float & weight, float L, float dw, float dL);

#endif
}
