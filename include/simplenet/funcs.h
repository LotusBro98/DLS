#include "Dataset.h"

#include <cmath>

namespace sn
{
#ifndef FUNCS_H
#define FUNCS_H

	float sigmoid(float x);

    float sqrDiffNorm(Dataset * dataset1, Dataset * dataset2);
	
	template <float (*dist)(float * vec1, float * vec2, int n)> 
	float customMeanNorm(Dataset * dataset1, Dataset * dataset2)
	{
        if (dataset1->getNPoints() != dataset2->getNPoints())
            throw;

        int nPoints = dataset1->getNPoints();
        int nOut = dataset1->getNOut();
        float L = 0;
        for (int point = 0; point < nPoints; ++point) {
            float Li = dist(dataset1->getPointOut(point), dataset2->getPointOut(point), nOut);
            L += Li / nOut;
        }

        return L / nPoints;
    }


    void optimizeWeightNewton(float & weight, float L, float dw, float dL);

    void optimizeWeightGradient(float & weight, float L, float dw, float dL);

#endif
}
