#include <cmath>

#include <dls.h>

namespace dls
{
	Points::Points(int n)
	{
		this->xs = new float[n];
		this->ys = new float[n];
		this->n = n;
	}

	Points::Points(std::vector<std::pair<float, float>> * pts)
	:Points(pts->size())
	{
		int i = 0;
		for (auto pt : *pts)
		{
			xs[i] = pt.first;
			ys[i] = pt.second;
			i++;
		}
	}

	std::vector<std::pair<float, float>> * Points::toVec()
	{
		auto vec = new std::vector<std::pair<float,float>>();

		for (int i = 0; i < n; i++)
			vec->push_back(std::pair<float,float>(xs[i], ys[i]));

		return vec;
	}

	void Points::set(int i, float x, float y)
	{
		xs[i] = x;
		ys[i] = y;
	}

	float Points::getX(int i)
	{
		return xs[i];
	}

	float Points::getY(int i)
	{
		return ys[i];
	}

	int Points::nPts()
	{
		return n;
	}

	float * Points::getXs()
	{
		return xs;
	}

	float * Points::getYs()
	{
		return ys;
	}

	Points::~Points()
	{
		delete [] xs;
		delete [] ys;
	}

//----------------------------------------


	float LorentzFunc(float G, float x)
	{
		return 1 / M_PI * G / (x * x + G * G);
	}

}
