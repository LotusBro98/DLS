#ifndef DLS_H
#define DLS_H

#include <vector>

#include <simplenet/Net.h>

namespace dls
{
	class Points
	{
	public:

		Points(int n);
		Points(std::vector<std::pair<float, float>> * pts);

		std::vector<std::pair<float, float>> * toVec();

		void set(int i, float x, float y);
		float getX(int i);
		float getY(int i);
		int nPts();

		float * getYs();
		float * getXs();

		~Points();

	private:
		float * xs;
		float * ys;
		int n;
	};

	float LorentzFunc(float G, float x);
}

#endif
