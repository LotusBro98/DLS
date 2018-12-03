#include <iostream>

#include <dls.h>

int main()
{
	float G;

	while (true)
	{
		std::cin >> G;

		auto vec = dls::genSpectrum(G);

		G = dls::recognizeGamma(vec);

		std::cout << G << std::endl;

		delete vec;
	}

	return 0;
}
