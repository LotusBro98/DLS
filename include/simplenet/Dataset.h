#include <iostream>

namespace sn 
{
	#ifndef DATASET_H
	#define DATASET_H

    class Dataset
    {
    public:
        Dataset(int nPoints, int nIn, int nOut);
        
		~Dataset();
       
		float * getPointIn(int point);
        
        float * getPointOut(int point);

		float * getIn();

		float * getOut();

        int getNPoints() const;

        int getNIn() const;

        int getNOut() const;

        explicit Dataset(std::istream& is);

        explicit Dataset(Dataset * dataset);
	
		explicit Dataset(void (*func)(float * from, float * to), int nFrom, int nTo, float min = -1, float max = 1, int steps = 10, bool stepFrom = true);

        friend std::ostream& operator << (std::ostream& os, Dataset * ds);

    private:
        int nPoints;
        int nIn;
        int nOut;
        float * in;
        float * out;
    };

	#endif

}
