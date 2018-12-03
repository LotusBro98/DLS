#include <simplenet/Net.h>

//#include <opencv2/opencv.hpp>

/*
void showDistribution(sn::Net * net, sn::Dataset * dataset = nullptr,
		float xMin = -1, float xMax = 1, 
		float yMin = -1, float yMax = 1, 
		int nX = 50, int nY = 50,
		int sizeX = 500, int sizeY = 500 
		)
{
	if (net->getNIn() != 2 || net->getNOut() != 1)
		return;

	cv::Mat M0(nY, nX, CV_8UC3);
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
*/





namespace sn
{
	Net::Net(int nIn, int nOut, int nHiddenLayers, int nNeurons[],
			float (*activation)(float x),
			float (*norm)(Dataset * prediction, Dataset * experiment),
			void (*optimizeWeight)(float & weight, float L, float dw, float dL),
			int nPoints, int nThreads)
	{
		this->nLayers = nHiddenLayers + 1;
		layers = new Layer*[nLayers];

		if (nHiddenLayers == 0)
			layers[0] = new Layer(nIn, nOut, activation, nPoints, nThreads);
		else {
			layers[0] = new Layer(nIn, nNeurons[0], activation, nPoints, nThreads);

			for (int i = 1; i < nHiddenLayers; i++)
				layers[i] = new Layer(layers[i - 1], nNeurons[i]);

			layers[nHiddenLayers] = new Layer(layers[nHiddenLayers - 1], nOut);
		}

		this->norm = norm;
		this->optimizeWeight = optimizeWeight;
	}

	Net::Net(Dataset * trainset, int nHiddenLayers, int nNeurons[],
			float (*activation)(float x),
			float (*norm)(Dataset * prediction, Dataset * experiment),
			void (*optimizeWeight)(float & weight, float L, float dw, float dL),
			int nThreads)
		: Net(trainset->getNIn(), trainset->getNOut(), nHiddenLayers, nNeurons, activation, norm, optimizeWeight, trainset->getNPoints(), nThreads)
	{
	}


	Net::Net(std::istream& is, 
			float (*activation)(float x),
			float (*norm)(Dataset * prediction, Dataset * experiment),
			void (*optimizeWeight)(float & weight, float L, float dw, float dL),
			int nPoints, int nThreads)
	{
		is >> this->nLayers;
		layers = new Layer*[nLayers];

		layers[0] = new Layer(is, activation, nPoints, nThreads);
		for (int i = 1; i < nLayers; i++)
		{
			layers[i] = new Layer(is, activation, nPoints, nThreads);
			layers[i]->setInput(layers[i - 1]->getOut());
		}

		this->norm = norm;
		this->optimizeWeight = optimizeWeight;
	}

	Net::~Net()
	{
		for (int i = 0; i < nLayers; i++)
			delete layers[i];
		delete [] layers;
	}

	std::ostream& operator << (std::ostream& os, Net* net)
	{
		os << net->nLayers <<std::endl;

		for (int i = 0; i < net->nLayers; i++)
			os << net->layers[i];

		return os;
	}

	int Net::getNIn()
	{
		return layers[0]->getNIn();
	}

	int Net::getNOut()
	{
		return layers[nLayers - 1]->getNOut();
	}

	int Net::getNLayers()
	{
		return nLayers;
	}

	float * Net::process(float * input, float * output)
	{
		if (output == nullptr)
			output = layers[nLayers - 1]->getOut();

		float * outPrev = layers[nLayers - 1]->getOut();

		layers[0]->setInput(input);
		layers[nLayers - 1]->setOut(output);

		for (int i = 0; i < nLayers; i++)
			layers[i]->process();

		layers[nLayers - 1]->setOut(outPrev);

		return output;
	}

	void Net::processAll()
	{
		for (int i = 0; i < nLayers; i++)
			layers[i]->processAll();
	}

	float Net::evaluateTrain(Dataset * dataset, Dataset * calcBuffer)
	{
		int nPoints = dataset->getNPoints();

		processAll();
		
		float L = norm(dataset, calcBuffer);
		
		return L;
	}

	float Net::evaluate(Dataset * dataset)
	{
		Dataset * calcBuffer = new Dataset(dataset);

		//TODO: improve this
		for (int point = 0; point < dataset->getNPoints(); point++)
			process(dataset->getPointIn(point), calcBuffer->getPointOut(point));

		float L = norm(dataset, calcBuffer);
		
		delete calcBuffer;

		return L;
	}

	float Net::train(Dataset * dataset, float toLoss)
	{
		Dataset * calcBuffer = new Dataset(dataset);

		float dw = 0.0001;
		float Lfin = 1;

		float * outPrev = layers[nLayers - 1]->getOut();
		
		int i = 0;
		do
		{
			layers[0]->setInput(dataset->getIn());
			layers[nLayers - 1]->setOut(calcBuffer->getOut());

			for (int layer = 0; layer < nLayers; ++layer)
			{
				int nWeights = layers[layer]->getNOut() * (layers[layer]->getNIn() + 1);
				for (int i = 0; i < nWeights; ++i)
				{
					float & w = layers[layer]->getWeights()[i];
					float L0 = evaluateTrain(dataset, calcBuffer);
					w += dw;
					float L = evaluateTrain(dataset, calcBuffer);
					w -= dw;
					float dL = L - L0;
					optimizeWeight(w, L0, dw, dL);
				}
			}

			Lfin = evaluateTrain(dataset, calcBuffer);
			std::cerr << Lfin << std::endl;
			i++;
			
		//	if (this->getNIn() == 2 && this->getNOut() == 1)
		//		showDistribution(this, dataset);
		}
		while (Lfin > toLoss);
		
	//	if (this->getNIn() == 2 && this->getNOut() == 1)
	//		cv::waitKey();

		layers[nLayers - 1]->setOut(outPrev);

		delete calcBuffer;

		std::cerr << "--------------- " << i << " ---------------" << std::endl;
	
		return Lfin;
	}
}
