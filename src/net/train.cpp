//
// Created by alex on 29.10.18.
//

#include <iostream>
#include <fstream>
#include <string>
#include <list>

#include <unistd.h>

#include <simplenet/Net.h>

using namespace std;

int main(int argc, char * argv[])
{	
	if (argc < 2)
	{
		cerr << "Usage:	./train <dataset> [options]" << endl
			 << "Options:" << endl
			 << "-L <trainLoss>" << endl
		//	 << "-t threads" << endl
			 << "-n nNeurons -- add hidden layer with nNeurons" << endl;
		exit(1);
	}

    ifstream file(argv[1]);

	if (!(file.good()))
	{
		cerr << "No such dataset file: \"" << argv[1] << "\"" << endl;
		exit(1);
	}

	sn::Dataset * dataset;
	try {
		dataset = new sn::Dataset(file);
	} catch (exception e) {
		cerr << "Failed to read dataset from " << argv[1] << endl;
		exit(1);
	}

	float trainLoss = 0.05;
	int nHidden = 0;
	int nThreads = 1;
	std::list<int> nNeuronsList;


	int opt;
	while ((opt = getopt(argc, argv, "L:t:n:")) != -1)
	{
		try {
			switch (opt)
			{
				case 'L':
					trainLoss = stof(optarg);
					break;
				case 'n':
					nNeuronsList.push_back(stoi(optarg));
					break;
				case 't':
					nThreads = stoi(optarg);
					break;
				default:
					cerr << "Unrecognized option: -" << (char)opt << endl;
					exit(1);
					break;
			}
		} catch (exception e) {
			cerr << "Failed to parse option -" << (char)opt << " from \"" << optarg << "\""<< endl;
			exit(1);
		}
	}

	if (optind != argc - 1)
	{
		cerr << "Too many arguments." << endl;
		exit(-1);
	}

	nHidden = nNeuronsList.size();
	int * nNeurons = (nHidden == 0 ? nullptr : new int[nHidden]);
	for (int i = 0; i < nHidden; i++)
	{
		nNeurons[i] = nNeuronsList.front();
		nNeuronsList.pop_front();
	}


	//--------------------------

	srand(time(NULL));

    sn::Net * net = new sn::Net(dataset, nHidden, nNeurons, sn::sigmoid, sn::sqrDiffNorm, sn::optimizeWeightNewton, nThreads);

    float L = net->train(dataset, trainLoss);
    
	cout << net;

	delete nNeurons;

    return 0;
}
