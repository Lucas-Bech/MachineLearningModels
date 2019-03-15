#include "./linearRegressionModel.h"

int main()
{
	std::vector<Point> trainingDataset = 
	{
		{ 1.0, 2.8 },
		{ 2.0, 2.9 },
		{ 3.0, 7.6 },
		{ 4.0, 9.0 },
		{ 5.0, 8.6 }
	};

	LinearRegressionModel model(trainingDataset);
	model.train(2000, 1, -1);

	std::cout << "Y value when X is 3: " <<  model.regress(3) << std::endl;
	return 0;
}