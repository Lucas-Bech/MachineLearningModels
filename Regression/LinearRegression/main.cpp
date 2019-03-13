#include "./linearRegressionModel.h"

int main(int argc, char** argv)
{
	std::vector<double> x({1, 2, 3, 4, 5});
	std::vector<double> y({2.8, 2.9, 7.6, 9, 8.6});

	LinearRegressionModel model(x, y);
	model.train(1000, 3, -10);

	std::cout << model.regress(3);

	return 0;
}