#include <iostream>
#include <vector>
#include <tuple>
#include <numeric>
#include <cmath>
#include <limits>

class LinearRegressionModel
{
	bool isConverged();
	void computeGradients(std::vector<double>& gradients);
	void takeStep(double step, std::vector<double>& gradients);
	std::vector<double> m_xVals;
	std::vector<double> m_yVals;
	double m_num_elems;
    double m_old_error;
	double m_a;
	double m_b;

public:
	LinearRegressionModel(std::vector<double>& xVals, std::vector<double> yVals);
	void train(int iterationsToDo, double aInitialValue, double bInitialValue);
	
	double regress(double x);
};