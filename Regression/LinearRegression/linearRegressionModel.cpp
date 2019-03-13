#include "linearRegressionModel.h"
#include <vector>

LinearRegressionModel::LinearRegressionModel(std::vector<double>& xVals, std::vector<double> yVals)
		: m_xVals(xVals), m_yVals(yVals), m_num_elems(yVals.size()), m_old_error(std::numeric_limits<double>::max()){}

void LinearRegressionModel::computeGradients(std::vector<double>& gradients)
{
	// Compute gradient of error with respect to a
	for (int i = 0; i < m_num_elems; ++i)
		gradients[0] += m_xVals[i] * ((m_a * m_xVals[i] + m_b)) - m_yVals[i];
	gradients[0] /= m_num_elems * 0.5;
	
	// Compute gradient of error with respect to b
	for (int i = 0; i < m_num_elems; ++i)
		gradients[1] += ((m_a * m_xVals[i] + m_b)) - m_yVals[i];
	gradients[1] /= m_num_elems * 0.5;
}

void LinearRegressionModel::takeStep(double step, std::vector<double>& gradients)
{
	m_a -= step * gradients[0];
	m_b -= step * gradients[1];
}

void LinearRegressionModel::train(int maxIterations, double aInitialValue, double bInitialValue)
{
	m_a = aInitialValue;
	m_b = bInitialValue;

	for (int currentIteration = 0; !isConverged() && currentIteration < maxIterations; ++currentIteration)
	{
		double step = 2.0 / static_cast<double>(currentIteration + 2);

		std::vector<double> gradients = { 0, 0 };

		computeGradients(gradients);
		takeStep(step, gradients);

		std::cout
			<< "Coefficient A:\t" << m_a
			<< "\nCoefficient B:\t" << m_b
			<< "\nGradient A:\t" << gradients[0]
			<< "\nGradient B:\t" << gradients[1]
			<< "\n";
	}
}

bool LinearRegressionModel::isConverged()
{
	double error = 0;
	double threshold = 0.001;
	for(int i = 0; i < m_num_elems; ++i)
		error += pow(m_a * m_xVals[i] + m_b - m_yVals[i], 2);
	error /= m_num_elems;
	std::cout << "Error: " << error << "\r\n";
	
	bool res = (abs(error) < m_old_error - threshold && abs(error) > m_old_error + threshold);
	m_old_error = abs(error);
	return res;
}

double LinearRegressionModel::regress(double x)
{
	return m_a * x + m_b;
}