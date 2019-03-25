#include "linearRegressionModel.h"
#include <vector>

Point::Point(double x, double y)
	: x(x), y(y) {}

LinearRegressionModel::LinearRegressionModel(std::vector<Point>& points)
		: m_points(points), m_old_error(std::numeric_limits<double>::max()) {}

std::vector<double> LinearRegressionModel::makePredictions()
{
	std::vector<double> predictions;
	for (Point point : m_points)
	{
		double prediction = regress(point.x);
		predictions.push_back(prediction);
	}
	return predictions;
}

std::vector<double> LinearRegressionModel::calculateErrors(std::vector<double>& predictions)
{
	std::vector<double> errors;
	for (unsigned int i = 0; i < m_points.size(); ++i)
		errors.emplace_back(predictions[i] - m_points[i].y);
	return errors;
}

// Changes the coefficients to better fit the line
void LinearRegressionModel::updateCoefficients(std::vector<double>& predictions, std::vector<double>& errors)
{
	for (unsigned int i = 0; i < predictions.size(); ++i)
	{
		m_a -= learningRate * errors[i] * m_points[i].x;
		m_b -= learningRate * errors[i];
	}
}

void LinearRegressionModel::train(int iterations, double aInitialValue, double bInitialValue)
{
	m_a = aInitialValue;
	m_b = bInitialValue;

	for (int currentIteration = 0; !isConverged() && currentIteration < iterations; ++currentIteration)
		for (unsigned int i = 0; i < m_points.size(); ++i)
		{
			std::vector<double> predictions = makePredictions();
			std::vector<double> errors = calculateErrors(predictions);
			updateCoefficients(predictions, errors);
		}
}

bool LinearRegressionModel::isConverged()
{
	double error = 0;
	double threshold = 0.000002;

	for(unsigned int i = 0; i < m_points.size(); ++i)
		error += pow(m_a * m_points[i].x + m_b - m_points[i].y, 2);
	error /= m_points.size();
	std::cout << "Error: " << error << "\r\n";

	bool res = (fabs(error) > m_old_error - threshold && fabs(error) < m_old_error + threshold);
	m_old_error = fabs(error);

	return res;
}

double LinearRegressionModel::regress(double x)
{
	return m_a * x + m_b;
}