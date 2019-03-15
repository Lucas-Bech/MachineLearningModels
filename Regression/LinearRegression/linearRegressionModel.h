#include <iostream>
#include <vector>
#include <tuple>
#include <numeric>
#include <cmath>
#include <limits>

struct Point
{
	double x;
	double y;

	Point(double _x, double _y);	
};

class LinearRegressionModel
{
	std::vector<Point> m_points;
	double learningRate = 0.0015;
    double m_old_error;
	double m_a;
	double m_b;

	std::vector<double> makePredictions();
	std::vector<double> calculateErrors(std::vector<double>& predictions);
	void updateCoefficients(std::vector<double>& predictions, std::vector<double>& errors);
	bool isConverged();

public:
	LinearRegressionModel(std::vector<Point>& points);
	void train(int iterationsToDo, double aInitialValue, double bInitialValue);
	
	double regress(double x);
};