#include <iostream>
#include <valarray>
#include <algorithm>
#include <vector>
#include <utility>
#include <cmath>

class KMeans
{
    int m_k;
    int m_features;
    std::valarray<std::pair<double, double>> m_means;
    std::valarray<std::vector<std::pair<double, double>>> m_data;
    
    bool computeMeans();
    void assignLabels();
    int computeClosestCentroid(const std::pair<double, double>& point);

public:
    KMeans(std::vector<std::pair<double, double>>& data, int k);
    void clusterData(std::valarray<std::pair<double, double>>& initialMeans, int iterations);
    void printClusters() const;
};