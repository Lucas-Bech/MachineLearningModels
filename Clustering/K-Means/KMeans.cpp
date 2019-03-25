#include "KMeans.h"

KMeans::KMeans(std::vector<std::pair<double, double>>& data, int k)
    : m_k(k), m_means(k), m_data(k)
{
    m_data[0] = data;
}

bool KMeans::computeMeans()
{
    bool res = true;
    for (int i = 0; i < m_k; ++i)
    {
        std::pair<double, double> mean(0, 0);
        int num_features_of_k = m_data[i].size();
        for (auto& i : m_data[i])
        {
            std::get<0>(mean) += std::get<0>(i);
            std::get<1>(mean) += std::get<1>(i);
        }
        std::get<0>(mean) /= num_features_of_k;
        std::get<1>(mean) /= num_features_of_k;
        res = (m_means[i] == mean && res == true) ? true : false;
        m_means[i] = mean;
        std::cout << "Cluster centroid " << i << ":\tx " << std::get<0>(mean) << ", y " << std::get<1>(mean) << "\t";
    }
    return res;
}

void KMeans::assignLabels()
{
    std::valarray<std::vector<std::pair<double, double>>> new_data(m_k);
    for (auto& clust : m_data)
        for (auto& feature : clust)
        {
            new_data[computeClosestCentroid(feature)].push_back(feature);
        }
    m_data = new_data;
}

int KMeans::computeClosestCentroid(const std::pair<double, double>& point)
{
    std::valarray<double> distances(m_k);
    for (int i = 0; i < m_k; ++i)
    {
        double del_x = std::get<0>(point) - std::get<0>(m_means[i]);
        double del_y = std::get<1>(point) - std::get<1>(m_means[i]);
        double dist = sqrt(pow(del_x, 2) + pow(del_y, 2));
        distances[i] = dist;
    }
    auto closest_mean = std::distance(std::begin(distances), std::min_element(std::begin(distances), std::end(distances)));
    return closest_mean;
}

void KMeans::clusterData(std::valarray<std::pair<double, double>>& initialMeans, int iterations)
{
    m_means = initialMeans;
    assignLabels();

    for (int i = 0; i < iterations; ++i)
    {
        std::cout << "Running iteration: " << i << "\r\n";
        computeMeans();
        assignLabels();
    }
}
void KMeans::printClusters() const
{
    for (int i = 0; i < m_k; ++i)
    {
        for (auto& it : m_data[i])
        {
            std::cout << " [" << std::get<0>(it) << " , " << std::get<1>(it) << "] ";
        }
        std::cout << std::endl;
    }
}