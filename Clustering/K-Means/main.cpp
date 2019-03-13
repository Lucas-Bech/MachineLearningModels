#include "KMeans.h"

int main()
{
    std::vector<std::pair<double, double>> data =
    {
        { 1.1, 1.0 },
        { 1.4, 2.0 },
        { 3.8, 7.0 },
        { 5.9, 8.0 },
        { 4.3, 6.0 },
        { 8.0, 5.0 },
        { 6.0, 8.5 },
        { 3.0, 2.0 },
        { 9.0, 6.0 },
        { 9.1, 4.0 }
    };

    KMeans model(data, 3);

    std::valarray<std::pair<double, double>> initial_means = 
    {
        { 1.0, 1.0 },
        { 3.0, 4.0 },
        { 8.0, 8.0 }
    };

    model.clusterData(initial_means, 3);

    model.printClusters();
    return 0;
}