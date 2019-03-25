import numpy as np
import matplotlib
import matplotlib.pyplot as plot
from sklearn.datasets.samples_generator import make_blobs

def get_distances(sample, dataset):
    distances = np.zeros(len(dataset))

    # Populate output array 'distances' with the distances
    # to all other samples in the dataset
    for i in range(0,len(dataset)):
        euclidean_distance = np.sqrt(sum(pow(np.subtract(sample,dataset[i]),2)))
        distances[i] = euclidean_distance
    return distances

def add_sample(newsample, data, features):
    distances = np.zeros((len(data),len(data[0])))
    
    # Calculate the distances between the new sample and the current data
    distances = get_distances(newsample, data)
    closest_3_neighbors = np.argpartition(distances, 3)[:3]
    closest_3_groups = features[closest_3_neighbors]
    
    return np.argmax(np.bincount(closest_3_groups))

def knn (newdata, data, features):
    for i in newdata:
        test = add_sample(i, data, features);
        features = np.append(features, [test], axis = 0)
        data = np.append(data, [i], axis = 0)
    return data,features

data, features = make_blobs(
        n_samples = 100, 
        n_features = 2, 
        centers = 4, 
        shuffle = True, 
        cluster_std = 0.8
        )

figure, axes = plot.subplots()
axes.set_title('Prior to adding new samples')
axes.scatter(
        data.transpose()[0], 
        data.transpose()[1], 
        c = features,
        marker = 'o', 
        s = 100
        )

# Add new samples and plot
newsamples = np.random.rand(20, 2) * 20 - 8. 
finaldata, finalfeatures = knn(newsamples, data, features)

figure, axes = plot.subplots()
axes.set_title('After adding new samples')
axes.scatter(finaldata.transpose()[0], finaldata.transpose()[1],
c = finalfeatures, marker = 'o', s = 100)
axes.scatter(
        newsamples.transpose()[0], 
        newsamples.transpose()[1],
        c='none',
        marker = 's',
        s = 100 )

plot.show()
