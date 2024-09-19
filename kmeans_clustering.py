import numpy as np
import argparse
import scipy.io
from sklearn.cluster import KMeans

class MykmeansClustering:
    def __init__(self, dataset_file):
        self.model = None
        self.data = None
        self.dataset_file = dataset_file
        self.read_mat()

    def read_mat(self):
        mat = scipy.io.loadmat(self.dataset_file)
        self.data = mat['X']
        
    def model_fit(self):
        '''
        initialize self.model here and execute kmeans clustering here
        '''
        self.model = KMeans(n_clusters=3)
        self.model.fit(self.data)
        
        cluster_centers = self.model.cluster_centers_
        return cluster_centers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kmeans clustering')
    parser.add_argument('-d','--dataset_file', type=str, default = "dataset_q2.mat", help='path to dataset file')
    args = parser.parse_args()
    classifier = MykmeansClustering(args.dataset_file)
    clusters_centers = classifier.model_fit()
    print(clusters_centers)
    