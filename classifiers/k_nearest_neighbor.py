import numpy as np

class KNearestNeighbor(object):
    """ a kNN classifier with Euclidean distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_vectorized(X)
        elif num_loops == 1:
            dists = self.compute_distances_with_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_with_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #####################################################################
        # TODO (15):                                                        #
        # Loop over num_test (outer loop) and num_train (inner loop) and    #
        # compute the Euclidean distance between the ith test point and the #
        # jth training point, and store the result in dists[i, j]. You      #
        # should not use a loop over dimension.                             #
        #####################################################################
        for i in range(num_test):
            for j in range(num_train):
                #solution based on:
                #http://www.codehamster.com/2015/03/09/different-ways-to-calculate-the-euclidean-distance-in-python/
                #
                #follwing calc needs much ressorces/time
                dists[i][j] = np.linalg.norm(X[i]-self.X_train[j])
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return dists

    def compute_distances_vectorized(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #######################################################################
        # TODO (30):                                                            #
        # Compute the Euclidean distance between all test points and all        #
        # training points without using any explicit loops, and store the       #
        # result in dists.                                                      #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # Hint: Try to formulate the Euclidean distance using matrix            #
        #       multiplication and two broadcast sums.                          #
        #######################################################################
        #solution based on:
        #https://stackoverflow.com/questions/27948363/numpy-broadcast-to-perform-euclidean-distance-vectorized
        aSumSquare = np.sum(np.square(X),axis=1);
        bSumSquare = np.sum(np.square(self.X_train),axis=1);
        mul = np.dot(X,self.X_train.T);
        dists = np.sqrt(aSumSquare[:,np.newaxis]+bSumSquare-2*mul)
        #######################################################################
        #                         END OF YOUR CODE                              #
        #######################################################################
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors
            # to the ith test point.
            closest_y = []

            ###################################################################
            # TODO (5):                                                             #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            ###################################################################
            dists_for_i = np.argsort(dists[i])[:k]
            y_pred[i] = 
            

            ###################################################################
            # TODO (5):                                                             #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            ###################################################################

            ###################################################################
            #                           END OF YOUR CODE                            #
            ###################################################################
        return y_pred
