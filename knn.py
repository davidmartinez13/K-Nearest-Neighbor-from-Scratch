from collections import Counter
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

def plot(X_train, X_test, y_train, y_test, predictions, k_neighbors,
        colors_train = ['orange','lime','skyblue'], colors_test = ['red','seagreen','blue']):
    labels = ['Iris-Setosa','Iris-Versicolour','Iris-Virginica']

    xlabel = 'X sepal length cm'
    ylabel = 'Y sepal width in cm'
    zlabel = 'Z petal length in cm'

    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Iris plants dataset \n Sphere radius : petal width in cm')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0] = fig.add_subplot(121, projection='3d')
    ax[1] = fig.add_subplot(122, projection='3d')

    ax[0].scatter(X_train[:,0], X_train[:,1], X_train[:,2], 
                    s=np.pi*X_train[:,3]**2*10, 
                    c =y_train, 
                    cmap=matplotlib.colors.ListedColormap(colors_train), 
                    alpha=0.75)

    legendhandle = [ax[0].plot([],[], marker="o", ls="", color=color)[0] for color in colors_train]
    ax[0].legend(legendhandle,labels,loc="upper right", frameon=True)
    ax[0].title.set_text('Train Data')
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_zlabel(zlabel)

    ax[1].scatter(X_test[:,0], X_test[:,1], X_test[:,2], 
                    s=np.pi*X_test[:,3]**2*10, 
                    c =predictions, 
                    cmap=matplotlib.colors.ListedColormap(colors_test), 
                    alpha=0.75)
    ax[1].scatter(X_train[:,0], X_train[:,1], X_train[:,2], 
                    s=np.pi*X_train[:,3]**2*10, 
                    c =y_train, 
                    cmap=matplotlib.colors.ListedColormap(colors_train), 
                    alpha=0.75)
    
    legendhandle = [ax[1].plot([],[], marker="o", ls="", color=color)[0] for color in colors_test]
    ax[1].legend(legendhandle,labels,loc="upper right", frameon=True)
    ax[1].title.set_text('KNN classification result.\n k:{} Accuracy:{}'.format(str(k_neighbors),str(accuracy(y_test, predictions))) )
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    ax[1].set_zlabel(zlabel)
    plt.show()

if __name__ == "__main__":
    
    #----------------Data Loading----------------------------
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    #----------------Model Fitting----------------------------
    k_neighbors = 3
    clf = KNN(k=k_neighbors)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    #----------------Plotting----------------------------
    plot(X_train, X_test, y_train, y_test, predictions, k_neighbors)