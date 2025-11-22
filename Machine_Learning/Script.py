import os.path
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.datasets import load_digits

# TODO: Add necessary imports here
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.filters import sobel

# The lines below shall not be modified!
class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute an average Sobel estimator on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self # No fitting needed for this processing
    
    def transform(self, X):
        sobel_feature = np.array([np.mean(sobel(img.reshape((8,8)))) for img in X]).reshape(-1, 1)
        return sobel_feature

class ZonalInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute zone information on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion

       TODO: Continue this work
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self # No fitting needed for this processing
    
    def transform(self, X):
        zone_features=[]
        for image in X:
            img=image.reshape((8, 8))
            top=img[0:3, :]
            middle=img[3:5, :]
            bottom=img[5:8, :]
            features = [np.mean(top),np.mean(middle),np.mean(bottom)]
            zone_features.append(features)
        return np.array(zone_features)

class PCAFeatureExtractor(BaseEstimator, TransformerMixin):
    '''Extract 3 principal components using PCA'''
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)
    
# The following will be replaced by our own 
if os.path.isfile("test_data.npy"):
    X_test = np.load("test_data.npy")
    y_test = np.load("test_labels.npy")
    X_train, y_train = load_digits(return_X_y=True)
else:
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO: Prepare your learning pipeline and set all the parameters for your final algorithm.
all_features = FeatureUnion([('pca', PCAFeatureExtractor(n_components=20)),('sobel', EdgeInfoPreprocessing()), ('zones', ZonalInfoPreprocessing())])
clf = Pipeline([('features', all_features),('scaler', StandardScaler()), ('classifier', SVC(kernel='rbf', C=100.0, gamma=0.001))])
#C=100 et gamma=0.001 sont les meilleurs paramètres que l'on a trouvé

# The next lines shall not be modified
clf.fit(X_train, y_train)
print(f"Score on the test set {clf.score(X_test, y_test)}")