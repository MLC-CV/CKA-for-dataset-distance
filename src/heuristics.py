
"""
Any heuristic function should take as input the source dataset, the destination dataset, and the model. It should output a score, which is > 0. 
"""

import numpy  as np

def cka_heuristic(features_x, features_y, model):
    """
    Official code for CKA can found here:
    https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
    [TODO]: calculate CKA between ds_src, ds_dest using some output features from model
    
    jose's note1: there is a debiased method which i ommited, considering that we have # example > # features 
    jose's note1: the following is a faster computation than through Gram matrix. same results
    """ 

    #centering the features
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    #similarity
    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2

    #normalization
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    cka = dot_product_similarity / (normalization_x * normalization_y)
    return print(f"CKA calculated from feature space: {cka} of model {model}")

def intermediary_feature_moments(features):
    """for spatial feature maps"""
    spatial_mean = np.mean(features, (1,2,), keepdims=False)
    spatial_std = np.std(features, (1,2,), keepdims=False)
    return spatial_mean, spatial_std

def feature_moments(features):
    """For moments of feature vectors"""
    feature_mean = np.mean(features, 0, keepdims=True)
    feature_std = np.std(features, 0, keepdims=True)
    return feature_mean, feature_std


def contrastive_heuristic(ds_src, ds_dest, model):
    """
    Formula for contrastive loss can be found here:
        https://keras.io/examples/vision/semisupervised_simclr/
    labelled as def contrastive_loss(self, projections_1, projections_2)
    [TODO]: Calculate contrastive_loss for (1) ds_src and augmented ds_src, (2) ds_dest and augmented ds_dest
    
    Because we aren't training we can simply cache features for augmented and non-augmented image like they do in moco.
    """
    return -1

def dummy_heuristic(ds_src, ds_dest, model):
    print("I'm a dummy")
    return -1