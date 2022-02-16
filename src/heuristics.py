
"""
Any heuristic function should take as input the source dataset, the destination dataset, and the model. It should output a score, which is > 0. 
"""

def cka_heuristic(ds_src, ds_dest, model):
    """
    Official code for CKA can found here:
    https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
    [TODO]: calculate CKA between ds_src, ds_dest using some output features from model
    """
    return -1

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