from .hog_extractor import *

def get_features(img):
    features = extract_hog_features(img, cspace='YUV', hog_channel='ALL')
    return features
