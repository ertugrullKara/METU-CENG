from cyvlfeat.sift.dsift import dsift
from cyvlfeat.sift.sift import sift
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import pickle
import json

class _SiftExtractor:
    def __init__(self):
        self.detector = sift
    
    def detect(self, gray_im):
        return self.detector(gray_im, 
                             compute_descriptor=True)    
class _DenseSiftExtractor:
    def __init__(self, step=1, size=3):
        self.detector = dsift
        self.step = step
        self.size = size
    
    def detect(self, grayim):
        return self.detector(grayim, 
                             step=self.step, 
                             size=self.size)
class Extractor:
    def __init__(self, sift_type, step=1, size=3):
        if sift_type.lower() == "sift":
            self._extractor = _SiftExtractor()
        else:
            self._extractor = _DenseSiftExtractor(step, size)
    
    def extract(self, grayim):
        return self._extractor.detect(grayim)
            
class CBIR:
    def __init__(self, sift_type, cluster_size, descriptor_filename, step=1, size=3):
        self.sift_type = sift_type.lower()
        self.sift = Extractor(sift_type, step=step, size=size)
        self.set_cluster_size(cluster_size)
        self.load = False
        self.data = {}
        self.descriptors = {}
        self.desc_list = None
        self.descriptor_filename = descriptor_filename
        
    def set_cluster_size(self, cluster_size):
        self.cluster_size = cluster_size
        self.cluster_centers = {}
        self.clustering = KMeans(n_clusters=self.cluster_size, n_init=5, max_iter=150, n_jobs=-2, precompute_distances=False)
        
    def extract_features(self, im, imname):
        self.descriptors[imname] = {}
        _, descriptor = self.sift.extract(im)
        if self.sift_type == "sift":
            reduced_descriptors = descriptor[np.random.choice(descriptor.shape[0], size=int(descriptor.shape[0]*.2)), :]
        else:
            reduced_descriptors = descriptor
        if self.desc_list is not None:
            self.desc_list = np.concatenate((self.desc_list, reduced_descriptors), axis=0)
        else:
            self.load = True
            self.desc_list = np.array(reduced_descriptors)
        self.descriptors[imname]["descriptor"] = reduced_descriptors
        if self.sift_type == "sift":
            self.descriptors[imname]["all_descriptor"] = descriptor
    
    def extract_save_clusters(self):
        if not self.load:
            print("Loading descriptors.")
            self.load_descriptors(load_as_list=True)
        
        print("Extracting cluster centers.")
        self.cluster_centers[self.cluster_size] = self.clustering.fit(self.desc_list).cluster_centers_
        self.descriptors["{}_cluster_centers".format(str(self.cluster_size))] = self.cluster_centers[self.cluster_size]
        
        print("Saving.")
        self.save_clusters()

    def bof(self, cluster_size):
        if not self.load:
            print("Loading descriptors.")
            self.load_descriptors()
        print("Creating Bag of Features representation for images.")
        for imname in self.descriptors.keys():
            if "jpg" not in imname:
                continue
            if self.sift_type == "sift":
                imdescriptor = np.array(self.descriptors[imname]["all_descriptor"])
            else:
                imdescriptor = np.array(self.descriptors[imname]["descriptor"])
            cluster_ids = self.clustering.predict(imdescriptor)
            histogram, _ = np.histogram(cluster_ids, bins=np.arange(cluster_size))
            histogram = histogram / cluster_size
            self.descriptors[imname]["{}_cluster_id".format(str(cluster_size))] = histogram
        print("Saving.")
        self.save_clusters()
        
    def save_clusters(self):
        self.save_descriptors()
    
    def save_descriptors(self):
        print("Descriptors:", self.desc_list.shape)
        pickle.dump(self.descriptors, open(self.descriptor_filename, 'wb'), protocol=4)

    def load_descriptors(self, load_as_list=False):
        print("Extracting descriptors.")
        self.load = True
        self.descriptors = pickle.load(open(self.descriptor_filename, 'rb'))
        if not load_as_list:
            return
        descriptors = None
        for imname in self.descriptors.keys():
            if "jpg" not in imname:
                continue
            if descriptors is not None:
                descriptors = np.concatenate((descriptors, np.array(self.descriptors[imname]["descriptor"])), axis=0)
            else:
                descriptors = np.array(self.descriptors[imname]["descriptor"])
            break
        self.desc_list = np.array(descriptors)
        print(self.desc_list.shape)
        