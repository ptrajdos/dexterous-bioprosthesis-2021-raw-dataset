from sklearn.metrics import silhouette_score, silhouette_samples
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import SetCreator
from dexterous_bioprosthesis_2021_raw_datasets.distance_matrix_calculators.distance_matrix_calculator import DistanceMatrixCalculator
from sklearn.cluster import AgglomerativeClustering
from math import sqrt
import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_mds import no_flatten

class SetCreatorClusterDist(SetCreator):

    def __init__(self, distance_calculator:DistanceMatrixCalculator, flatten_function = no_flatten) -> None:
        super().__init__()
        self.distance_calculator:DistanceMatrixCalculator = distance_calculator
        self.flatten_function = flatten_function
        self.channel_selected_attribs = None # List containing number of attributes for each channel

    def _max_clusters_to_try(self, raw_signals:RawSignals):
        return int( sqrt( len(raw_signals) ) ) + 1
    

    def _fit(self, raw_signals: RawSignals):

        initial_distance_matrix = self.distance_calculator.calculate_distance_matrix(raw_signals=raw_signals)
        distance_matrix = self.flatten_function(initial_distance_matrix,keepdims=True)

        is_channel_attribute_mapping = True if initial_distance_matrix.shape[0] == distance_matrix.shape[0] else False

        max_clusters = self._max_clusters_to_try(raw_signals)
        n_channels = distance_matrix.shape[0]

        best_samples_ids = []

        for channel_id in range(n_channels):

            channel_dist_matrix = distance_matrix[channel_id,:,:]

            silh = []

            for k in range(2,max_clusters):
                ag = AgglomerativeClustering(metric = 'precomputed', linkage='average', n_clusters=k)
                clusters = ag.fit_predict(channel_dist_matrix)
                sil_val = silhouette_score(channel_dist_matrix,labels=clusters, metric='precomputed')
                silh.append((k,sil_val,clusters))

            best_k, best_silh, best_cluster_assig = sorted(silh, key=lambda x: x[1], reverse=True)[0]
            samples_silh = silhouette_samples(channel_dist_matrix, labels=best_cluster_assig, metric='precomputed')

            all_res = [*zip(range(len(raw_signals)), samples_silh, best_cluster_assig)]

            for k in range(best_k):
                res_subset = [*filter(lambda x: x[2]==k, all_res)]
                best_id = sorted(res_subset, key= lambda x: x[1], reverse=True)[0][0]
                best_samples_ids.append(best_id)
                

        best_samples_ids = np.asanyarray(best_samples_ids)

        self.reference_samples = RawSignals(raw_signals[best_samples_ids], sample_rate=raw_signals.sample_rate)

        representation = np.concatenate( distance_matrix[:,:,best_samples_ids],axis=1)
    
        if is_channel_attribute_mapping:
            self.channel_selected_attribs = []
            n_best_sample_ids = len(best_samples_ids)
            offset = 0
            for _ in range(n_channels):
                self.channel_selected_attribs.append( offset + np.arange(n_best_sample_ids))
                offset += n_best_sample_ids

        return representation, raw_signals.get_labels(), raw_signals.get_timestamps()

        

    def fit(self, raw_signals: RawSignals, y=None):
        self._fit(raw_signals)

        return self

    def fit_transform(self, raw_signals: RawSignals, y=None):
        return self._fit(raw_signals)
    
    def transform(self, raw_signals: RawSignals):

        repr = np.concatenate( self.flatten_function( self.distance_calculator.calculate_distance_matrix_set_2_set(raw_signals_1=raw_signals, raw_signals_2=self.reference_samples), keepdims=True),axis=1)

        return repr, raw_signals.get_labels(), raw_signals.get_timestamps()
    
    def get_channel_attribs_indices(self):
        return self.channel_selected_attribs
    