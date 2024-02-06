
from itertools import combinations, product
from joblib import delayed
import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.distance_matrix_calculators.distance_matrix_calculator import DistanceMatrixCalculator
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.tools.progressparallel import ProgressParallel


from fastdtw import fastdtw


class DistanceMatrixCalculatorDTWFast2(DistanceMatrixCalculator):
    """
    Calculates the pairwise distance matrix using fast dtw approach.
    According to:
    Stan Salvador, and Philip Chan. “FastDTW: Toward accurate dynamic time warping in linear time and space.” Intelligent Data Analysis 11.5 (2007): 561-580.
    """

    def __init__(self,dtw_options: dict={"dist":2},tqdm_disable=True, n_jobs=None) -> None:
        """
        Arguments:
        ---------
        dtw_options -- options for the dtw algorithm (as a dict) or an empty dict
        """
        super().__init__()
        self.dtw_options = self._filter_dtw_options( dtw_options )
        self.tqdm_disable = tqdm_disable
        self.n_jobs = n_jobs

    def _filter_dtw_options(self,dtw_options):
        
        return dtw_options

    def raw_signal_dist(self, raw_signal_a: RawSignal, raw_signal_b:RawSignal):
        signal_a  = raw_signal_a.signal
        signal_b = raw_signal_b.signal
        n_channels = signal_a.shape[1]

        channel_distances = np.zeros( (n_channels) )

        for i in range(n_channels):
            distance_1, _ = fastdtw(signal_a[:,i],signal_b[:,i], **self.dtw_options)
            distance_2, _ = fastdtw(signal_b[:,i],signal_a[:,i], **self.dtw_options)
            channel_distances[i] = 0.5 * (distance_1 + distance_2)


        overall_distance = channel_distances

        return overall_distance

    def _compute_raw_signals_dist(self, raw_signals, i):

        dist_list = []

        for j in range(i+1, len(raw_signals)):
            distance = self.raw_signal_dist(raw_signal_a=raw_signals[i], raw_signal_b=raw_signals[j])
            dist_list.append(  (distance,i,j) )

        return dist_list


    def calculate_distance_matrix(self, raw_signals: RawSignals):

        num_signals = len(raw_signals)
        num_channels = len( raw_signals[0].channel_names)
        distance_matrix = np.zeros( (num_channels, num_signals, num_signals) )

        
        pair_dist_list = []
        total_iter_number = num_signals

        pair_dist_list_pre = ProgressParallel(n_jobs=self.n_jobs,use_tqdm=not self.tqdm_disable,total=total_iter_number)(delayed(self._compute_raw_signals_dist)( raw_signals,i ) for i in range(num_signals)  )

        for it in pair_dist_list_pre:
            for i in it:
                pair_dist_list.append(i)

        for dist,i,j in pair_dist_list:
            distance_matrix[:, i, j] = dist
            distance_matrix[:, j, i] = dist

        return distance_matrix
    
    def _compute_raw_signal_2_set_dist(self, raw_signals,raw_signal, i):
        distance = self.raw_signal_dist(raw_signal_a=raw_signal, raw_signal_b=raw_signals[i])

        return distance, i
    
    def raw_signal_dist_2_set(self, raw_signal: RawSignal, raw_signals: RawSignals):

        num_signals = len(raw_signals)
        num_channels = len( raw_signals[0].channel_names)

        distance_matrix = np.zeros( (num_channels, num_signals) )
        total_dist_number  = num_signals

        dist_list = ProgressParallel(n_jobs=self.n_jobs,use_tqdm=not self.tqdm_disable,total=total_dist_number)(delayed(self._compute_raw_signal_2_set_dist)( raw_signals,raw_signal,i ) for i in range(num_signals) )

        for dist,i in dist_list:
            distance_matrix[:, i] = dist
            
        return distance_matrix
    
    def _compute_raw_signals_dist_2_set(self, raw_signals_a, raw_signals_b, i,j):
        distance = self.raw_signal_dist(raw_signal_a=raw_signals_a[i], raw_signal_b=raw_signals_b[j])

        return distance, i, j
    
    def calculate_distance_matrix_set_2_set(self, raw_signals_1: RawSignals, raw_signals_2: RawSignals):

        num_signals_1 = len(raw_signals_1)
        num_signals_2 = len(raw_signals_2)
        num_channels = len( raw_signals_1[0].channel_names)
        distance_matrix = np.zeros( (num_channels, num_signals_1, num_signals_2) )

        total_pair_number = num_signals_1 * num_signals_2

        pair_dist_list = ProgressParallel(n_jobs=self.n_jobs,use_tqdm=not self.tqdm_disable,total=total_pair_number)(delayed(self._compute_raw_signals_dist_2_set)( raw_signals_1, raw_signals_2,i,j ) for i,j in product( range(num_signals_1), range(num_signals_2))  )

        for dist,i,j in pair_dist_list:
            distance_matrix[:, i, j] = dist

        return distance_matrix