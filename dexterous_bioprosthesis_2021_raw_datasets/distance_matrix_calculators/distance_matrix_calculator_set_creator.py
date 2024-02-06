
import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.distance_matrix_calculators.distance_matrix_calculator import DistanceMatrixCalculator
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import SetCreator
from sklearn.metrics import pairwise_distances
from sklearn.exceptions import NotFittedError

default_distance_parameters = {
    "metric": "euclidean"
}

class DistanceMatrixCalculatorSetCreator(DistanceMatrixCalculator):
    """
    Calculates the pairwise distance matrix using dtw approach.
    """

    def __init__(self, set_creator:SetCreator ,distance_parameters = default_distance_parameters, n_jobs=None) -> None:
        """
        Arguments:
        ---------
        dtw_options -- options for the dtw algorithm (as a dict) or an empty dict
        """
        super().__init__()
        
        self.n_jobs = n_jobs
        self.set_creator = set_creator
        self.distance_parameters =  self.process_distance_parameters(distance_parameters)
        self.is_fitted=False

    def process_distance_parameters(self, distance_parameters):
        distance_parameters.pop("n_jobs", None)

        metric_val = distance_parameters.pop("metric")
        if metric_val != "precomputed":
            distance_parameters["metric"] = metric_val

        return distance_parameters

    

    def raw_signal_dist(self, raw_signal_a: RawSignal, raw_signal_b:RawSignal):
        if not self.is_fitted:
            raise NotFittedError("The distance matrix must be calculated first!")

        tmp_signals = RawSignals()
        tmp_signals.append(raw_signal_a)
        tmp_signals.append(raw_signal_b)

        X,y,t = self.set_creator.transform(tmp_signals)

        distance = np.asanyarray([pairwise_distances(X, n_jobs=self.n_jobs, **self.distance_parameters)[0,1]])
        
        return distance


    def raw_signal_dist_2_set(self, raw_signal: RawSignal, raw_signals: RawSignals):

        
        X,_,_ = self.set_creator.fit_transform(raw_signals=raw_signals)
        self.is_fitted = True

        tmp_signals = RawSignals()
        tmp_signals.append(raw_signal)

        Y,_,_ = self.set_creator.transform(tmp_signals)

        distances = pairwise_distances(Y,X, n_jobs=self.n_jobs, **self.distance_parameters)


        return distances    


    def calculate_distance_matrix(self, raw_signals: RawSignals):

        
        X,y,t = self.set_creator.fit_transform(raw_signals)
        self.is_fitted = True
        
        dm = pairwise_distances(X=X, n_jobs=self.n_jobs, **self.distance_parameters)
        dm[ np.diag_indices_from(dm)] = 0.0 #TODO Why on Linux it is needed?

        dist_matrix = np.asanyarray( [dm])

        return dist_matrix
    

    def calculate_distance_matrix_set_2_set(self, raw_signals_1: RawSignals, raw_signals_2: RawSignals):

        X,_,_ = self.set_creator.fit_transform(raw_signals_1)
        self.is_fitted = True
        Y,_,_ = self.set_creator.transform(raw_signals_2)

        dm = pairwise_distances(X,Y, n_jobs=self.n_jobs, **self.distance_parameters )
        

        dist_matrix = np.asanyarray( [dm])

        return dist_matrix
