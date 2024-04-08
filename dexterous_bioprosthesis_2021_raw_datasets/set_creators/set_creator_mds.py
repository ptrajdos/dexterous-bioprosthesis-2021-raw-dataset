
from joblib import delayed
from dexterous_bioprosthesis_2021_raw_datasets.distance_matrix_calculators.distance_matrix_calculator import DistanceMatrixCalculator
from dexterous_bioprosthesis_2021_raw_datasets.distance_matrix_calculators.distance_matrix_calculator_dtw_fast import DistanceMatrixCalculatorDTWFast
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import SetCreator

import numpy as np
from dtw import dtw
from sklearn.manifold import MDS
from tqdm import tqdm
from itertools import combinations
from sklearn.exceptions import NotFittedError
import pygad
from kneed.knee_locator import KneeLocator

from dexterous_bioprosthesis_2021_raw_datasets.tools.progressparallel import ProgressParallel

default_ga_options={
    "num_generations": 10000,
    "num_parents_mating": 10,
    "parent_selection_type": "sss",
    "keep_parents": -1,
    "crossover_type": "single_point",
    "mutation_type": "random",
    "mutation_percent_genes": 20,
    "sol_per_pop_mult": 1,
    "suppress_warnings":True
}

def no_flatten(in_data, axis=None, keepdims=None):
    return np.copy(in_data)

def euclid_flatten(distance_matrix, keepdims=False, axis=0):
    n_channels =  distance_matrix.shape[0]
    squared_sum = np.sum( np.square(distance_matrix),axis=axis, keepdims=keepdims)
    flatten = np.sqrt(squared_sum/n_channels)
    return flatten

class SetCreatorMDS(SetCreator):
    """
     Creates the set using DTW and then MDS approach.
     Implementation inspired by:
     @inproceedings{Wang2013,
        doi = {10.1109/sibgrapi.2013.11},
        url = {https://doi.org/10.1109/sibgrapi.2013.11},
        year = {2013},
        month = aug,
        publisher = {{IEEE}},
        author = {Quan Wang and Kim L. Boyer},
        title = {Feature Learning by Multidimensional Scaling and Its Applications in Object Recognition},
        booktitle = {2013 {XXVI} Conference on Graphics,  Patterns and Images}
        }
    """
    def __init__(self, dist_matrix_calc:DistanceMatrixCalculator=  DistanceMatrixCalculatorDTWFast() , mds_options: dict={},ga_options:dict = default_ga_options,tqdm_disable=True, n_jobs=None,
     flatten_function = no_flatten, n_attr:int=2, find_best:bool=False,step:int=1) -> None:
        """
        Arguments:
        ---------
        distance_matrix_calc: DistanceMatrixCalculator -- an object to calculate the distance matrix
        mds_options: dict -- options for the mds algorithm (as a dict) or an empty dict
        tqdm_disable: bool -- determines whetrher print tqdm progress bar
        n_jobs: int/None -- determines the number of jobs to be used to calculate MDS
        flatten_function -- function used to flatten the multi-channel distance matrix
        n_attr: int -- number of attributes to select. Max number of attributes to select if find_best is True
        find_best: bool -- Decides whether perform searching for the best number of attributes
        step: int -- step for attribute search 
        """
        super().__init__()
        self.mds_options =  self._filter_mds_options( mds_options )
        self.ga_options = self._filter_ga_options(ga_options)
        self.tqdm_disable = tqdm_disable
        self.n_jobs = n_jobs
        self.distance_matrix_calc = dist_matrix_calc
        self.flatten_function = flatten_function
        self.n_attr = n_attr
        self.find_best = find_best
        self.step = step


        self.raw_signals_set:RawSignals = None
        self.dataset = None
        self.channel_selected_attribs = None # List containing number of attributes for each channel


    def _filter_mds_options(self,mds_options):
        mds_options.pop('dissimilarity',None)
        mds_options.pop('n_jobs',None)
        mds_options.pop('n_components',None)
        mds_options.pop('normalized_stress',None)
        return mds_options

    def _filter_ga_options(self, ga_options):
        ga_options.pop("initial_population", None)
        ga_options.pop("fitness_function", None)
        return ga_options

    def _calculate_mds(self, distance_matrix):

        n_slices = distance_matrix.shape[0] #n_channels
        self.channel_selected_attribs = [] #number of attributes for each channel

        if not self.find_best:
            
            mds_result = None
            for n_slice in range(n_slices):
                mds_obj = MDS(dissimilarity='precomputed', n_jobs=self.n_jobs,n_components=self.n_attr, normalized_stress=False ,**self.mds_options)

                tmp_mds_result = mds_obj.fit_transform(distance_matrix[n_slice])
                self.channel_selected_attribs.append(tmp_mds_result.shape[1])

                if mds_result is None:
                    mds_result = tmp_mds_result
                else:
                    mds_result = np.concatenate(( mds_result, tmp_mds_result  ), axis=1)
                 
            return mds_result
        

        n_attr_to_check = [*range(2,self.n_attr, self.step)]

        mds_result = None

        for n_slice in range(n_slices):
            stress_values = []

            for n_at in tqdm(n_attr_to_check, desc="Finding best number of attributes", total=len(n_attr_to_check) ,leave=False, disable= self.tqdm_disable):
                mds_obj = MDS(dissimilarity='precomputed', n_jobs=self.n_jobs,n_components=n_at,normalized_stress=False,  **self.mds_options)
                _ = mds_obj.fit_transform(distance_matrix[n_slice])
                stress_values.append(mds_obj.stress_)

            kneedle = KneeLocator(n_attr_to_check, stress_values, S=1.0, curve="convex", direction="decreasing", online=False)
            n_atts_final = kneedle.knee
            

            mds_obj = MDS(dissimilarity='precomputed', n_jobs=self.n_jobs,n_components=n_atts_final, normalized_stress=False, **self.mds_options)
            mds_result_tmp = mds_obj.fit_transform(distance_matrix[n_slice])

            self.channel_selected_attribs.append(mds_result_tmp.shape[1])

            if mds_result is None:
                mds_result = mds_result_tmp
            else:
                mds_result = np.concatenate( (mds_result, mds_result_tmp), axis=1 )

        return mds_result



    def create_set_from_distance_matrix(self, raw_signals, distance_matrix):
        
        X = self._calculate_mds(distance_matrix)

        extracted_objs_classes = np.asanyarray([rs.get_label() for rs in raw_signals])
        extracted_objs_timestamps = np.asanyarray([rs.get_timestamp() for rs in raw_signals])

        return X, extracted_objs_classes, extracted_objs_timestamps


    def fit_transform(self, raw_signals:RawSignals, y=None):

        self.fit(raw_signals)

        return self.dataset

    def fit(self, raw_signals: RawSignals, y=None):
        super().fit(raw_signals)

        self.raw_signals_set = raw_signals

        distance_matrix_channels = self.distance_matrix_calc.calculate_distance_matrix(raw_signals)

        distance_matrix = self.flatten_function(distance_matrix_channels, axis=0, keepdims=True)

        self.dataset = self.create_set_from_distance_matrix(raw_signals, distance_matrix)

        self.attrib_indices_list = None
        if distance_matrix_channels.shape[0] == distance_matrix.shape[0]:
            #Not flattened at all
            self._calculate_attrib_indices()


        return self
    
    def _calculate_attrib_indices(self):
        if self.channel_selected_attribs is None:
            return None

        self.attrib_indices_list = []
        n_channels = len(self.channel_selected_attribs)

        offset = 0 
        for ch_idx in range(n_channels):
            n_channel_attribs = self.channel_selected_attribs[ch_idx]
            self.attrib_indices_list.append( offset + np.arange( n_channel_attribs) )
            offset += n_channel_attribs


    def _distance_raw_signal_training_data(self,raw_signal:RawSignal):
        """
        Return distance from the raw signal to all raw signals in the training set

        Arguments:
        ---------

        raw_signal: RawSignal -- raw signal to calculate the distances

        Returns:
        -------
        Numpy array containing distances to all training objects.
        Distances may be vectors

        """

        distances = []

        for t_sig_id, training_signal in enumerate(self.raw_signals_set):
            dist = self.flatten_function( self.distance_matrix_calc.raw_signal_dist(training_signal, raw_signal), axis=0, keepdims=True)
            distances.append(dist)

        dist_matrix = np.asanyarray(distances)

        return dist_matrix

    def _calculate_distances_to_training_data(self, raw_signals:RawSignals):
        """
        Calculates distances from raw signals to all points in the training dataset

        Arguments:
        ---------
        raw_signals: RawSignals -- distance to calculate the distances.

        Returns:
        --------
        Numpy array containing the distances. shape (len(raw_signals), len(training_data))
        """
        
        signal_distances_list = ProgressParallel(n_jobs=self.n_jobs,use_tqdm=not self.tqdm_disable,total=len(raw_signals),desc="Distances from training to new")(delayed(self._distance_raw_signal_training_data)( raw_signal ) for raw_signal in raw_signals )

        signal_distances_array = np.asanyarray(signal_distances_list)
        return signal_distances_array


    def _new_X_vec(self,new_to_train_distances,n_idx):
        """
        Returns a set of metric attributes.

        Arguments:
        ---------
        new_to_train_distances: np.array -- distance table generated by _calculate_distances_to_training_data
        n_idx: int -- index of a new point for which the new representation is calculated

        Returns:
        -------
        A vector containing metric representation of a given raw_signal

        """

        n_slices = len( self.channel_selected_attribs)
        solution = []

        slice_offset = 0

        for n_slice in range(n_slices):
            n_sel = self.channel_selected_attribs[n_slice]
            start = slice_offset
            stop = slice_offset + n_sel
            slice_offset+= n_sel
            
        
            def fitness_function(solution, solution_idx):
                """
                Fitness function for the genetic algirithm
                """
                s_sum = 0
                for i in range( len(self.raw_signals_set) ):
                    val = np.square( np.sqrt( np.sum( np.square(solution - self.dataset[0][i,start:stop]))) - new_to_train_distances[n_idx,i,n_slice])
                    s_sum+=val
                fitness = 1.0/ (s_sum + 1E-6)

                return fitness

            init_pop = self.initialize_population(start,stop)

            ga_instance = pygad.GA(fitness_func=fitness_function,
                            initial_population= init_pop,
                            **self.ga_options
                        )

            slice_solution = list(ga_instance.best_solution()[0])
            solution+= slice_solution

        np_solution = np.asanyarray(solution)
            
        return np_solution

    def initialize_population(self,start,stop):
        train_solutions = self.dataset[0][:,start:stop]

        new_solutions_number = train_solutions.shape[0] * self.ga_options.pop("sol_per_pop_mult",1)

        new_solutions = np.zeros( (new_solutions_number, train_solutions.shape[1]) )

        min_vals = np.min(train_solutions, axis= 0)
        max_vals = np.max(train_solutions, axis= 0)

        for attr_idx in range(new_solutions.shape[1]):
            new_solutions[:,attr_idx] = np.random.uniform(low=min_vals[attr_idx], high=max_vals[attr_idx], size=(new_solutions_number,))

        initial_population = np.concatenate( (train_solutions, new_solutions), axis=0)


        return initial_population
        
            

    def transform(self, raw_signals: RawSignals):
        if self.raw_signals_set is None:
            raise NotFittedError("The model is not fitted!")

        new_to_train_distances = self._calculate_distances_to_training_data(raw_signals)
    
        new_X = ProgressParallel(n_jobs=self.n_jobs,use_tqdm=not self.tqdm_disable,total=len(raw_signals),desc="Fitting new points")(delayed(self._new_X_vec)( new_to_train_distances,i ) for i in range(len(raw_signals)) )

        tr_dataset = np.asanyarray(new_X)

        extracted_objs_classes = [x.object_class for x in raw_signals]
        extracted_objs_timestamps = [x.timestamp for x in raw_signals]

        return tr_dataset, extracted_objs_classes, extracted_objs_timestamps
    

    def get_channel_attribs_indices(self):

        return self.attrib_indices_list