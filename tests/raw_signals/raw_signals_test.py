import random
import unittest
import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal

from copy import deepcopy

from tests.testing_tools import get_pickled_obj

class RawSignalsTest(unittest.TestCase):

    def test_creation(self):
        try:
            signals = RawSignals()
        except Exception as ex:
            self.fail("An exception has been caught during object creation: " + str(ex))

    def test_appending(self):

        signals = RawSignals()
        N = 10
        M=10
        C = 6
        try:
            for i in range(1,N+1):
                signals.append( RawSignal(signal=np.zeros( (M*i,C))) )
        except Exception as ex:
            self.fail("An exception has been caught during appendint elements to the dataset. "   + str(ex))

        self.assertTrue(len(signals) == N, "Not all signals have been added!")

        try:
            signals.append( RawSignal( signal= np.zeros( (M,C+5) )) )
            self.fail("Incompatible signal was allowed to append")
        except Exception as ex:
            self.assertIsInstance(ex, ValueError, "Not Value error has been raised. Raised: {}".format(ex))

        try:
            signals.append(1)
            self.fail("Object of incompatible type has been appended!")
        except Exception as ex:
            self.assertIsInstance(ex, ValueError, "Not Value error has been raised. Raised: {}".format(ex))

    def test_list_concat(self):

        signals_1 = RawSignals()
        signals_2 = RawSignals()
        N = 10
        M=10
        C = 6
        signal_list  = []
        try:
            for i in range(1,N+1):
                signals_1.append( RawSignal(signal=np.zeros( (M*i,C))) )
                signals_2.append( RawSignal(signal=np.zeros( (M,C))) )
                signal_list.append( RawSignal(signal=np.zeros( (M,C))) )
        except Exception as ex:
            self.fail("An exception has been caught during appendint elements to the dataset. "   + str(ex))

        self.assertTrue( len(signals_1) == len(signals_2), "Signals object have different lengths!")

        lengths = len(signals_1)

        try:
            signals_1 += signals_2
            self.assertTrue( len(signals_1) == 2*lengths, "Not all objects have been added!")
        except Exception as ex:
            self.fail("An exception has been caught during datasets concatenation. "   + str(ex))

        
        try:
            signals_2 += signal_list
        except Exception as ex:
            self.fail("An exception has been caught during concatenating RawSignals with list. "   + str(ex))

        try:
            signals_2 += [1,2]
            self.fail("Concatenating with list of incompatible types")
        except:
            pass

    def generate_sample_data(self,signal_number=10, column_number=3, samples_number=12)->RawSignals:
        signals = RawSignals()

        for i in range(1,signal_number+1):
            signals.append( RawSignal(signal=np.zeros( (samples_number,column_number)),object_class=1.0) )

        return signals

    def test_getitem(self):
        signals = self.generate_sample_data()

        try:
            it  = signals[0]
            self.assertIsNotNone(it, "Object get is None")
            self.assertIsInstance(it, RawSignal, "Object get is not an instance of RawSignal")
        except Exception as ex:
            self.fail("An exception has been caught: {}".format(ex))

    def test_getitem_one_element_tuple_single(self):

        signals = self.generate_sample_data()

        try:
            it  = signals[0,]
            self.assertIsNotNone(it, "Object get is None")
            self.assertIsInstance(it, RawSignal, "Object get is not an instance of RawSignal")
        except Exception as ex:
            self.fail("An exception has been caught: {}".format(ex))

    def test_getitem_one_element_tuple_triple(self):

        signals = self.generate_sample_data()

        try:
            it  = signals[0,:,:]
            self.assertIsNotNone(it, "Object get is None")
            self.assertIsInstance(it, RawSignal, "Object get is not an instance of RawSignal")
        except Exception as ex:
            self.fail("An exception has been caught: {}".format(ex))


    def test_getitem_slices(self):
        signals = self.generate_sample_data()

        try:
            it  = signals[::3]
            self.assertIsNotNone(it, "Object get is None")
            self.assertIsInstance(it, RawSignals, "Object get is not an instance of RawSignals")
        except Exception as ex:
            self.fail("An exception has been caught: {}".format(ex))

    def test_getitem_slices_tuple(self):
        signals = self.generate_sample_data()

        try:
            it  = signals[::3,:]
            self.assertIsNotNone(it, "Object get is None")
            self.assertIsInstance(it, RawSignals, "Object get is not an instance of RawSignals")
        except Exception as ex:
            self.fail("An exception has been caught: {}".format(ex))

    def test_getitem_list(self):
        signals = self.generate_sample_data()

        try:
            sel_list = [1,3,5,7]
            it  = signals[sel_list]
            self.assertIsNotNone(it, "Object get is None")
            self.assertIsInstance(it, RawSignals, "Object get is not an instance of RawSignals")
            self.assertTrue( len(it) == len(sel_list), "The length of new object is wrong.")
        except Exception as ex:
            self.fail("An exception has been caught: {}".format(ex))

    def test_getitem_list_tuple_one(self):
        signals = self.generate_sample_data()

        try:
            sel_list = [1,3,5,7]
            it  = signals[sel_list,]
            self.assertIsNotNone(it, "Object get is None")
            self.assertIsInstance(it, RawSignals, "Object get is not an instance of RawSignals")
            self.assertTrue( len(it) == len(sel_list), "The length of new object is wrong.")
        except Exception as ex:
            self.fail("An exception has been caught: {}".format(ex))

    def test_getitem_iterable(self):
        signals = self.generate_sample_data()

        try:
            sel_list =  [1,3,5,7]
            sel_iter = iter(sel_list)
            it  = signals[sel_iter]
            self.assertIsNotNone(it, "Object get is None")
            self.assertIsInstance(it, RawSignals, "Object get is not an instance of RawSignals")
            self.assertTrue( len(it) == len(sel_list), "The length of new object is wrong.")
        except Exception as ex:
            self.fail("An exception has been caught: {}".format(ex))

    def test_getitem_list_bool(self):
        signals = self.generate_sample_data()

        try:
            sel_bool = np.zeros(len(signals), dtype=bool)
            sel_bool[0] = True
            sel_bool[2] = True
            sel_bool[3] = True

            sel_num  = np.sum(sel_bool)
            it  = signals[sel_bool]
            self.assertIsNotNone(it, "Object get is None")
            self.assertIsInstance(it, RawSignals, "Object get is not an instance of RawSignals")
            self.assertTrue( len(it) == sel_num, "The length of new object is wrong.")
        except Exception as ex:
            self.fail("An exception has been caught: {}".format(ex))


    def test_creation_with_list(self):
        signals = self.generate_sample_data()

        tmp_obj = RawSignals(signals)

    def test_get_labels(self):
        signals = self.generate_sample_data()

        y = signals.get_labels()

        self.assertIsNotNone(y, "Label list is none")
        self.assertTrue(len(y) == len(signals), "Length of the label list is wrong.")

    def test_get_timestamps(self):
        signals = self.generate_sample_data()

        timestamps = signals.get_timestamps()

        self.assertIsNotNone(timestamps, "Timestamp list is none")
        self.assertTrue(len(timestamps) == len(signals), "Length of the timestamp list is wrong.")

    def test_equality(self):
        signals = self.generate_sample_data()

        self.assertTrue( signals == signals, "Self equality")

        self.assertTrue( signals != signals.raw_signals_list, "Other type")

        object_copy = deepcopy(signals)
        self.assertTrue( signals == object_copy , "Equality with deep copy")

        other_signals =  self.generate_sample_data(signal_number=30)   

        self.assertTrue( signals != other_signals, "Other signals.")


    def test_label_setting_good(self):
        signals = self.generate_sample_data()

        n_signals = len(signals)

        new_labels = np.zeros(n_signals)
        signals.set_labels(new_labels)

        self.assertTrue(np.allclose( new_labels, signals.get_labels()), "Wrong labels has been set")

    def test_label_setting_bad(self):
        signals = self.generate_sample_data()

        li = [0,1]

        try:
            signals.set_labels(li)
        except ValueError:
            pass
        except Exception:
            self.fail("Wrong exception has been raised.")

    def test_pickle(self):
        signals = self.generate_sample_data()
        pickled = get_pickled_obj(signals)

        self.assertTrue( signals == pickled, "Object should have been equall with its pickled copy")

    def test_random_choice(self):
        signals = self.generate_sample_data()
        sig = random.choice(signals)

        self.assertIsNotNone(sig, "Signal is none")
        self.assertIsInstance(sig, RawSignal, "Not a raw Signal object")

    def test_initialize_empty(self):
        signals = self.generate_sample_data()
        empty_sigs = signals.initialize_empty()

        self.assertIsInstance(empty_sigs, RawSignals, "Not an instance of RawSignals")
        self.assertTrue( len( empty_sigs ) == 0, "Not empty")



        

    
if __name__ == '__main__':
    unittest.main()