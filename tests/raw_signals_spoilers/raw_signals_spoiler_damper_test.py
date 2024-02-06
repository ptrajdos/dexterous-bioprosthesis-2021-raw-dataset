
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_damper import RawSignalsSpoilerDamper
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_silencer import RawSignalsSpoilerSilencer
from tests.raw_signals_spoilers.raw_signals_spoiler_test import RawSignalsSpoilerTest

import numpy as np

class RawSignalsSpoilerDamperTest(RawSignalsSpoilerTest):

    __test__ = True

    def get_spoilers(self):
        return [
            RawSignalsSpoilerDamper(),
            RawSignalsSpoilerDamper(channels_spoiled_frac=0),

        ]
    
    def get_spoiler_class(self):
        return RawSignalsSpoilerDamper
    
    def is_test_snr(self):
        return True
    
    def get_snrs(self):
        return [0,1,2,3,4]
    
    def test_silencing(self):
        silencers = [RawSignalsSpoilerDamper(snr=0)]

        data = self.generate_sample_data()

        for silencer in silencers:

        
            t_data = silencer.fit_transform(data)

            for sig, t_sig in zip(data, t_data):

                sig_abs_sum =  np.sum( np.abs( sig.to_numpy()))
                t_sig_abs_sum =  np.sum( np.abs( t_sig.to_numpy()))

                if np.equal(silencer.channels_spoiled_frac,0.0):
                    self.assertTrue( np.equal( sig_abs_sum, t_sig_abs_sum ) , "Sums of absolute values")
                else:
                    self.assertTrue(sig_abs_sum > t_sig_abs_sum, "Sums of absolute values")

                if np.equal(silencer.channels_spoiled_frac,1.0):
                    self.assertTrue(np.equal( t_sig_abs_sum, 0.0 ))

    def test_damping(self):
        dampers = [ RawSignalsSpoilerDamper(),  RawSignalsSpoilerDamper(snr=4),  RawSignalsSpoilerDamper(snr=10),
                   RawSignalsSpoilerDamper(snr=-1), RawSignalsSpoilerDamper(snr=-2)
                    ]

        data = self.generate_sample_data()

        for damper in dampers:

        
            t_data = damper.fit_transform(data)

            for sig, t_sig in zip(data, t_data):

                sig_abs_sum =  np.sum( np.abs( sig.to_numpy()))
                t_sig_abs_sum =  np.sum( np.abs( t_sig.to_numpy()))


                self.assertTrue(sig_abs_sum > t_sig_abs_sum, "Sums of absolute values.")
                

                