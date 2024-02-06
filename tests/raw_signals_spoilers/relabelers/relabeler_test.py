import unittest
import numpy as np

class RelabelerTest(unittest.TestCase):

    __test__ = False
    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise  unittest.SkipTest("Skipping")
        
    def get_relabelers(self):
        """
        gets the spoilers to be tested
        """
        raise  unittest.SkipTest("Skipping")
    

    def test_relabelers(self):


        relabelers = self.get_relabelers()

        dummy_labels = np.random.choice( a=[0,1,2], size=20)
        n_dummy_labels = len(dummy_labels)


        for relabeler in relabelers:

            relabeler.fit(dummy_labels)

            new_labels = relabeler.transform(dummy_labels)

            self.assertIsNotNone(new_labels, "New labels are None!")
            self.assertTrue( n_dummy_labels == len(new_labels), "Wrong number of transformed labels!" )

    def test_relabelers_ft(self):


        relabelers = self.get_relabelers()

        dummy_labels = np.random.choice( a=[0,1,2], size=20)
        n_dummy_labels = len(dummy_labels)


        for relabeler in relabelers:

            new_labels = relabeler.fit_transform(dummy_labels)

            self.assertIsNotNone(new_labels, "New labels are None!")
            self.assertTrue( n_dummy_labels == len(new_labels), "Wrong number of transformed labels!" )