import unittest
from dexterous_bioprosthesis_2021_raw_datasets.tools.assoc_array import assoc_array


class AssocArrayTest(unittest.TestCase):


    def test_assoc_array(self):
        aa = assoc_array()

        first_level_keys=["A", "B", "C"]
        second_level_keys=["E", "F", "G"]

        for fk in first_level_keys:
            for sk  in second_level_keys:
                aa[fk][sk] = [1,2,3]

        flk = [k for k in aa.keys()]
        self.assertTrue( flk == first_level_keys)

        for fk in aa.keys():
            slk = [k for k in aa[fk]]
            self.assertTrue( slk == second_level_keys )
        
if __name__ == '__main__':
    unittest.main()
