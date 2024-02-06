import unittest
from sklearn.datasets import load_iris
import tempfile
import os
from dexterous_bioprosthesis_2021_raw_datasets.tools.arff_saver import save_arff
import pandas as pd

class ArffSaverTest(unittest.TestCase):

    def test_save_np(self):
        X,y = load_iris(return_X_y=True)

        arff_path = os.path.join(tempfile.mkdtemp(), "test_np.arff")
        save_arff(X,y,None,arff_path)


    def test_save_np_t(self):
        X,y = load_iris(return_X_y=True)
        t = [i for i in range(X.shape[0])]

        arff_path = os.path.join(tempfile.mkdtemp(), "test_np_t.arff")
        save_arff(X,y,t,arff_path)

    def test_save_df_t(self):
            X,y = load_iris(return_X_y=True)
            t = [i for i in range(X.shape[0])]

            arff_path = os.path.join(tempfile.mkdtemp(), "test_df_t.arff")
            X2 = pd.DataFrame(X)
            save_arff(X2,y,t,arff_path)

    def test_save_df(self):
            X,y = load_iris(return_X_y=True)
            
            arff_path = os.path.join(tempfile.mkdtemp(), "test_df.arff")
            X2 = pd.DataFrame(X)
            save_arff(X2,y,None,arff_path,relation_name="relxx")



if __name__ == '__main__':
    unittest.main()