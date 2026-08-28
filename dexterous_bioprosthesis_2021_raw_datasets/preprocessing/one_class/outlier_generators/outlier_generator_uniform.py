"""Module implementing uniform random outlier generation.

Generates synthetic outlier samples by drawing values uniformly from
an extended range of each feature column, simulating out-of-distribution
data points.
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted
from dexterous_bioprosthesis_2021_raw_datasets.preprocessing.one_class.outlier_generators.outlier_generator import OutlierGenerator


class OutlierGeneratorUniform(OutlierGenerator):
      """Outlier generator using uniform random sampling.

      Generates outliers by sampling uniformly from each column's range,
      extended by ``extend_factor``.

      Args:
          outlier_label: Label value for generated outliers.
          extend_factor: Fraction by which to extend column min/max range.
          element_fraction: Fraction of input rows to generate (minimum 1).

      """

      def __init__(self, outlier_label=-1, extend_factor=0.1, element_fraction=1.0 ) -> None:
         super().__init__(outlier_label=outlier_label)
         self.extend_factor  = extend_factor
         self.element_fraction = element_fraction



      def fit(self, X, y):
         """Learn column ranges from the training data.

         Args:
             X: Training feature matrix.
             y: Training label vector.

         Returns:
             The fitted generator.

         """
         super().fit(X,y)

         column_mins = np.min(X,axis=0)
         column_maxs = np.max(X,axis=0)

         self.column_mins_ = column_mins - np.abs(column_mins) * self.extend_factor
         self.column_maxs_ = column_maxs + np.abs(column_maxs) * self.extend_factor

         n_rows, self.n_cols_ = X.shape

         self.n_rows_ = int( np.max( (X.shape[0] * self.element_fraction , 1))) 

         return self
     
      def generate(self):
         """Generate uniform random outlier samples.

         Returns:
             Tuple of (outlier feature array, outlier label array).

         """
         super().generate()
         out = np.zeros((self.n_rows_, self.n_cols_))

         for col_idx in range(self.n_cols_):
            out[:,col_idx] = np.random.uniform(low=self.column_mins_[col_idx],
                                                high =self.column_maxs_[col_idx],
                                                  size=(self.n_rows_))

         labels = np.zeros((self.n_rows_,)).astype(self.outlier_label_dtype_)
         labels[:]=self.outlier_label_

         return out, labels
        
