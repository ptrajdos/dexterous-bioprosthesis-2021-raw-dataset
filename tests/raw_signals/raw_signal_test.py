import unittest
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
import numpy as np

from tests.testing_tools import get_pickled_obj


class RawSignalTest(unittest.TestCase):

    def test_rawsignal(self):
        R, C = 50, 10
        sig = np.zeros((R, C))

        obj = RawSignal(signal=sig, timestamp=10, object_class=0)

        obj2 = RawSignal(
            signal=sig,
            timestamp=333,
            object_class=0,
            channel_names=["X{}".format(i) for i in range(C)],
        )

        obj3 = RawSignal(signal=sig, timestamp=10, object_class=1.1 )

        obj4 = RawSignal(signal=sig, timestamp=10, object_class="class1")

        obj5 = RawSignal(signal=sig, timestamp=10, object_class=np.array([1, 2, 3]))

    def test_equaity(self):
        R, C = 50, 10
        sig = np.zeros((R, C))
        sig2 = np.ones((R, C))

        obj = RawSignal(signal=sig, timestamp=10, object_class=np.sqrt(2))
        self.assertTrue(obj == obj, "Object should have been equal. Continous label")

        obj = RawSignal(signal=sig, timestamp=10, object_class=0)
        self.assertTrue(obj == obj, "Object should have been equal")

        self.assertTrue(obj != "obj2", "Object should not have been equal")

        obj2 = RawSignal(signal=sig, timestamp=10, object_class=0)
        self.assertTrue(obj == obj2, "Object should have been equal")

        obj2 = RawSignal(signal=sig2, timestamp=10, object_class=0)
        self.assertTrue(obj != obj2, "Object should not have been equal")

        obj2 = RawSignal(signal=sig, timestamp=1, object_class=0)
        self.assertTrue(obj != obj2, "Object should not have been equal")

        obj2 = RawSignal(signal=sig, timestamp=10, object_class=1)
        self.assertTrue(obj != obj2, "Object should not have been equal")

        obj2 = RawSignal(
            signal=sig,
            timestamp=10,
            object_class=0,
            channel_names=["X{}".format(i) for i in range(C)],
        )
        self.assertTrue(obj != obj2, "Object should not have been equal")

        obj3 = RawSignal(
            signal=sig,
            timestamp=10,
            object_class=np.asanyarray([0, 1]),)
        
        obj4 = RawSignal(
            signal=sig,
            timestamp=10,
            object_class=np.asanyarray([0, 1]),)

        obj5 = RawSignal(
            signal=sig,
            timestamp=10,
            object_class=np.asanyarray([0, 2]),)
        
        self.assertTrue(obj3 == obj4, "Objects 3 and 4 should have been equal")
        self.assertTrue(obj3 != obj5, "Objects 3 and 5 should not have been equal")

    def test_more_equality(self):
        R, C = 50, 10
        sig = np.zeros((R, C))
        sig2 = np.ones((R, C))
        dtypes = [np.float32, np.float64, np.single, np.double,str, object, np.str_, np.object_]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                c1 = np.int32(1)
                c2 = np.int32(2)
                obj = RawSignal(signal=sig, timestamp=10, object_class=c1.astype(dtype))
                obj2 = RawSignal(signal=sig, timestamp=10, object_class=c1.astype(dtype))
                obj3 = RawSignal(signal=sig, timestamp=10, object_class=c2.astype(dtype))

                self.assertTrue(obj == obj2, "Objects should have been equal")
                self.assertTrue(obj != obj3, "Objects should not have been equal")

    def test_more_equality_arrays(self):
        R, C = 50, 10
        sig = np.zeros((R, C))
        sig2 = np.ones((R, C))
        dtypes = [np.float32, np.float64, np.single, np.double,np.str_, np.object_, str, object]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                a1 = np.array([1, 2, 3], dtype=dtype)
                a2 = np.array([1, 2, 4], dtype=dtype)
                obj = RawSignal(signal=sig, timestamp=10, object_class=a1)
                obj2 = RawSignal(signal=sig, timestamp=10, object_class=a1)
                obj3 = RawSignal(signal=sig, timestamp=10, object_class=a2)

                self.assertTrue(obj == obj2, "Objects should have been equal")
                self.assertTrue(obj != obj3, "Objects should not have been equal")

    def test_array_label_2d(self):
        R, C = 50, 10
        sig = np.zeros((R, C))
        dtypes = [np.float32, np.float64, np.str_, np.object_]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                a1 = np.array([[1, 2], [3, 4]], dtype=dtype)
                a2 = np.array([[1, 2], [3, 4]], dtype=dtype)
                a3 = np.array([[1, 2], [3, 5]], dtype=dtype)
                obj = RawSignal(signal=sig, timestamp=10, object_class=a1)
                obj2 = RawSignal(signal=sig, timestamp=10, object_class=a2)
                obj3 = RawSignal(signal=sig, timestamp=10, object_class=a3)

                self.assertTrue(obj == obj2, "2D array labels should have been equal")
                self.assertTrue(obj != obj3, "2D array labels should not have been equal")

    def test_array_label_different_shapes(self):
        R, C = 50, 10
        sig = np.zeros((R, C))
        a1 = np.array([1, 2, 3])
        a2 = np.array([1, 2])
        obj = RawSignal(signal=sig, timestamp=10, object_class=a1)
        obj2 = RawSignal(signal=sig, timestamp=10, object_class=a2)

        self.assertTrue(obj != obj2, "Arrays with different shapes should not be equal")

    def test_array_label_empty(self):
        R, C = 50, 10
        sig = np.zeros((R, C))
        dtypes = [np.float64, np.str_]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                a1 = np.array([], dtype=dtype)
                a2 = np.array([], dtype=dtype)
                obj = RawSignal(signal=sig, timestamp=10, object_class=a1)
                obj2 = RawSignal(signal=sig, timestamp=10, object_class=a2)

                self.assertTrue(obj == obj2, "Empty array labels should have been equal")

    def test_array_label_vs_scalar(self):
        R, C = 50, 10
        sig = np.zeros((R, C))
        obj_arr = RawSignal(signal=sig, timestamp=10, object_class=np.array([1]))
        obj_scalar = RawSignal(signal=sig, timestamp=10, object_class=1)

        self.assertTrue(obj_arr != obj_scalar, "Array label vs scalar label should not be equal")

    def test_array_label_numeric_tolerance(self):
        R, C = 50, 10
        sig = np.zeros((R, C))
        a1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        a2 = np.array([1.0 + 1e-8, 2.0 - 1e-8, 3.0 + 1e-8], dtype=np.float64)
        a3 = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        obj = RawSignal(signal=sig, timestamp=10, object_class=a1)
        obj2 = RawSignal(signal=sig, timestamp=10, object_class=a2)
        obj3 = RawSignal(signal=sig, timestamp=10, object_class=a3)

        self.assertTrue(obj == obj2, "Numeric arrays within tolerance should be equal")
        self.assertTrue(obj != obj3, "Numeric arrays with large diff should not be equal")

    def test_array_label_string_values(self):
        R, C = 50, 10
        sig = np.zeros((R, C))
        a1 = np.array(["class_a", "class_b"], dtype=np.str_)
        a2 = np.array(["class_a", "class_b"], dtype=np.str_)
        a3 = np.array(["class_a", "class_c"], dtype=np.str_)
        obj = RawSignal(signal=sig, timestamp=10, object_class=a1)
        obj2 = RawSignal(signal=sig, timestamp=10, object_class=a2)
        obj3 = RawSignal(signal=sig, timestamp=10, object_class=a3)

        self.assertTrue(obj == obj2, "String array labels should have been equal")
        self.assertTrue(obj != obj3, "String array labels should not have been equal")

    def test_array_label_serialization(self):
        R, C = 50, 10
        sig = np.zeros((R, C))
        dtypes = [np.float32, np.float64, np.str_, np.object_]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                a = np.array([1, 2, 3], dtype=dtype)
                obj = RawSignal(signal=sig, timestamp=10, object_class=a)
                pickled = get_pickled_obj(obj)

                self.assertIsNotNone(pickled, "Pickled should not have been None!")
                self.assertTrue(obj == pickled, "Pickled object should have been equal")

    def test_array_label_getitem_preserves(self):
        R, C = 50, 10
        sig = np.zeros((R, C))
        a = np.array([10, 20, 30], dtype=np.float64)
        obj = RawSignal(signal=sig, timestamp=10, object_class=a)

        sliced = obj[:20]
        self.assertTrue(np.array_equal(sliced.object_class, a), "Slicing should preserve array label")

        sliced2 = obj[:, :5]
        self.assertTrue(np.array_equal(sliced2.object_class, a), "Column slicing should preserve array label")

    def test_getitem(self):

        R, C = 50, 10
        sig = np.zeros((R, C))

        obj = RawSignal(signal=sig, timestamp=10, object_class=0)

        obj_cp = obj[:, :]
        self.assertTrue(obj == obj_cp, "Whole slicing. Objects should have been equal")

        # one idx selection
        S = 20
        s_obj = obj[:S]
        self.assertTrue(
            s_obj.signal.shape == (S, obj.signal.shape[1]),
            "One index. Shapes after selection",
        )
        self.assertTrue(
            s_obj.channel_names == obj.channel_names, "One index. Channel names"
        )

        # one element tuple
        s_obj = obj[:S,]
        self.assertTrue(
            s_obj.signal.shape == (S, obj.signal.shape[1]),
            "One elem tuple. Shapes after selection",
        )
        self.assertTrue(
            s_obj.channel_names == obj.channel_names, "One slem tuple. Channel names"
        )

        # int selection
        s_obj = obj[[S]]
        self.assertTrue(
            s_obj.signal.shape == (1, obj.signal.shape[1]),
            "Int. Shapes after selection",
        )
        self.assertTrue(s_obj.channel_names == obj.channel_names, "Int. Channel names")
        self.assertTrue(s_obj.signal.shape[0] == 1, "Int selection. One row")

        # Two indices selection
        # slice
        SC = 3
        s_obj = obj[:S, :SC]
        self.assertTrue(
            s_obj.signal.shape == (S, SC), "Two indices, slice. Shapes after selection"
        )
        self.assertTrue(
            s_obj.channel_names == obj.channel_names[:SC],
            "Two indices, slice. Colum names",
        )

        # Collection
        SC = [1, 2, 3]
        s_obj = obj[:S, SC]
        self.assertTrue(
            s_obj.signal.shape == (S, len(SC)),
            "Two indices, collection. Shapes after selection",
        )
        #TODO this fail if channel names are not tuple. Should we force channel names to be tuple?
        self.assertTrue(
            s_obj.channel_names == tuple([obj.channel_names[i] for i in SC]),
            "Two indices, collection. Colum names",
        )

        # Boolean slicing
        bool_sel = [bool(1) for i in range(C)]
        obj_cp = obj[:, bool_sel]
        self.assertTrue(
            obj == obj_cp, "Whole slicing via booleans. Objects should have been equal"
        )

        try:
            s_obj = obj[:, :, :]
            self.fail("Wrong number of slices. Code shouldn't have reached this point!")
        except IndexError:
            pass
        except Exception as ex:
            self.fail("Wrong number of slices. Wrong exception")

    def test_len(self):

        R, C = 50, 10
        sig = np.zeros((R, C))

        obj = RawSignal(signal=sig, timestamp=10, object_class=0)
        self.assertTrue(len(obj) == R, "RawSignal len. Wrong value")

    def test_to_numpy(self):
        R, C = 50, 10
        sig = np.zeros((R, C))

        obj = RawSignal(signal=sig, timestamp=10, object_class=0)
        X = obj.to_numpy()
        self.assertIsInstance(X, np.ndarray, "Wrong type")
        self.assertTrue(X.shape == (R, C), "Wrong size")

    def test_label_set(self):
        R, C = 50, 10
        sig = np.zeros((R, C))

        obj = RawSignal(signal=sig, timestamp=10, object_class=0)

        object_label = 1

        obj.set_label(object_label)

        self.assertTrue(object_label == obj.get_label(), "Wrong label has been set")

    def test_serialization(self):
        R, C = 50, 10
        sig = np.zeros((R, C))

        obj = RawSignal(signal=sig, timestamp=10, object_class=0)

        pickled = get_pickled_obj(obj)

        self.assertIsNotNone(pickled, "Pickled should not have been None!")
        self.assertTrue(
            obj == pickled, "Object and object reqd from pickle should have been equall"
        )

    def test_serialization_arrays(self):
        R, C = 50, 10
        sig = np.zeros((R, C))

        obj = RawSignal(signal=sig, timestamp=10, object_class=np.zeros((1,3)))

        pickled = get_pickled_obj(obj)

        self.assertIsNotNone(pickled, "Pickled should not have been None!")
        self.assertTrue(
            obj == pickled, "Object and object reqd from pickle should have been equall"
        )

    def test_dtype(self):
        R, C = 50, 10
        dtypes = [np.float32, np.float64, np.single, np.double]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                sig = np.zeros((R, C), dtype=dtype)

                obj = RawSignal(signal=sig, timestamp=10, object_class=0)

                self.assertTrue(
                    np.issubdtype(obj.signal.dtype, dtype),
                    f"Signal dtype should be {dtype}, got {obj.signal.dtype}",
                )


if __name__ == "__main__":
    unittest.main()
