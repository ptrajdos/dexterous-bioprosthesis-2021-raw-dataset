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

    def test_equaity(self):
        R, C = 50, 10
        sig = np.zeros((R, C))
        sig2 = np.ones((R, C))

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
        self.assertTrue(
            s_obj.channel_names == [obj.channel_names[i] for i in SC],
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
