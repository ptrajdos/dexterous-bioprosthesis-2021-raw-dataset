import unittest

from dexterous_bioprosthesis_2021_raw_datasets.tools.counter import Counter

class CounterTest(unittest.TestCase):

    def test_counter(self):

        init_val = 1
        step = 3

        cnt = Counter(init_val)

        self.assertEqual( int(cnt),init_val, "Wrong at initialization. int()")
        self.assertEqual( cnt,init_val, "Wrong at initialization. eq")

        cnt+=step
        self.assertEqual( cnt,init_val+step, "Increasing")

        cnt-=step
        self.assertEqual( cnt,init_val, "Decreasing")

    def test_as_function_arg(self):
        init_val = 1
        step = 3

        cnt = Counter(init_val)

        def fun(counter,step):
            counter+=step

        fun(cnt,step)
        self.assertEqual( cnt,init_val+step, "Increasing")

        fun(cnt,-step)
        self.assertEqual( cnt,init_val, "Decreasing")





if __name__ == '__main__':
    unittest.main()
 