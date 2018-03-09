import unittest
import tensorflow as tf
from CaptioningModel import CaptioningModel

class CaptioningModelTest(unittest.TestCase):
    
    def test(self):
        model = CaptioningModel({'<START>': 0, '<END>': 0, '<NULL>': 0}, dim_embed=128, dim_hidden=128)
        loss = model.build_model()
        tf.get_variable_scope().reuse_variables()
        outputs = model.build_sampler()
        pass


if __name__ == '__main__':
    unittest.main()