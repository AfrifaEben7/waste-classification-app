import unittest
from src.training.model import EfficientNetModel  # Adjust the import based on your model class name
from src.training.config import Config  # Adjust the import based on your config class name

class TestEfficientNetModel(unittest.TestCase):

    def setUp(self):
        self.model = EfficientNetModel()
        self.config = Config()

    def test_model_initialization(self):
        self.assertIsNotNone(self.model)

    def test_model_output_shape(self):
        input_tensor = self.model.input_shape  # Adjust based on how you define input shape
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (self.config.batch_size, self.config.num_classes))  # Adjust based on your config

    def test_model_compile(self):
        try:
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        except Exception as e:
            self.fail(f"Model compilation failed with error: {e}")

if __name__ == '__main__':
    unittest.main()