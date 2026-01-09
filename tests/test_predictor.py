import unittest
from src.inference.predictor import load_model, predict
from src.inference.utils import preprocess_image
import numpy as np
from PIL import Image

class TestPredictor(unittest.TestCase):

    def setUp(self):
        self.model_path = 'path/to/your/model.h5'  # Update with the actual model path
        self.model = load_model(self.model_path)

    def test_load_model(self):
        self.assertIsNotNone(self.model)

    def test_preprocess_image(self):
        test_image = Image.new('RGB', (224, 224))  # Create a dummy image
        processed_image = preprocess_image(test_image)
        self.assertEqual(processed_image.shape, (1, 224, 224, 3))

    def test_predict(self):
        test_image = Image.new('RGB', (224, 224))  # Create a dummy image
        processed_image = preprocess_image(test_image)
        prediction = predict(self.model, processed_image)
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(prediction.shape[0], 1)  # Check if prediction is for one sample

if __name__ == '__main__':
    unittest.main()