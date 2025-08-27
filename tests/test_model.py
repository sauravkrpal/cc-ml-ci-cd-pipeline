import unittest
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.model_path = "model/churn_model.pkl"
        self.columns_path = "model/trained_columns.pkl"

    def test_model_file_exists(self):
        """Test if the model file exists"""
        self.assertTrue(os.path.exists(self.model_path),
                        "Model file not found. Run train.py first.")

    def test_model_loading(self):
        """Test if the model can be loaded"""
        try:
            model = joblib.load(self.model_path)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Failed to load model: {str(e)}")

    def test_model_type(self):
        """Test if the loaded model is RandomForestClassifier"""
        model = joblib.load(self.model_path)
        self.assertIsInstance(model, RandomForestClassifier,
                              "Model should be RandomForestClassifier")

    def test_model_features_saved(self):
        """Test if feature columns were saved"""
        self.assertTrue(os.path.exists(self.columns_path),
                        "trained_columns.pkl file not found.")
        trained_columns = joblib.load(self.columns_path)
        self.assertGreater(len(trained_columns), 0,
                           "trained_columns.pkl should not be empty")

    def test_model_has_feature_importances(self):
        """Test if the model has feature importances"""
        model = joblib.load(self.model_path)
        self.assertTrue(hasattr(model, "feature_importances_"),
                        "Model should have feature_importances_ attribute")
        self.assertGreaterEqual(len(model.feature_importances_), 4,
                                "Model should have at least 4 features")

if __name__ == "__main__":
    unittest.main(verbosity=2)
