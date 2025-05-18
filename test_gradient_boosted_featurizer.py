import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from GradientBoostedFeaturizer import GradientBoostedFeatureGenerator

@pytest.fixture
def classification_data():
    """Create a simple classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]), y

@pytest.fixture
def regression_data():
    """Create a simple regression dataset."""
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42
    )
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]), y

def test_initialization_classification(classification_data):
    """Test initialization with classification data."""
    X, y = classification_data
    gbf = GradientBoostedFeatureGenerator(X, y, n_trees=10, classification=True)
    
    assert gbf.classification is True
    assert gbf.n_trees == 10
    assert gbf.n_leaves == 50
    assert gbf.build_poly is False
    assert gbf.tree_built is True
    assert gbf.lin_built is True

def test_initialization_regression(regression_data):
    """Test initialization with regression data."""
    X, y = regression_data
    gbf = GradientBoostedFeatureGenerator(X, y, n_trees=10, classification=False)
    
    assert gbf.classification is False
    assert gbf.n_trees == 10
    assert gbf.n_leaves == 50
    assert gbf.build_poly is False
    assert gbf.tree_built is True
    assert gbf.lin_built is True

def test_polynomial_features(classification_data):
    """Test polynomial feature generation."""
    X, y = classification_data
    gbf = GradientBoostedFeatureGenerator(X, y, n_trees=10, build_poly=True)
    
    # Check if polynomial features were created
    assert gbf.poly_built is True
    assert hasattr(gbf, 'top_features')
    assert len(gbf.top_features) == 5  # We expect 5 top features
    
    # Check if polynomial features exist in the training data
    poly_features = [col for col in gbf.X_train.columns if '|' in col]
    assert len(poly_features) > 0

def test_feature_generation(classification_data):
    """Test feature generation process."""
    X, y = classification_data
    gbf = GradientBoostedFeatureGenerator(X, y, n_trees=10)
    
    # Test feature generation on new data
    new_X = X.iloc[:10]  # Use first 10 samples as new data
    features = gbf.build_features(new_X)
    
    # Check if features were generated correctly
    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == 10  # Same number of rows as input
    assert features.shape[1] > 0  # Should have generated features

def test_predictions_classification(classification_data):
    """Test prediction generation for classification."""
    X, y = classification_data
    gbf = GradientBoostedFeatureGenerator(X, y, n_trees=10)
    
    # Test predictions on new data
    new_X = X.iloc[:10]
    predictions = gbf.build_predictions(new_X)
    
    # Check prediction format
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == 10
    assert predictions.shape[1] == 2  # Binary classification probabilities

def test_predictions_regression(regression_data):
    """Test prediction generation for regression."""
    X, y = regression_data
    gbf = GradientBoostedFeatureGenerator(X, y, n_trees=10, classification=False)
    
    # Test predictions on new data
    new_X = X.iloc[:10]
    predictions = gbf.build_predictions(new_X)
    
    # Check prediction format
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == 10
    assert predictions.ndim == 1  # Single value per prediction

def test_input_validation():
    """Test input validation."""
    # Test invalid X type
    with pytest.raises(TypeError):
        GradientBoostedFeatureGenerator(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    
    # Test invalid y type
    with pytest.raises(TypeError):
        GradientBoostedFeatureGenerator(pd.DataFrame([[1, 2], [3, 4]]), [0, 1])
    
    # Test mismatched lengths
    with pytest.raises(ValueError):
        GradientBoostedFeatureGenerator(
            pd.DataFrame([[1, 2], [3, 4]]),
            np.array([0, 1, 2])
        )
    
    # Test invalid n_trees
    with pytest.raises(ValueError):
        GradientBoostedFeatureGenerator(
            pd.DataFrame([[1, 2], [3, 4]]),
            np.array([0, 1]),
            n_trees=0
        )

def test_model_training_order(classification_data):
    """Test that models must be trained in the correct order."""
    X, y = classification_data
    gbf = GradientBoostedFeatureGenerator(X, y, n_trees=10)
    
    # Reset training state
    gbf.tree_built = False
    gbf.lin_built = False
    
    # Try to build predictions without training
    with pytest.raises(RuntimeError):
        gbf.build_predictions(X)

def test_feature_consistency(classification_data):
    """Test that features are consistent between training and prediction."""
    X, y = classification_data
    gbf = GradientBoostedFeatureGenerator(X, y, n_trees=10)
    
    # Generate features for training data
    train_features = gbf.build_features(gbf.X_train)
    
    # Generate features for test data
    test_features = gbf.build_features(gbf.X_test)
    
    # Check that feature columns are consistent
    assert set(train_features.columns) == set(test_features.columns) 