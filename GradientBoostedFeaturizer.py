from typing import Optional, Union, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import lightgbm as lgb

class GradientBoostedFeatureGenerator:
    """
    A feature generator that uses gradient boosting trees to create new features
    for a linear model. This approach combines the power of tree-based models
    with the interpretability of linear models.
    
    The process works as follows:
    1. Split data into train/test sets
    2. Train a gradient boosting model on the training set
    3. Use the leaf indices from the trees as new features
    4. Train a linear model on these new features
    
    Attributes:
        n_trees (int): Number of trees in the gradient boosting model
        n_leaves (int): Maximum number of leaves per tree
        classification (bool): Whether this is a classification task
        build_poly (bool): Whether to build polynomial features
    """
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_trees: int = 50,
        classification: bool = True,
        build_poly: bool = False,
        random_state: int = 42
    ) -> None:
        """
        Initialize the feature generator.
        
        Args:
            X: Training features as a pandas DataFrame
            y: Target values as a numpy array
            n_trees: Number of trees to build
            classification: Whether this is a classification task
            build_poly: Whether to build polynomial features
            random_state: Random seed for reproducibility
        """
        self._validate_inputs(X, y, n_trees)
        
        # Initialize state flags
        self.lin_built = False
        self.tree_built = False
        self.poly_built = False
        
        # Store configuration
        self.classification = classification
        self.build_poly = build_poly
        self.n_trees = n_trees
        self.n_leaves = 50  # TODO: Make this configurable
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.5, random_state=random_state
        )
        
        # Store original feature names
        self.gb_features = self.X_train.columns
        
        # Build features and train models
        if self.build_poly:
            self._build_poly_features()
            
        self._train_feature_trees()
        self._train_feature_lin()
    
    @staticmethod
    def _validate_inputs(X: pd.DataFrame, y: np.ndarray, n_trees: int) -> None:
        """Validate input parameters."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if n_trees < 1:
            raise ValueError("n_trees must be positive")
    
    def _train_feature_trees(self) -> lgb.LGBMClassifier | lgb.LGBMRegressor:
        """
        Train the gradient boosting model on the training set.
        
        Returns:
            The trained gradient boosting model
        """
        model_class = lgb.LGBMClassifier if self.classification else lgb.LGBMRegressor
        self.gb = model_class(
            n_estimators=self.n_trees,
            num_leaves=self.n_leaves
        )
        
        self.gb.fit(self.X_train, self.y_train)
        self.tree_built = True
        return self.gb
    
    def _train_feature_lin(self) -> Union[LogisticRegression, LinearRegression]:
        """
        Train the linear model on the generated features.
        
        Returns:
            The trained linear model
        """
        if not self.tree_built:
            raise RuntimeError("Must train tree model before linear model")
            
        model_class = LogisticRegression if self.classification else LinearRegression
        self.lin = model_class(solver="lbfgs")
        
        X_gen = self.build_features(self.X_test)
        self.lin.fit(X_gen, self.y_test)
        self.lin_built = True
        self.lin_feat_cols = X_gen.columns
        
        return self.lin
    
    def _build_poly_features(self) -> None:
        """
        Build polynomial features from the top most important features
        identified by a quick gradient boosting model.
        """
        if self.poly_built:
            return
            
        # Train a quick model to identify important features
        model_class = lgb.LGBMClassifier if self.classification else lgb.LGBMRegressor
        gb = model_class(n_estimators=10)
        gb.fit(self.X_train, self.y_train)
        
        # Get top 5 most important features
        self.top_features = [
            self.X_train.columns[idx]
            for idx in np.argsort(gb.feature_importances_)[::-1][:5]
        ]
        
        # Create polynomial features
        for col1 in self.top_features:
            for col2 in self.top_features:
                feature_name = f"{col1}|{col2}"
                if feature_name not in self.X_train.columns:
                    self.X_train[feature_name] = self.X_train[col1] * self.X_train[col2]
                    self.X_test[feature_name] = self.X_test[col1] * self.X_test[col2]
        
        self.poly_built = True
    
    def build_features(
        self,
        X_raw: pd.DataFrame,
        ohe: bool = True
    ) -> pd.DataFrame:
        """
        Generate features from the gradient boosting model.
        
        Args:
            X_raw: Input features to generate new features from
            ohe: Whether to one-hot encode the leaf indices
            
        Returns:
            DataFrame containing the generated features
        """
        if self.build_poly:
            self._add_poly_features(X_raw)
            
        # Get leaf indices from the gradient boosting model
        leaf_node_output = self.gb.predict(X_raw[self.gb_features], pred_leaf=True)
        
        if not ohe:
            return leaf_node_output
            
        # Create DataFrame of leaf indices
        leaf_df = pd.DataFrame(
            leaf_node_output,
            columns=[f"leaf_index_tree{n}" for n in range(self.n_trees)]
        )
        
        # One-hot encode the leaf indices
        self.leaf_df = pd.get_dummies(
            leaf_df.astype("category"),
            prefix=[f"OHE_{col}" for col in leaf_df.columns]
        )
        
        # Handle missing columns in validation/test data
        if self.lin_built:
            for col in self.lin_feat_cols:
                if col not in self.leaf_df.columns:
                    self.leaf_df[col] = 0
            return self.leaf_df[self.lin_feat_cols]
            
        return self.leaf_df
    
    def _add_poly_features(self, X_raw: pd.DataFrame) -> None:
        """Add polynomial features to the input DataFrame."""
        for col1 in self.top_features:
            for col2 in self.top_features:
                feature_name = f"{col1}|{col2}"
                if feature_name not in X_raw.columns:
                    X_raw[feature_name] = X_raw[col1] * X_raw[col2]
    
    def build_predictions(self, X_input: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained models.
        
        Args:
            X_input: Input features to generate predictions for
            
        Returns:
            Array of predictions
        """
        if not (self.tree_built and self.lin_built):
            raise RuntimeError("Models must be trained before making predictions")
            
        X_gen = self.build_features(X_input)
        
        if self.classification:
            return self.lin.predict_proba(X_gen)
        return self.lin.predict(X_gen)
