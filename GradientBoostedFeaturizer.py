class GradientBoostedFeatureGenerator(object):
    # TODO : Add in any learner, not necessarily LogReg ?
    # TODO : Enable enhanced functionality on the Train/Test split
    # TODO : Should you train GB and LR on the full dataset, or keep the split
    def __init__(self, X, y, nTrees=50, classification=True, build_poly=False):
        """
        Initialize our tree builder with the number of trees needed,
        X and y data to be trained on. We then randomly split the data,
        train the GradientBooster and Linear models.

        The data input should be transformed already (e.g., scaling, encoding, ...)

        INPUTS:
        ------

        X : np.array() = Training features
        y : np.array() = Binary, 1-dimensional target vector
        nTrees: int = Number of trees to build our solution upon
        classification: Bool = Is our target variable
        """
        import numpy as np
        from sklearn.model_selection import train_test_split

        assert len(X) == len(y)
        assert nTrees >= 0

        # We do not want to try to make any predictions if the models are not trained
        self.lin_built = False
        self.tree_built = False
        self.poly_built = False
        # Is our problem classification or regression?
        self.classification = classification
        self.build_poly = build_poly
        # Set our maximum number of trees + leaves
        self.nTrees = nTrees
        self.nLeaves = 50  # Hardcoded at this time!

        # 42: The answer to life, the universe, everything...
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )

        self.X_train, self.X_test, self.y_train, self.y_test = (
            X_train,
            X_test,
            y_train,
            y_test,
        )

        if self.build_poly:
            self._build_poly_features()

        self.gb_features = self.X_train.columns
        # Build our GradBoost and LogReg
        self._train_feature_trees()
        self._train_feature_lin()

    def _train_feature_trees(self):
        """
        Build our Gradient boosted model trained on
        a portion of the input data
        """
        # from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
        import lightgbm as lgb

        if self.classification:
            self.gb = lgb.LGBMClassifier(
                n_estimators=self.nTrees, num_leaves=self.nLeaves
            )
        else:
            self.gb = lgb.LGBMRegressor(
                n_estimators=self.nTrees, num_leaves=self.nLeaves
            )

        self.gb.fit(self.X_train, self.y_train)
        self.tree_built = True
        # If the user wants, you can get the trained tree
        return self.gb

    def _train_feature_lin(self):
        """
        Build our Linear on the remaining fraction of the input data.
        First, the features are generated
        """
        from sklearn.linear_model import LinearRegression, LogisticRegression

        # Instantiate a linear model
        if self.classification:
            self.lin = LogisticRegression(solver="lbfgs")
        else:
            self.lin = LinearRegression()

        # Build our features from the tree
        if self.tree_built:
            X_gen = self.build_features(self.X_test)
            # Train
            self.lin.fit(X_gen, self.y_test)
            self.lin_built = True
            # What columns are we returning?
            self.lin_feat_cols = X_gen.columns

        else:
            print("Error: You did not build a tree first")

        # If the user wants, you can get the trained linear model
        return self.lin

    def _build_poly_features(self):
        """
        Idea: Build a set of poly features, train a simple model,
            and then record top N of these features (10 perhaps?).

        Output: The top N features will then be saved and recreated for the
                main training dataset with the Gradient Booster.

        On any large dataset, this breaks out in a `MemoryError`

        """
        import numpy as np
        import lightgbm as lgb

        if self.poly_built == False:
            if self.classification:
                gb = lgb.LGBMClassifier(n_estimators=10)
            else:
                gb = lgb.LGBMRegressor(n_estimators=10)

        gb.fit(self.X_train, self.y_train)
        self.top_features = [
            self.X_train.columns[idx]
            for idx in np.argsort(gb.feature_importances_)[::-1][:5]
        ]

        for col1 in self.top_features:
            for col2 in self.top_features:
                if (col1 + "|" + col2 not in self.X_train.columns) and (
                    col2 + "|" + col1 not in self.X_train.columns
                ):
                    self.X_train[col1 + "|" + col2] = (
                        self.X_train[col1] * self.X_train[col2]
                    )
                    self.X_test[col1 + "|" + col2] = (
                        self.X_test[col1] * self.X_test[col2]
                    )

        self.poly_built = True

    def build_features(self, X_raw, ohe=True):
        """
        From the GBC's output, we dump out the index of the leaf nodes
        from each classifier as a OHE column (`ohe=True` by default). You can also
        just dump out the leaf indices for each tree as a column

        INPUTS:
        ------
        X_raw: np.array() = Array of the same features as `X`, but new data

        """
        import pandas as pd

        if self.build_poly:
            for col1 in self.top_features:
                for col2 in self.top_features:
                    if (col1 + "|" + col2 not in X_raw.columns) and (
                        col2 + "|" + col1 not in X_raw.columns
                    ):
                        X_raw[col1 + "|" + col2] = X_raw[col1] * X_raw[col2]
        # This gives us a np.array() of each tree's leaf index output
        leaf_node_output = self.gb.predict(X_raw[self.gb_features], pred_leaf=True)

        if not ohe:
            # This just returns a NP array of nRows,nTrees with integer leaf indices
            # Good if you are using these as categorical inputs for an embedding column
            return leaf_node_output

        # Returns the leaf indices for each tree
        leaf_df = pd.DataFrame(
            leaf_node_output[:, :],
            columns=["leaf_index_tree" + str(n) for n in range(self.nTrees)],
        )

        # Now we do a One-Hot of our leaf index to provide to our linear model
        self.leaf_df = pd.get_dummies(
            leaf_df.astype("category"),
            prefix=["OHE_" + str(col) for col in leaf_df.columns],
        )

        # Sometimes the leaf indices never show up in the valid/test data, so fill with 0s
        if self.lin_built:
            for col in self.lin_feat_cols:
                if col not in self.leaf_df.columns:
                    self.leaf_df[col] = 0

            # Return same order column
            return self.leaf_df[self.lin_feat_cols]
        return self.leaf_df

    def build_predictions(self, X_input):
        """
        Finally build our prediction set
        """
        if self.tree_built and self.lin_built:
            X_gen = self.build_features(X_input)
            # Either predict probabilities or real values
            if self.classification:
                y_prob = self.lin.predict_proba(X_gen)
            else:
                y_prob = self.lin.predict(X_gen)

        # Return the scores
        return y_prob
