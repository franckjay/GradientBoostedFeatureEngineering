class GradientBoostedFeatureGenerator(object):
    # TODO : Add in XGBoost/LightGBM instead of sklearn's GBC
    # TODO : Add in any learner, not necessarily LogReg ?
    # TODO : Enable enhanced functionality on the Train/Test split
    # TODO : Finish a Polynomial feature builder
    def __init__(self, X, y, nTrees=50, classification=True):
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
        from sklearn.model_selection import train_test_split

        assert (len(X) == len(y))
        assert (nTrees >= 0)

        # We do not want to try to make any predictions if the models are not trained
        self.lin_built = False
        self.tree_built = False
        # Is our problem classification or regression?
        self.classification = classification
        # Set our maximum number of trees
        self.nTrees = nTrees

        # 42: The answer to life, the universe, everything...
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        # Build our GradBoost and LogReg
        self._train_feature_trees()
        self._train_feature_lin()

    def _train_feature_trees(self):
        """
        Build our Gradient boosted model trained on
        a portion of the input data
        """
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

        if self.classification:
            self.gb = GradientBoostingClassifier(n_estimators=self.nTrees)
        else:
            self.gb = GradientBoostingRegressor(n_estimators=self.nTrees)

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
        else:
            print("Error: You did not build a tree first")

        # If the user wants, you can get the trained linear model
        return self.lin

    def _build_poly_features(self):
        """
        Idea: Build a set of poly features, train a simple model (RF?),
            and then record top N of these features (10 perhaps?).

        Output: The top N features will then be saved and recreated for the
                main training dataset with the Gradient Booster.

        On any large dataset, this breaks out in a `MemoryError`

        """

        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(2)
        poly.fit_transform(self.X_train)

    def build_features(self, X_raw):
        """
        From the GBC's output, we dump out the index of the leaf nodes
        from each classifier

        INPUTS:
        ------
        X_raw: np.array() = Array of the same features as `X`, but new data

        """
        import pandas as pd

        # This gives us a np.array() of each tree's leaf index output
        leaf_node_output = self.gbc.apply(X_raw)

        # Returns the leaf indices for each tree
        leaf_df = pd.DataFrame(leaf_node_output[:, :],
                               columns=["leaf_index_tree" + str(n) for n in range(self.nTrees)])

        # Now we do a One-Hot of our leaf index to provide to our linear model
        self.leaf_df = pd.get_dummies(leaf_df.astype('category'),
                                      prefix=["OHE_" + str(col) for col in leaf_df.columns])

        return self.leaf_df

    def build_predictions(self, X_input):
        """

        """
        if self.tree_built and self.lin_reg_built:
            X_gen = self.build_features(X_input)
            # Either predict probabilities or real values
            if self.classification:
                y_prob = self.lin.predict_proba(X_gen)
            else:
                y_prob = self.lin.predict(X_gen)

        # Return the scores
        return y_prob