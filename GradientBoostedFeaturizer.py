class GradientBoostedFeatureGenerator(object):
    # TODO : Add in XGBoost/LightGBM instead of sklearn's GBC
    # TODO : Add in any learner, not necessarily LogReg ?
    # TODO : Enable enhanced functionality on the Train/Test split

    def __init__(self, X, y, nTrees=50):
        """
        Initialize our tree builder with the number of trees needed,
        X and y data to be trained on. We then randomly split the data,
        train the GradientBooster and LogReg models.

        The data input should be transformed already (e.g., scaling, encoding, ...)

        INPUTS:
        ------
        nTrees: int = Number of trees to build our solution upon
        X : np.array() = Training features
        y : np.array() = Binary, 1-dimensional target vector
        """
        from sklearn.model_selection import train_test_split

        assert (len(X) == len(y))
        assert (nTrees >= 0)

        # We do not want to try to make any predictions if the models are not trained
        self.log_reg_built = False
        self.tree_built = False

        # Set our maximum number of trees
        self.nTrees = nTrees

        # 42: The answer to life, the universe, everything...
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        # Build our GradBoost and LogReg
        self._train_feature_trees()
        self._train_feature_log_reg()

    def _train_feature_trees(self):
        """
        Build our Gradient boosted classifier set on
        a portion of the input data
        """
        from sklearn.ensemble import GradientBoostingClassifier

        self.gbc = GradientBoostingClassifier(n_estimators=self.nTrees)
        self.gbc.fit(self.X_train, self.y_train)
        self.tree_built = True
        # If the user wants, you can get the trained tree
        return self.gbc

    def _train_feature_log_reg(self):
        """
        Build our LogReg on the remaining fraction of the input data.
        First, the features are generated
        """
        from sklearn.linear_model import LogisticRegression

        # Instantiate a LogReg model
        self.log_reg = LogisticRegression(solver='lbfgs')
        # Build our features from the tree
        if self.tree_built:
            X_gen = self.build_features(self.X_test)
            # Train
            self.log_reg.fit(X_gen, self.y_test)
            self.log_reg_built = True
        else:
            print("Error: You did not build a tree first")

        # If the user wants, you can get the trained linear model
        return self.log_reg

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
        leaf_df = pd.DataFrame(leaf_node_output[:, :, 0],
                               columns=["leaf_index_tree" + str(n) for n in range(self.nTrees)])

        # Now we do a One-Hot of our leaf index to provide to our linear model
        self.leaf_df = pd.get_dummies(leaf_df.astype('category'),
                                      prefix=["OHE_" + str(col) for col in leaf_df.columns])

        return self.leaf_df

    def build_predictions(self, X_input):
        """

        """
        if self.tree_built and self.log_reg_built:
            X_gen = self.build_features(X_input)
            y_prob = self.log_reg.predict_proba(X_gen)

        # Return the scores
        return y_prob