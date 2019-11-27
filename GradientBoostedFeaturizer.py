class GradientBoostedFeatureGenerator(object):
    def __init__(self, X, y, nTrees=50):
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        """
        Initialize our tree builder with the number of trees needed,
        X and y data to be trained on.
        INPUTS:
        ------
        nTrees: int = Number of trees to build our solution upon
        X : np.array() = Training features
        y : np.array() = Binary, 1-dimensional target vector
        """
        assert (len(X) == len(y))
        assert (nTrees >= 0)
        self.nTrees = nTrees
        self.X = X
        self.y = y

    def train_feature_trees(self):
        """
        Build our Gradient boosted classifier set
        """
        self.gbc = GradientBoostingClassifier(n_estimators=self.nTrees)
        self.gbc.fit(self.X, self.y)

        # If the user wants, you can get the trained tree
        return self.gbc

    def train_feature_log_reg(self):
        """
        Build our Gradient boosted classifier set
        """
        self.log_reg = LogisticRegression(solver='lbfgs')
        X_trans = self.build_features()
        self.log_reg.fit(X_trans, self.y)

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

        leaf_node_output = self.gbc.apply(X_raw)
        # Returns the leaf indices for each tree
        self.leaf_df = pd.DataFrame(leaf_node_output[:, :, 0],
                                    columns=["leaf_index_tree" + str(n) for n in range(self.nTrees)])
        # Now we do a One Hot of our leaf index to provide to our linear model
        self.leaf_df = pd.get_dummies(leaf_df.astype('category'),
                                      prefix=["OHE_" + str(col) for col in leaf_df.columns])
        return self.leaf_df