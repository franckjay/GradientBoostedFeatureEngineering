# GradientBoostedFeatureEngineering

Inspired by this [paper](https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf)

To run in its simplest form, throw in some `X`, `y` binary classification training data:
`gbc_feat = GradientBoostedFeatureGenerator(X, y, nTrees=50, classification=True)`

Builds the Feature Generator. If you are getting too many output features for your model,
lower the number of trees in the generator. To get the output features in Pandas format (for testing):
`gbc_feat.build_features(X)`

```
idx	OHE_leaf_index_tree0_1.0	OHE_leaf_index_tree0_2.0	
0	0	                        1	
1	1	                        0	    

```
To get the pipeline to work generating predictions:
`gbc_feat.build_predictions(X)`

which outputs the LinReg/LogReg predicted probabilities:
```
array([[0.40241641, 0.59758359],
       [0.8265274 , 0.1734726 ],
       [0.48789265, 0.51210735],
```