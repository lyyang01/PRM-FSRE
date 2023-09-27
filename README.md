# PRM-FSRE
Sorry for the delayed release of the code! 
The code is for NAACL2022-findings long paper "Learn from Relation Information: Towards Prototype Representation Rectification for Few-Shot Relation Extraction".

## Dataset

The dataset used is the publicly available FewRel dataset. Please refer to the following link for information on how to obtain the related dataset: [https://github.com/lylylylylyly/SimpleFSRE].

## Code
We provide our PRM-FSRE and the baseline Proto-FSRE in the code.
You can get the main results of our experiment by running the following code. 
```
sh run_my.sh
```
For the cross-domain condition, you can run:
```
sh run_pubmed.sh
``` 
When you have obtained your submission file, you can execute the following code to get the test results from the FewRel official website.
```
sh run_submit.sh
sh run_pubmed_submit.sh
```
We also provide the visualization code for the visualized figure in the paper. You can just run
```
sh run_visual.sh
```
