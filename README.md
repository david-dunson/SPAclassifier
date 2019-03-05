# Classification via local manifold approximation
Authors: Didong Li and David B. Dunson (2019)
https://arxiv.org/abs/1903.00985

Classifiers label data as belonging to one of a set of groups based on input features. It is challenging to obtain accurate classification performance when the feature distributions in the different classes are complex, with nonlinear, overlapping and intersecting supports. This is particularly true when training data are limited. To address this problem, this article proposes a new type of classifier based on obtaining a local approximation to the support of the data within each class in a neighborhood of the feature to be classified, and assigning the feature to the class having the closest support. This general algorithm is referred to as LOcal Manifold Approximation (LOMA) classification. As a simple and theoretically supported special case having excellent performance in a broad variety of examples, we use spheres for local approximation, obtaining a SPherical Approximation (SPA) classifier. We illustrate substantial gains for SPA over competitors on a variety of challenging simulated and real data examples.

## SPA classifier: Code Documentation 

As a realization of LOcal Manifold Approximation (LOMA) classifier,
SPherical Approximation (SPA) classifier is designed for classification when the feature in different classes have complex, nonlinear, overlapping and intersecting support, especially when the training data are limited.  

Here is a list of files:

`SPCA.m`: the function for fitting data by a sphere.
`cls_spherelets.m`: the main function to classify data with given tuning parameters k and d
`SPA_tune.m`: the function to tune parameters k and d given training samples
`FunkyCurves_noise.mat`: An example data set consisting of 1500 points sampled from three entangled funky curves with Gaussian noise. 
`FunkyCurves_demo.m`: Apply SPA to this Funky curves data set, plot the classification accuracy when the training sample size ranges from 150 to 300.

To use the code: 
Input: `X_tr, y_tr, X_te, y_te`
Step 1: run `SPA_tune(X_tr,y_tr)` to get the optimal k and d
Step 2: run `cls_spherelets(X_tr,y_tr,X_te,y_te,k,d)` for classification

In `SPA_tune`, there are two default set of choices for `k` and `d`. 
User may change the default sets, depending on the specific dataset. For example, when the intrinsic dimension of the manifold is know, like the funky curves case `(d=1)`, `d` does not need to be tuned. Similarly, if there is some information about d, the set can be changed accordingly. 
If the dimension of the ambient space is too large, user can replace the “PCA” step in SPCA function by sparse PCA or other efficient alternatives of PCA. 
If the total sample size is too large, user can randomly choose a smaller and computable subset and tune d only (not k) on that subset, since d does not depend on the sample size. Then return to the entire dataset, the only tuning parameter is k, making the tuning process much more efficient.

Some restrictions for tuning parameters:
`d` must be integers between 1 and p-1;
`k` must be integers between 3 and n_min/2, where n_min is the size of the smallest class;
`k` is no smaller than `d+1`.
Depending on the computation time, user can try more/fewer pairs of `k` and `d`.
