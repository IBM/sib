<!-- This should be the location of the title of the repository, normally the short name -->
# sequential Information Bottleneck (sIB)


<!-- Build Status, is a great thing to have at the top of your repository, it shows that you take your CI/CD as first class citizens -->
<!-- [![Build Status](https://travis-ci.org/jjasghar/ibm-cloud-cli.svg?branch=master)](https://travis-ci.org/jjasghar/ibm-cloud-cli) -->
[![GitHub Actions CI status](https://github.com/ibm/sib/workflows/Build/badge.svg)](https://github.com/ibm/sib/actions)

<!-- Not always needed, but a scope helps the user understand in a short sentence like below, why this repo exists -->
## Scope

This project provides an efficient implementation of the text clustering algorithm "sequential Information Bottleneck" (sIB), introduced by [Slonim, Friedman and Tishby (2002)](#reference). The project is packaged as a python library with a cython-wrapped C++ extension for the partition optimization code. A pure python implementation is included as well. The implementation is documented [here](./docs/sib_implementation.pdf).


## Installation

```pip install sib-clustering```


<!-- A more detailed Usage or detailed explanation of the repository here -->
## Usage
The main class in this library is `SIB`, which implements the clustering interface of [SciKit Learn][sklearn], providing methods such as `fit()`, `fit_transform()`, `fit_predict()`, etc. 

The sample code below clusters the 18.8K documents of the 20-News-Groups dataset into 20 clusters:

```python

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
from sib import SIB

# read the dataset
dataset = fetch_20newsgroups(subset='all', categories=None,
                             shuffle=True, random_state=256)

gold_labels = dataset.target
n_clusters = np.unique(gold_labels).shape[0]

# create count vectors using the 10K most frequent words
vectorizer = CountVectorizer(max_features=10000)
X = vectorizer.fit_transform(dataset.data)

# SIB initialization and clustering; parameters:
# perform 10 random initializations (n_init=10); the best one is returned.
# up to 15 optimization iterations in each initialization (max_iter=15)
# use all cores in the running machine for parallel execution (n_jobs=-1)
sib = SIB(n_clusters=n_clusters, random_state=128, n_init=10,
          n_jobs=-1, max_iter=15, verbose=True)
sib.fit(X)

# report standard clustering metrics
print("Homogeneity: %0.3f" % metrics.homogeneity_score(gold_labels, sib.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(gold_labels, sib.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(gold_labels, sib.labels_))
print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(gold_labels, sib.labels_))
```

Expected result:
```
sIB information stats on best partition:
	I(T;Y) = 0.5685, H(T) = 4.1987
	I(T;Y)/I(X;Y) = 0.1468
	H(T)/H(X) = 0.2956
Homogeneity: 0.616
Completeness: 0.633
V-measure: 0.624
Adjusted Rand-Index: 0.507
```

See the [Examples](examples) directory for more illustrations and a comparison against K-Means.


<!-- License and Authors is optional here, but gives you the ability to highlight who is involed in the project -->
## License

```text
Copyright IBM Corporation 2020

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

```

If you would like to see the detailed LICENSE click [here](LICENSE).


## Authors 
- Algorithm and pseudo-code: [Slonim, Friedman and Tishby (2002)](#reference)
- First python implementation: [Daniel Hershcovich](https://danielhers.github.io/)
- Optimization work: Assaf Toledo and Elad Venezian
- Development and maintenance: [Assaf Toledo](https://github.com/assaftibm)


<!-- Questions can be useful but optional, this gives you a place to say, "This is how to contact this project maintainers or create PRs -->
If you have any questions or issues you can create a new [issue here][issues].

## Reference
N. Slonim, N. Friedman, and N. Tishby (2002). Unsupervised Document Classification using Sequential Information Maximization. SIGIR 2002.
https://dl.acm.org/doi/abs/10.1145/564376.564401


[issues]: https://github.com/IBM/sib/issues/new
[sklearn]: https://scikit-learn.org
