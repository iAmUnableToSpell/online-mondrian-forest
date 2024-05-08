# Online Mondrian Forests

# Abstract

The majority of the work completed in this report focuses on an implementation of Mondrian Forests, first introduced by Lakshminarayanan et. al. ${[1]}$ in their 2015 paper discussing them as a potential improvement on existing random forest algorithms due to their capability to adapt to new data without rebuilding the ensemble as a whole.  

The fundamental concept driving Mondrian Forests is the Mondrian Process ${[2]}$, a nonparametric prior on random partitions of the input space. This process forms the basis for constructing decision trees in an online fashion, with each incoming data point influencing the evolving model. The algorithm incorporates elements of random forests, where multiple decision trees are created, and the final prediction is an average over all the trees.

The Mondrian Forest is constructed similarly to a random forest: Given training data $D_{1:N}$, a collection of $M$ mondrian trees (described below) are constructed, each tree $m$ outputting a prediction $p_{T_m}(y|x, D_{1:N})$ for each class label $y$, for an data point $x$ ${[1]}$. The prediction for the ensemble as a whole is then the average $\frac{1}{M} \sum^{M}_{m=1}p_{T_m}(y|x, D_{1:N})$ ${[1]}$. Mondrian forests integrate new data at each example in $N$, where each mondrian tree in $T$ is updated. This update is performed by sampling an extended tree $T'$ from a distribution $MTx(\lambda, T, D_{N+1})$ This sampling searches for a $T'$ such that $T'=T$ on $D_{1:N-1}$ and $T'$ is distributed according to $MT(\lambda, D_{1:N})$.

In more natural language, the goal is to construct a distribution $MTx$ at each step such that the new tree $T'$ functions the same as $T$ on previous data $D_{1:N}$, but follows the same distribution as if it had been trained on the dataset $D_{1:N+1}$.

The key insight made by Lakshminarayanan et. al. is that if we are able to perform the above steps, the resultant set of online trained mondrian trees would be functionally identical to a set of the same sized trained in batch learning, i.e. the trees produced are agnostic to the order the training data is in.

# Survey

## Mondrian Processes

First introduced by Roy and Teh ${[2]}$, mondrian processes may be described as a recursive process which makes cuts along dimensions in an input space. Let us consider again the one dimensional unit interval case $\mathbb{R}^D = [0,1]$. A mondrian process starts with an initial "budget" $\lambda$ and makes a sequence of partitions, each one with a cost. Eventually, $\lambda$ will be depleted and we will have a finite partition $m$. The cost of a given partition $I$ is given by $E_I$, which is a distribution along the inverse mean of the interval being partitioned. In other words: $E_{[0,1]} \sim \text{Exp}(1)$. Upon obtaining this cost, we can reduce $\lambda$ like so: $\lambda'= \lambda − E[0,1].$ If $\lambda$ falls below 0, we do not partition and instead return $[0,1]$. Otherwise, we make a cut at random, splitting into subintervals A and B, and recurse on each with a budget of $\lambda'$. When working in higher dimensions, we still cut only one dimension at a time, with the probability of choosing each dimension weighted by the domain of that dimension. 

This definition gives mondrian processes some interesting and useful characteristics. First, the subtrees of any given partition are conditionally independent given $\lambda'$, the dimension cut, and threshold at the first cut. Secondly, given a draw from a mondrian on some domain, the partition on any subdomain has the same distribution as if we sampled a mondrian process directly on that subdomain. This second point is key, because it means we can apply new examples to subdomains without partitioning anything beyond that subdomain.  

___

## Decision Trees and Notation

Decision trees by definition consist of nested partitioning operations on  $\mathbb{R}^D$, the domain of our training data. Decision trees must be finite in depth, have a single root, and follow the binary tree structure such that each non-root node has exactly one parent, and each node $j$ is the parent of either zero or two child nodes, denoted $right(j)$ and $left(j)$.

If we imagine each node as a partition, it naturally follows that each node $j$ is associated with a block of the input space, denoted $B_j$, where $B_j \subseteq \mathbb{R}^D$. Each node also represents a partition of its parent along a given dimension $\delta$ at at a given point $\xi$. Expressed mathematically, we have: 

$$B_{left(j)} := \{x \in B_j : x_{\delta_j} \leq \xi_j\} \text{ and } B_{right(j)} := \{x \in B_j : x_{\delta_j} \gt \xi_j\}$$ 

For instance, in a one dimensional input space $[0,1]$, the root node $r$ would be associated with block $B_r = \mathbb{R}^D$. If we take $\xi_r = 5$ (Since our data is one-dimensional $\delta_r$ = 0), we know $B_{right(r)}$ is $[0,0.5]$, and that $B_{left(r)}$ is $(0.5,1]$. 

To describe the mondrian adaptation to decision trees, the authors introduce some notation ${[1]}$ that will be useful later:

Let $parent(j)$ denote the parent of node $j$.

Let $N(j)$ denote the indices of examples in $B_j$ i.e. $N(j) = \{ n \in \{1...N \} : x_n \in B_j \}$.

Let $l^x_{jd}$ and $u^x_{jd}$ denote the lower and upper bounds of examples in $B_j$ along dimension $d$.

Let $B^x_j$ denote the smallest rectangle which encloses the points in $B_j$.

___

## Adaptation to Mondrian Trees

Lakshminarayanan et. al. propose a set of mondrian trees on the finite range of attributes in $D$. As such, we can represent a mondrian decision tree with 4 values $(T, \delta, \xi, \tau)$, where $(T, \delta, and \xi)$ represent a decision tree and $\tau = \{\tau_j\}_{j \in T}$ represents a split time for node $j$. Split times monotonically increase with depth, and $\tau_{\text{parent}(r)} = 0$.

Given a positive budget $\lambda$ and training data $D_{1:n}$, the process for sampling trees can be defined in 2 algorithms:

___

### Algorithm 1: SampleMondrianTree($\lambda, D_{1:n}$)

$\text{T} = \emptyset$

$leaves(\text{T}) = \emptyset$

$\delta = \emptyset$

$\xi = \emptyset$

$\tau = \emptyset$

$N(r) = \{1,2....n\}$

SampleMondrianBlock($\lambda$, $D_{N(r)}, r$)

___

### Algorithm 2: SampleMondrianBlock($\lambda, D_{N(j)}, j$)

Add $j$ to $\text{T}$ 

For all $d$, set $l_{jd}^{x} = min(X_{N(j), d}), u_{jd}^{x} = max(X_{N(j), d})$

Sample E from expontential distribution with rate $\sum_d(u_{jd}^{x} - l_{jd}^{x})$

if $\tau_{parent(j)} + E < \lambda$ do:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $\tau_j = \tau_{parent(j)} + E$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample split dimension $\delta_j$, selecting $d$ with probability proportional to $u_{jd}^{x} - l_{jd}^{x}$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample split location $\xi_j$ from $[u_{jd}^{x} , l_{jd}^{x}]$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $N(left(j)) = \{n \in N : X_{n, \delta_j} \leq \xi_j\}$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $N(right(j)) = \{n \in N : X_{n, \delta_j} \gt \xi_j\}$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SampleMondrianBlock($left(j), D_{N(left(j))}, \lambda$)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SampleMondrianBlock($right(j), D_{N(right(j))}, \lambda$)

else do:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $\tau_j = \lambda$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Add $j$ to $leaves(T)$

___

Algorithm 1, SampleMondrianTree, should be called to instantiate a mondrian tree. After that, algorithm 2 should recurse down the tree, building it as it goes. For each new node, we first compute $u_{jd}^{x}$ and $ l_{jd}^{x}$ for each dimension. E is then sampled from the linear dimension $B^x_r$, the sum $\sum_d(u_{jd}^{x} - l_{jd}^{x})$. Since $\tau_{parent(r)} = 0$, $E + \tau_{parent(r)} = E$. If we are above our budget $\lambda$, we assign $j$ as a leaf and return. In the case that $j$ is an internal node, we sample a dimension $\delta_j$ from the distribution over all d for $u_{jd}^{x} - l_{jd}^{x}$, followed by sampling a split point $\xi_j$ from the interval $[u_{jd}^{x}, l_{jd}^{x}]$ for our selected $\delta_j$. Finally, we perform the cut implied by $\delta_j$ and $\xi_j$ and recurse on the resulting partitions.

Thanks to the projective rules of mondrian processes defined by Roy and Teh and discussed in section 2.1, we can demonstrate some a useful property of the tree we have defined. If we sample a mondrian tree T from MT($\lambda$, $F$), where $F$ represents all finite sets of examples, and then restrict $T$ to some $F' \subseteq F$, then the restricted tree $T'$ has distribution MT($\lambda$, $F'$). This gives us a way to extend a mondrian block built on dataset $D_{1:N}$ to a larger dataset $D_{1:N+1}$. Lakshminarayanan et. al. exploit this property to incrementally grow a mondrian tree on a stream of examples using the following two algorithms:

___

### Algorithm 3: ExtendMondrianTree($T, \lambda, D$)

Input params: Tree T = (T, $\delta$, $\xi$, $\tau$), new example $D = (x, y)$

ExtendMondrianBlock($T, \lambda, r, D$)

___

### Algorithm 4: ExtendMondrianBlock($T, \lambda, j, D$)

Set $e^l = max(l^x_j - x, 0)$ and $e^u = max(x-u^x_j, 0)$

Sample E from $\sum_d(e^l_d + e^u_d)$

if $\tau_{parent(j)} + E \lt \tau_j$ do:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample split dimension $\delta$ from all d with probability proportional to $e^l_d + e^u_d$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample split location $\xi$ from $[u^x_{j, \delta}, x_{\delta}]$ if $x_{\delta} \gt u^x_{j, \delta}$ else from $[x_{\delta}, l^x_{j, \delta}]$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Insert a new node $j'$ above $j$, and a new leaf $j''$ as a peer, where:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\delta_{j'} = \delta$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\xi_{j'}=\xi$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\tau_{j'}=\tau_{parent(j)} + E$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$l^x_{j'} = min(l^x_j, x)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$u^x_{j'} = max(u^x_j, x)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$j'' = left(j')$ iff $x_{\delta_{j'}} \leq \xi_{j'}$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SampleMondrianBlock($\lambda, D, j''$)

else do: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;update $l^x_j \leftarrow  min(l^x_j, x)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;update $u^x_j \leftarrow  max(u^x_j, x)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if $j \notin leaves(T)$ do:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if $x_{\delta_j} \leq \xi_j$ child($j$) = left($j$) else child($j$) = right($j$)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ExtendMondrianBlock($T, \lambda, j, D$)

___

Similar to other decision tree growth algorithms, ExtendMondrianBlock halts partitioning on blocks when all labels are identical; if $x \in B_j$ of a paused block $j$ and has the same label, we remain paused. 

___

### Prediction with Mondrian Trees

For a given tree $T$ and example $x$, the prediction is given by $p_T(y | x, D_{1:N})$. Since we know the input space is distributed among the input nodes by our definition of mondrian processes, we know there will be some leaf node $j \in leaves(T)$ such that $x \in B_j$. 

Similar to other decision tree algorithms, we return the class as an empirical distribution of examples which fall in $B_j$. We also may want to smooth this value with those in neighbors $j'$ of $j$. To do this, Lakshminarayanan et. al. implemented a hierarchical Bayesian approach wherein each node is mapped to its label distribution and a prior is selected such that the label distribution of a node is similar to that of its parents. 

For every $j \in T$, let $G_j$ denote the distribution of labels in $B_j$. Therefore, for class $k$ the predictive distribution $p(y|x, T,G) = G_{leaf(x)}$

To predict the label of a new example $x$ using this tree, we first check if $x \in B^x_j$ for some $j \in leaves(T)$. In this case, the predictive distribution is as simple as $G_{leaf(x)}$. Otherwise, $x$ must be integrated into the tree. To do this, we call algorithm 3 ExtendMondrianTree($T, \lambda, x$). 

___

# Methods

Several datasets were chosen for this project in order to expose the algorithm to a wide variety of problems. A few sample datasets used for experimentation were placed in /datasets, from which the model pulls from. 

Mondrian Forests were implemented in /randomforests following the algorithms and explanations above, in python through the use of several popular libraries. The implementation involves an abstract classifier type, of which Mondrian Decision Forest is a subtype. This was done to facilitate easy comparison between both common implementations of MF and other decision tree algorithms. 

Finally, several jupyter notebooks were created in /notebooks to facilitate efficient fine-tuning and experimentation. These were used both to refine hyperparameters and to perform the comparisons discussed above. Please note that several of the notebooks take quite a bit of time to run, on account of various hyperparameter choices. Referencing Bonab and Can [3], the ideal number of classifiers for ensemble models is simply the number of classes, but that assumes perfectly independent classifiers, which are functionally infeasible. As a result, Bonab and Can recommend increasing the number of classifiers inversely relative to their independence. Additional hyperparameter tests were implemented over a range of arbitrary values, with the range extended should one extreme prove most effective. 


# Research

The purpose of mondrian decision trees is to sacrifice accuracy for decreased training time. As a result, we might term mondrian trees as weak classifiers, of which we form an ensemble to compensate. Increasing the accuracy was the primary goal of this section, and several methods were attempted (though some of them must be disqualified due to converting our online learner into a batch caching learner). 

First, it must be noted that most of the common strategies for increasing the accuracy of ensemble classifiers cannot be directly applied in an online learning context. Additionally, though increasing accuracy was the primary goal, methods were implemented to improve training time after the comparative section of this project. 

## Online Bagging

Bagging as applied to online learners is not terribly complicated, and following the implementation laid out by Oza and Russell [4] a bagging algorithm was implemented. 

___

### Algorithm 5: OnlineBaggingFit($x, y$)

for $tree$ in $T$ do:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;set $k = Poisson(1)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for $i$ in $range(k)$ do:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$tree.fit(x, y)$

___

This algorithm is incredibly simple, just running each example through each tree a variable number of times based on the Poisson distribution for 1

## Rolling Weight Boosting

The mondrian tree growth process was put forward by Lakshminarayanan et. al. as a trade off between accuracy and training efficiency. Because we construct a forest instead of a single tree, accuracy loss is somewhat mitigated. In an online context, training efficiency might be a critical metric, so the proposed MDF use cases are situations where accuracy is not critical and where training efficiency is. These cases exist, but they are rare; the goal of this research extension is then to improve upon the accuracy of the MDF without sacrificing overmuch training efficiency. 

Rolling weight boosting is inspired by traditional online boosting algorithms described by Oza and Russell [4]. We introduce $e_w$, a weight specific to a single example. We also introduce a list $r_t^c$ and $r_t^w$, holding a sum of correct and incorrect weights respectively for each classifier $t$ in $T$. 

___

### Algorithm 6: OnlineBoostingFit($x, y$)

Initialize $e_w = 1$

for $tree$ in $T$ do:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;set $k = Poisson(e_w)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for $i$ in $range(k)$ do:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$tree.fit(x, y)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$prob = tree.predict(x)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if $argmax(prob) \ne y$ do:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;update $ r_t^w \leftarrow  r_t^w +  e_w$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;update $e_w \leftarrow e_w * (\frac{i }{2r_t^w})$ 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else do:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;update $ r_t^c \leftarrow  r_t^c +  e_w$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;update $e_w \leftarrow e_w * (\frac{i }{2r_t^c})$ 

___

This procedure is executed for each new example, and sourced roughly from [4]. For each $t$ in $T$, we first fit the tree $t$ to $x, y$ as described by Lakshminarayanan et. al, and then determine if the tree now classifies $x$ correctly. 

We add the weight to the sum of weights for either correct or incorrect classifications, such that $r_t^w$ contains an index for each decision tree, holding the sum of weights of wrongly classified examples for that tree. Similarly, we construct a sum of correctly classified examples. 

After updating this, we then update the weight of the given example to be used in the remaining classifiers. We take $i$ into this formula so the early examples don't dominate others in terms of their effects on weighted sums. 

Then, when it comes time to classify our examples, we calculate a scaling factor equivalent to 

$$\eta_t = \frac{r_t^w}{r_t^w + r_t^c}$$

$$\beta_t = log(\frac{1-\eta_t}{\eta_t})$$

and weight the output of each $t$ in $T$ by $\beta_t$. 

The aim of this algorithm is to stabilize the random nature of mondrian forests. Those trees that correctly predict examples that others do not experience a greater positive change in ${r_t^c}$, and thus are rewarded in the next example. Since the change in $e_w$ is weighted by the ${r_t^w}$ or ${r_t^c}$ of the tree currently classifying it, if a classifier that is generally correct happens to be wrong (due to some not-yet-learned pattern), other trees which do predict it correctly have their weights increased such that each example adjust the weights of each tree to properly classify it.

# Results, Analysis, and Discussion

## Mondrian Forest Evaluation

Because of the notable lack of an efficient search heuristic for split criterions, each individual mondrian tree tends to be less powerful than a tree trained using an algorithm like ID3, which takes information gain into account. Conversely, mondrian trees are much more efficient to train, and are only linearly more complex with the addition of more attributes. As such, mondrian trees lend themselves to ensemble learning to a decent degree, but see quickly diminishing returns as n_tree increases, as we can see from our n_tree tests in tables 1-3. The reasoning for these quickly diminishing improvements is likely because the ExtendMondrianBlock algorithm takes random splits, and so the individual classifiers are not very closely related. As Bonab and Can noted [3], the ideal number of classifiers is the number of class labels when the classifiers are not interrelated, which supports the idea that mondrian trees are fairly independent and therefore have an ideal classifier count only slightly higher than the number of labels. 

### IRIS Dataset n_tree Comparison (Table 1)

| n_tree|   2  |  4   |  8   |  16  |
| :---: |:----:|:----:|:----:|:----:|
| Acc   | 0.75 | 0.85 | 0.90 | 0.95 |
| Pre   | 1.00 | 1.00 | 1.00 | 1.00 |
| Rec   | 0.38 | 0.63 | 0.75 | 0.88 |
| Auc   | 0.81 | 0.57 | 0.88 | 0.62 | 
| Time  | 0.02 | 0.03 | 0.06 | 0.14 |

### grape Dataset n_tree Comparison (Table 2)

| n_tree|   2  |  4   |  8   |  16  |
| :---: |:----:|:----:|:----:|:----:|
| Acc   | 0.87 | 0.90 | 0.87 | 0.90 |
| Pre   | 0.83 | 0.85 | 0.83 | 0.85 |
| Rec   | 0.90 | 0.95 | 0.93 | 0.94 |
| Auc   | 0.91 | 0.96 | 0.96 | 0.95 | 
| Time  | 2.95 | 6.64 | 13.9 | 26.6 |

### diabetes5050 Dataset n_tree Comparison (Table 3)

| n_tree|   2  |  4   |  8   |  16  |
| :---: |:----:|:----:|:----:|:----:|
| Acc   | 0.63 | 0.90 | 0.87 | 0.90 |
| Pre   | 0.58 | 0.85 | 0.83 | 0.85 |
| Rec   | 0.88 | 0.95 | 0.93 | 0.94 |
| Auc   | 0.69 | 0.96 | 0.96 | 0.95 | 
| Time  | 2.19 | 6.64 | 13.9 | 26.6 |

For remaining tests, 8 classifiers were used. 

## Rolling Weight Boosting and Bagging tests

When applying rolling weight boosting to the mondrian forest algorithm described by Lakshminarayanan et. al, we find surprisingly encouraging results with regards to model performance, similarly with online bagging.

### IRIS Dataset Rolling Boosting and Bagging Comparison (Table 4)

| Aggregation |  Rolling Boosting | Online Bagging |  Default Mondrian Forest  
| :---: |:----:|:----:|:----:|
| Acc   | 1.00 | 1.00 | 0.90 |
| Pre   | 1.00 | 1.00 | 1.00 |
| Rec   | 1.00 | 1.00 | 0.75 |
| Auc   | 1.00 | 1.00 | 0.88 | 
| Time  | 0.20 | 0.07 | 0.06 |

### grape Dataset Rolling Boosting and Bagging Comparison (Table 5)

| Aggregation |  Rolling Boosting | Online Bagging  |  Default Mondrian Forest  
| :---: |:----:|:----:|:----:|
| Acc   | 0.92 | 0.91 | 0.87 |
| Pre   | 0.92 | 0.92 | 0.83 |
| Rec   | 0.91 | 0.90 | 0.93 |
| Auc   | 0.97 | 0.97 | 0.96 | 
| Time  | 37.8 | 20.4 | 13.9 |

### diabetes5050 Dataset Rolling Boosting and Bagging Comparison (Table 6)

| Aggregation |  Rolling Boosting  | Online Bagging |  Default Mondrian Forest  
| :---: |:----:|:----:|:----:|
| Acc   | 0.62 | 0.65 | 0.67 |
| Pre   | 0.58 | 0.65 | 0.58 |
| Rec   | 0.85 | 0.82 | 0.88 |
| Auc   | 0.70 | 0.73 | 0.69 | 
| Time  | 324.2| 53.2 | 21.9 |

### spambase Dataset Rolling Boosting and Bagging Comparison (Table 7)

| Aggregation |  Rolling Boosting  | Online Bagging |  Default Mondrian Forest  
| :---: |:----:|:----:|:----:|
| Acc   | 0.62 | 0.85 | 0.78 |
| Pre   | 0.58 | 0.80 | 0.80 |
| Rec   | 0.85 | 0.82 | 0.25 |
| Auc   | 0.70 | 0.92 | 0.39 | 
| Time  | 84.2 | 11.3 | 9.93 |

In each dataset, the performance of the mondrian forest was generally increased by the rolling boost algorithm, but the tradeoff for increased training time is worrying. The increase is slight for smaller datasets, but it compounds as more examples are included. Mondrian decision trees are very efficient to grow, but classification is still relatively inefficient, and grows more so as the tree becomes more complex. As a result, for huge datasets, the first operation in RollingWeightFit - the classification of $x$ after $t$ has already fit to it - becomes increasingly expensive over time.

One advantage (unimplemented in this project) of mondrian decision trees is the ability to parallelize fitting over a batch of examples, thanks to the projectivity property of mondrian processes outlined by Roy and Teh [2] and implemented by Lakshminarayanan et. al. [1]. While Rolling Boosting is generally an improvement on the accuracy of mondrian forests, it by nature requires examples to be fit incrementally on one tree at a time, removing the possibility of massive parallelization.

As a result, we cannot say that rolling boosting is truly successful, despite accomplishing our original aim of increasing tree performance, as the increase is minimal and the decrease in training efficiency is too large.

On the other hand, online bagging seems to be incredibly useful, positively influencing nearly every metric on every dataset at the cost of a very small amount of training time. In essence, this algorithm artificially increases the size of our training data, and in doing so causes many more partitions to be made, increasing the overall expressivity of the model. However, we must also consider the fact that we lose the projectivity of mondrian processes with this extension as well, as by weighting each example based on the previous weight, the order which the examples arrive in is now a factor in the resultant model. Because we take a Poisson of the weight and use it to determine how many times each example is fitted, the order of examples now dictates the number of training examples. 

Overall, the results of these experiments were mixed. While the original mondrian tree algorithm performed fairly well, both of the extensions were ultimately held back by the fact that mondrian trees must ensure a specific set of circumstances to maintain projectivity, and the vast majority of feasible modifications to the algorithm break these rules. Ultimately, mondrian decision trees are an effective classification algorithm and should undoubtedly be considered for any online learning problem.

# Bibliography

[1] Lakshminarayanan, B., Roy, D. M., & Teh, Y. W. (2015). Mondrian Forests: Efficient Online Random Forests. arXiv [Stat.ML]. Retrieved from http://arxiv.org/abs/1406.2673

[2] Roy, D. M., & Teh, Y. (2008). The Mondrian Process. In D. Koller, D. Schuurmans, Y. Bengio, & L. Bottou (Eds.), Advances in Neural Information Processing Systems (Vol. 21). Retrieved from https://proceedings.neurips.cc/paper_files/paper/2008/file/fe8c15fed5f808006ce95eddb7366e35-Paper.pdf

[3] Bonab, H. R., & Can, F. (2016). A Theoretical Framework on the Ideal Number of Classifiers for Online Ensembles in Data Streams. Proceedings of the 25th ACM International on Conference on Information and Knowledge Management, 2053–2056. Presented at the Indianapolis, Indiana, USA. doi:10.1145/2983323.2983907

[4] Oza, N.C. & Russell, S.J.. (2001). Online Bagging and Boosting. Proceedings of the Eighth International Workshop on Artificial Intelligence and Statistics, in Proceedings of Machine Learning Research R3:229-236. Available from https://proceedings.mlr.press/r3/oza01a.html. Reissued by PMLR on 31 March 2021.

