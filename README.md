# Code for Group Meritocratic Fairness in Linear Contextual Bandits

This repo contains the code for the experiments in the paper [_Group Meritocratic Fairness in Linear Contextual Bandits_](https://arxiv.org/abs/2206.03150),
to appear at NeurIPS 2022.

## What is group meritocratic fairness?

Picture a hiring scenario where at each round, an employer has to select a candidate 
from a pool of candidates to perform a job and after that it receives a (noisy) reward which 
is a measure of the candidate's performance. 
Imagine also that each candidate belongs to a sensitive group (e.g. ethnicity or gender). 

Candidates from disdvantaged groups could be excluded by an employer whose goal is just to maximise the reward.
To give a chance also to those candidates, the employer 
could instead aim at choosing the candidate with best **relative rank**, i.e. a measures of how good the candidate performs among
others from the same sensitive group. We call such a policy **group meritocratic fair**.

In [our paper](https://arxiv.org/abs/2206.03150) we assume that there exist a linear relation between
the true reward and the feature vector encoding the properties of each candidate.
Furthermore, we show under some assumptions on the distribution of the candidates and
on the noise in the rewards, that a greedy policy (_Fair-Greedy_ and _Fair-Greedy V2_ in the paper) can efficiently learn
to be group meritocratic fair. Our Fair-Greedy policy combines ridge regression with 
an empirical CDF to estimate the relative rank of each candidate. 
The policy simply selects the candidate with the best estimate of the relative rank: no confidence intervals are used.



# How to run the experiments
First, install the packages in [requirements.txt](requirements.txt).
Then, run one of these files to execute the experiments:
- [simulation.py](simulation.py) for a synthetic simulation with diverse distibutions of rewards (weighted variants of [Irwin–Hall](https://en.wikipedia.org/wiki/Irwin–Hall_distribution)).
- [adult.py](adult.py) for an experiment using the US Census Data with a linear estimate of the income as the true reward.
- [adult_multigroup.py](adult_multigroup.py) like the one above but the sensitive group is sampled randomly together with the context, which is a more realistic scenario.

Adjust the parameters defined in the body of the function `main()` to change the number of rounds and other things. 
See the content of the above files for more details.

## Code structure

Python files with the suffix `_multigroup` contain the implementation for the (more realistic) case where the sensitive group is 
sampled together with contexts.

Loading and preprocessing of the US Census data is in [data.py](data.py) and [data_multigroup.py](data_multigroup.py) and relies on
[folktables](https://github.com/zykls/folktables).

[policies.py](policies.py) and [policies_multigroup.py](policies_multigroup.py) Contain the implementation of Fair-Greedy
and Fair-Greedy V2 in the class `FairGreedy`, and of other baselines policies like Uniform Random, OFUL and Greedy.
It also contains the class representing a bandit problem and the method used to test a given policy on such problems.



Plotting functionality used to generate figures is in [plot.py](plot.py)

## Cite us

```
@article{grazzi2022group,
  title={Group Meritocratic Fairness in Linear Contextual Bandits},
  author={Grazzi, Riccardo and Akhavan, Arya and Falk, John Isak Texas and Cella, Leonardo and Pontil, Massimiliano},
  journal={arXiv preprint arXiv:2206.03150},
  year={2022}
}
```