# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:17:16 2019

@author: Lando
"""

# https://www.datacamp.com/community/tutorials/probability-distributions-python

# for inline plots in jupyter
%matplotlib inline
# import matplotlib
import matplotlib.pyplot as plt
# for latex equations
from IPython.display import Math, Latex
# for displaying images
from IPython.core.display import Image
# import seaborn
import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})
# import uniform distribution
from scipy.stats import uniform


# Define Function to randomly draw "sample" samples from dist and return the sample means.
def get_sample_means(dist, sample):
    samples = []
    for i in range(1000):
        indices = np.random.choice(dist.shape[0], 1000, replace=False)
        dta = dist[indices]
        mean = dta.mean()
        samples.append(mean)
    return(samples)


## Uniform Distribution
#######################
n = 500000
start = 10
width = 20
data_uniform = uniform.rvs(size=n, loc = start, scale=width)

# Get Mean
data_uniform.mean()

# Plot the Distribution
ax = sns.distplot(data_uniform,
                  bins=50,
                  kde=True
                  )
ax.set(xlabel='Uniform Distribution ', ylabel='Frequency')


# Randomly Sample..
uniform_means = get_sample_means(data_uniform, 1000)

# Plot the Distribution
ax = sns.distplot(uniform_means,
                  bins=50,
                  kde=True
                  )
ax.set(xlabel='Samples Distribution', ylabel='Frequency')


## Gamma Distribution
#####################
from scipy.stats import gamma
data_gamma = gamma.rvs(a=1, size=500000)

# Plot it
ax = sns.distplot(data_gamma,
                  kde=True,
                  bins=50)
ax.set(xlabel='Gamma Distribution', ylabel='Frequency')

# Pull Samples and calculate mean

gamma_means = get_sample_means(data_gamma, 1000)

# Plot the Distribution
ax = sns.distplot(gamma_means,
                  bins=50,
                  kde=True
                  )
ax.set(xlabel='Samples Distribution', ylabel='Frequency')


## Exponential Distribution
###########################
from scipy.stats import expon
data_expon = expon.rvs(scale=3,loc=0,size=500000)

# Plot it
ax = sns.distplot(data_expon,
                  kde=True,
                  bins=100)
ax.set(xlabel='Exponential Distribution', ylabel='Frequency')


# Pull Samples and calculate mean
exp_mean = get_sample_means(data_expon, 1000)


# Plot the Distribution
ax = sns.distplot(exp_mean,
                  bins=50,
                  kde=True
                  )
ax.set(xlabel='Samples Distribution', ylabel='Frequency')

# Normal Distribution
#####################
from scipy.stats import norm
# generate random numbers from N(0,1)
data_normal = norm.rvs(size=10000,loc=0,scale=1)

# Pull Samples and calculate mean
norm_mean = get_sample_means(data_normal, 1000)


# Plot the Distribution
ax = sns.distplot(norm_mean,
                  bins=50,
                  kde=True
                  )
ax.set(xlabel='Samples Distribution', ylabel='Frequency')

#######################################################
# Discrete Distributions
#######################################################

## Poisson Distribution
#######################
from scipy.stats import poisson
data_poisson = poisson.rvs(mu=3, size=10000)

## Plot it
ax = sns.distplot(data_poisson,
                  bins=30,
                  kde=False)
ax.set(xlabel='Poisson Distribution', ylabel='Frequency')


# Pull Samples and calculate mean
poisson_mean = get_sample_means(data_poisson, 1000)


# Plot the Distribution
ax = sns.distplot(poisson_mean,
                  bins=50,
                  kde=True
                  )
ax.set(xlabel='Samples Distribution', ylabel='Frequency')
















