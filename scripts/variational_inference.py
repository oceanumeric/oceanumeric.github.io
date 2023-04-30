# %%
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture



def figure1():
    # set seed  
    np.random.seed(57)
    # sample size 100
    n = 100
    # sample mean 1 and 10
    mu1, mu2 = 1, 10
    # use same standard deviation 1
    sigma = 1
    # generate two normal distributions
    x1 = np.random.normal(mu1, sigma, n)
    x2 = np.random.normal(mu2, sigma, n)

    # combine two distributions
    x = np.concatenate((x1, x2))


    # plot the distributions
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(x[:n], np.zeros_like(x[:n]),
               alpha=0.5, marker=2, color="green")
    ax.scatter(x[n:], np.zeros_like(x[n:]),
               alpha=0.5, marker=2, color="#6F6CAE")
    _ = ax.set_yticks([])
    sns.histplot(x[:n], color="green", alpha=0.5,
                    kde=True,  ax=ax)
    sns.histplot(x[n:], color="#6F6CAE", alpha=0.5,
                    kde=True, ax=ax)
    ax.set_title("Two normal distributions")
    # add legend
    ax.legend(["$\mathcal{N}(1, 1)$", "$\mathcal{N}(10, 1)$"],
                        frameon=False)


class GMM:
    """
    Gaussian Mixture Model with EM algorithm
    
    It is a semi-supervised learning algorithm, which means user need to provide 
    the number of clusters.
    """
    
    def __init__(self, X, k=2):
        # set x as array
        X = np.array(X)
        self.n, self.m = X.shape  # n: sample size, m: feature size
        self.data = X.copy()
        self.k = k  # number of clusters
        
        # initialize parameters for EM algorithm
        
        # initialize the mean vector as random vector for each cluster
        self.mean = np.random.rand(self.k, self.m)
        # initialize the covariance matrix as identity matrix for each cluster
        self.sigma = np.array([np.eye(self.m)] * self.k)
        # initialize the prior probability as equal for each cluster
        self.phi = np.ones(self.k) / self.k
        # initialize the posterior probability as zero
        self.w = np.zeros((self.n, self.k))
        
    def _gaussian(self, x, mean, sigma):
        
        pdf = sp.stats.multivariate_normal.pdf(x, mean=mean, cov=sigma)
        
        return pdf
        
    
    def _e_step(self):
        # calculate the posterior probability based on equation (28)
        for i in range(self.n):
            density = 0 # initialize the density
            for j in range(self.k):
                temp = self.phi[j] * self._gaussian(self.data[i],
                                                        self.mean[j],
                                                        self.sigma[j])
                # update the density (marginal probability)
                density += temp
                # update the posterior probability (joint probability)
                self.w[i, j] = temp
            # normalize the posterior probability
            self.w[i] /= density
            # assert the sum of posterior probability is 1
            assert np.isclose(np.sum(self.w[i]), 1)
            
    def _m_step(self):
        # update the parameters
        for j in range(self.k):
            # get the sum of posterior probability for each cluster
            sum_w = np.sum(self.w[:, j])
            # update the prior probability based on equation (27)
            self.phi[j] = sum_w / self.n
            # update the mean vector based on equation (23)
            self.mean[j] = np.sum(self.w[:, j].reshape(-1, 1) * self.data,
                                                    axis=0) / sum_w
            # update the covariance matrix based on equation (24)
            self.sigma[j] = np.dot(
                    (self.w[:, j].reshape(-1, 1) * (self.data - self.mean[j])).T,
                                (self.data - self.mean[j])) / sum_w
            
    def _fit(self):
        self._e_step()
        self._m_step()
        
    def loglikelihood(self):
        # calculate the loglikelihood based on equation (21)
        ll = 0
        for i in range(self.n):
            temp = 0
            for j in range(self.k):
                temp += self.phi[j] * self._gaussian(self.data[i],
                                                        self.mean[j],
                                                        self.sigma[j])
            ll += np.log(temp)
            
        return ll
    
    def fit(self, max_iter=100, tol=1e-6):
        # initialize the loglikelihood
        ll = [self.loglikelihood()]
        # initialize the number of iteration
        i = 0
        # initialize the difference between two loglikelihood
        diff = 1
        # iterate until the difference is less than tolerance or reach the max iteration
        while diff > tol and i < max_iter:
            # update the parameters
            self._fit()
            # calculate the loglikelihood
            ll.append(self.loglikelihood())
            # calculate the difference
            diff = np.abs(ll[-1] - ll[-2])
            # update the number of iteration
            i += 1
            # print the loglikelihood every 2 iterations
            if i % 2 == 0:
                print("Iteration: {}, loglikelihood: {}".format(i, ll[-1]))
    

def test_gmm():
    """
    Test GMM class
    """
    # set seed
    np.random.seed(57)
    # generate a mixture of two normal distributions
    # with sample size 30 and 70 respectively
    # one normal distribution has mean (0, 3) and the other has mean (10, 5)
    # one normal distribution has covariance matrix [[0.5, 0], [0, 0.8]]
    # the other normal distribution has identity covariance matrix
    
    X = np.concatenate(
                (np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 30),
                    np.random.multivariate_normal([10, 5], np.eye(2), 70))
                )
    print("If we treat the data as one cluster:")
    print(X.shape, X.mean(axis=0), X.std(axis=0))
    
    print("-" * 60)
    print("Now, we use GMM to fit the data with 2 clusters:")
    
    gmm = GMM(X, k=2)
    gmm.fit()
    
    # print out the parameters
    print("Mean: \n", gmm.mean)
    print("Covariance matrix: \n", gmm.sigma)
    print("Prior probability: \n", gmm.phi)
    # print("Posterior probability: \n", gmm.w)
    
    # another test with 3 clusters and 3 features
    X = np.concatenate(
                (np.random.multivariate_normal([0, 3, 5], [[0.5, 0, 0], [0, 0.8, 0], [0, 0, 1]], 30),
                    np.random.multivariate_normal([10, 5, 3], np.eye(3), 70),
                    np.random.multivariate_normal([5, 10, 15], [[0.5, 0, 0], [0, 0.8, 0], [0, 0, 1]], 50)
                    )
                )
    print("Another test with 3 clusters and 3 features:")
    print(X.shape, X.mean(axis=0), X.std(axis=0))
    
    print("-" * 60)
    print("Now, we use GMM to fit the data with 3 clusters:")
    gmm = GMM(X, k=3)
    gmm.fit(tol=1e-10)
    
    # print out the parameters
    print("Mean: \n", gmm.mean)
    print("Covariance matrix: \n", gmm.sigma)
    print("Prior probability: \n", gmm.phi)
    
    # test with sckit-learn
    print("-" * 60)
    print("Now, we use sckit-learn to fit the data with 3 clusters:")
    gm = GaussianMixture(n_components=3, covariance_type='full').fit(X)
    # print out the parameters
    print("Mean: \n", gm.means_)
    print("Covariance matrix: \n", gm.covariances_)
    print("Prior probability: \n", gm.weights_)
    
    
    
        
if __name__ == "__main__":
    print(os.getcwd())
    # figure1()
    # set retina display in jupyter notebook
    # %config InlineBackend.figure_format = 'retina'
    test_gmm()
# %%
