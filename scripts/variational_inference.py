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
    # sample_size size 100
    n = 100
    # sample_size mean 1 and 10
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
        self.n, self.m = X.shape  # n: sample_size size, m: feature size
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
    # with sample_size size 30 and 70 respectively
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
    
    

class UGMM:
    """
    Univariate Gaussian Mixture Model
    """
    
    def __init__(self, X, K = 2, sigma = 1):
        self.X = X
        self.K = K
        self.N = X.shape[0]
        self.sigma2 = sigma**2
        
        # initialize the parameters
        # using dirichlet distribution to initialize the prior probability
        # we fix alpha in the range of [1, 10] for initialization
        # it can be changed to other values
        alpha_const = np.random.random()*np.random.randint(1, 10)
        self.phi = np.random.dirichlet([alpha_const]*self.K, size=self.N)
        # initialize the mean from uniform distribution
        self.m = np.random.uniform(min(self.X), max(self.X), self.K)
        # initialize the variance from uniform distribution
        self.s2 = np.random.uniform(0, 1, self.K)
        
    def _get_elbo(self):
        # calculate the evidence lower bound
        # term 1 in euqation (14)
        # although we use sigma^2 in equation (14) but we use s2 in the code
        # because we are not estimating sigma^2 but s2 (variational inference)
        elbo_term1 = np.log(self.s2) - self.m / self.s2
        elbo_term1 = elbo_term1.sum()
        # term is not exactly the same as equation (14)
        # herer we penalize the model with large variance
        # term 2 based on equation (17)
        # again the term is not exactly the same as equation (17)
        # but proportional to it
        elbo_term2 = -0.5 * np.add.outer(self.X**2, self.s2+self.m**2)
        elbo_term2 += np.outer(self.X, self.m)
        elbo_term2 -= np.log(self.phi)
        elbo_term2 *= self.phi
        elbo_term2 = elbo_term2.sum()
        
        return elbo_term1 + elbo_term2
    
    def _update_phi(self):
        t1 = np.outer(self.X, self.m)
        t2 = -(0.5*self.m**2 + 0.5*self.s2)
        exponent = t1 + t2[np.newaxis, :]
        self.phi = np.exp(exponent)
        self.phi = self.phi / self.phi.sum(1)[:, np.newaxis]
        
    def _update_m(self):
        self.m = (self.phi*self.X[:, np.newaxis]).sum(0) * (1/self.sigma2 + self.phi.sum(0))**(-1)
        assert self.m.size == self.K
        #print(self.m)
        self.s2 = (1/self.sigma2 + self.phi.sum(0))**(-1)
        assert self.s2.size == self.K
        
    def _cavi(self):
        self._update_phi()
        self._update_m()
    
    def fit(self, max_iter=100, tol=1e-10):
        # fit the model
        self.elbos = [self._get_elbo()]
        self.track_m = [self.m.copy()]
        self.track_s2 = [self.s2.copy()]
        
        for iter_ in range(1, max_iter+1):
            self._cavi()
            self.track_m.append(self.m.copy())
            self.track_s2.append(self.s2.copy())
            self.elbos.append(self._get_elbo())
            
            if iter_ % 10 == 0:
                print("Iteration: {}, ELBO: {}".format(iter_, self.elbos[-1]))
                
            if np.abs(self.elbos[-1] - self.elbos[-2]) < tol:
                # print convergence information at iteration i
                print("Converged at iteration: {}, ELBO: {}".format(iter_,
                                                                        self.elbos[-1]))
                break
    
    
def test_univariate_gmm():
    # test ugmm with 3 clusters
    np.random.seed(42)
    num_components = 3
    mu_arr = np.random.choice(np.arange(-10, 10, 2),
                        num_components) + np.random.random(num_components)
    sample_size = 1000
    X = np.random.normal(loc=mu_arr[0], scale=1, size=sample_size)
    for i, mu in enumerate(mu_arr[1:]):
        X = np.append(X, np.random.normal(loc=mu, scale=1, size=sample_size))
        
    # plot the data
    fig, ax = plt.subplots(figsize=(15, 4))
    sns.histplot(X[:sample_size], ax=ax, kde=True)
    sns.histplot(X[sample_size:sample_size*2], ax=ax, kde=True)
    sns.histplot(X[sample_size*2:], ax=ax, kde=True)
    
    # initialize the model
    ugmm = UGMM(X, K=3)
    ugmm.fit()
    
    # print out the true mean and estimated mean
    print("True mean: \n", sorted(mu_arr))
    print("Estimated mean: \n", sorted(ugmm.m))
                
        
if __name__ == "__main__":
    print(os.getcwd())
    # figure1()
    # set retina display in jupyter notebook
    # %config InlineBackend.figure_format = 'retina'
    # test_gmm()
    test_univariate_gmm()
    
# %%
