import numpy as np
import pandas as pd
import copy
import numpy as np
from scipy.optimize import curve_fit, minimize
import copy
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import FactorAnalysis
import warnings 


EPS=1e-6


def gen_data(sample_size=150,l=10,D=50,lam=0.2):
    z= np.random.multivariate_normal(np.zeros(l), np.eye(l), sample_size)
    A=np.random.rand(D,l)-0.5
    mu=np.random.rand(D)+5
    # print(mu.size)
    W=np.diag(np.abs(np.random.rand(D))+1)
    x=[]
    for i in range(sample_size):
        x.append( np.random.multivariate_normal(A.dot(z[i])+mu, W, 1)[0])
    x=np.array(x)
    p=np.exp(-lam*x*x )
    h=np.random.binomial(1,p)
    # print(h)
    y=x*(1-h)
    # print(z.shape)
    # print(x.shape)
    return z, y


def doubel_exp(x,Lam):
    return np.exp(-Lam*(x**2))

def index2d(x,i_x,i_y):
    x_index=[]
    for i in range(len(i_x)):
        if i_x[i]!=0:
            x_index.append(i)
    y_index=[]
    for i in range(len(i_y)):
        if i_y[i]!=0:
            y_index.append(i)
    return x[np.ix_(x_index,y_index)]

def set_value2d(x,values,i_x,i_y):
    x_index=[]
    for i in range(len(i_x)):
        if i_x[i]!=0:
            x_index.append(i)
    y_index=[]
    for i in range(len(i_y)):
        if i_y[i]!=0:
            y_index.append(i)
    # i_x=[i*i_x[i] for i in range(len(i_x)) ]
    # i_y=[i*i_y[i] for i in range(len(i_y)) ]
    # print(i_x)
    x[np.ix_(x_index,y_index)]=values
    
    return x

def decayCoefObjectiveFn(x, Y, EX2):
	"""
	Computes the objective function for terms involving lambda in the M-step.
	Checked.
	Input:
	x: value of lambda
	Y: the matrix of observed values
	EX2: the matrix of values of EX2 estimated in the E-step.
	Returns:
	obj: value of objective function
	grad: gradient
	"""
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		y_squared = Y ** 2
		Y_is_zero = np.abs(Y) < 1e-6
		exp_Y_squared = np.exp(-x * y_squared)
		log_exp_Y = np.nan_to_num(np.log(1 - exp_Y_squared))
		exp_ratio = np.nan_to_num(exp_Y_squared / (1 - exp_Y_squared))
		obj = sum(sum(Y_is_zero * (-EX2 * x) + (1 - Y_is_zero) * log_exp_Y))
		grad = sum(sum(Y_is_zero * (-EX2) + (1 - Y_is_zero) * y_squared * exp_ratio))

		if (type(obj) is not np.float64) or (type(grad) is not np.float64):
			raise Exception("Unexpected behavior in optimizing decay coefficient lambda. Please contact emmap1@cs.stanford.edu.")
		if type(obj) is np.float64:
			obj = -np.array([obj])
		if type(grad) is np.float64:
			grad = -np.array([grad])

		return obj, grad

class ZIFA():
    def __init__(self,K=10):
        self.K=K
        pass
    def fit(self,Y,max_iter=100):
        self.Y=Y
        # init
        self.init_em()
        print("Lambda",self.Lambda)
        print("A",self.A.mean())
        print("W",np.diag(self.W).mean())
        print("Mu",self.Mu.mean())
        for i in range(max_iter):
            # self.Lambda=1
            print("Iteration ",i)
            self.e_step()
            print("EZ",self.EZ.mean())
            print("EX",self.EX.mean())
            print("EX2",self.EX2.mean())
            print("EZZT",self.EZZT.mean())
            print("EXZ",self.EXZ.mean())
            self.m_step()
            print("Lambda",self.Lambda)
            print("A",self.A.mean())
            print("W",np.diag(self.W).mean())
            print("Mu",self.Mu.mean())
        pass
    def transform(self):
        # transform the data
        pass

    def e_step(self):
        N,D,K=self.N,self.D,self.K
        EX = np.zeros([N, D])
        EXZ = np.zeros([N, D, K])  # this is a 3D tensor.
        EX2 = np.zeros([N, D])
        EZ = np.zeros([N, K])
        EZZT = np.zeros([N, K, K])

        for i in range(N):
            y=self.Y[i,:]
            # print(y.shape)
            zero_index=np.abs(y)<EPS
            mean,cov=self.get_conditional_dist(y)

            EZ[i,:] = mean[:K][:,0]
            EX[i,:] = mean[K:][:,0]
            EX2[i,:] = EX[i,:] ** 2 + np.diag(cov[K:, K:])
            EZZT[i, :, :] = EZ[i,:].dot(EZ[i,:].transpose()) + cov[:K, :K]
            EXZ[i, :, :] = np.dot(mean[K:], mean[:K].transpose()) + cov[K:, :K]
        self.EX=EX
        self.EZ=EZ
        self.EX2=EX2
        self.EZZT=EZZT
        self.EXZ=EXZ

        pass

    def m_step(self):
        N,D,K=self.N,self.D,self.K
        A=np.zeros([D,K])
        Mu=np.zeros([D,1])
        W=np.zeros([D,D])
        Zero_index = np.abs(self.Y)<EPS

        B=np.eye(K+1)
        for i1 in range(K):
            for i2 in range(K):
                B[i1,i2]=np.sum(self.EZZT[:,i1,i2])# /np.sum(EZZT[:,i1,i1])
        for i2 in range(K):
            B[K,i2]=np.sum(self.EZ[:,i2])# 1/N*
        for i1 in range(K):
            B[i1,K]=np.sum(self.EZ[:,i1])# /np.sum( self.EZZT[:,i1,i1])
        B[K,K]=N


        tiled_Zero_index = np.tile(np.resize(Zero_index, [N, D, 1]), [1, 1, K])
        tiled_EZ = np.tile(np.resize(self.EZ, [N, 1, K]), [1, D, 1])
        tiled_Y = np.tile(np.resize(self.Y, [N, D, 1]), [1, 1, K])

        C = np.zeros([K + 1, D])
        for j in range(D):
            zero_index=Zero_index[:,j]
            temp=np.sum(self.EXZ[zero_index==1,j,:] ,axis=0)+np.sum(np.tile(self.Y[zero_index==0,j].reshape(-1,1) ,[1,K]) * self.EZ[zero_index==0,:] ,axis=0)
            C[:K,j]=temp
            C[K,j]=np.sum(self.EX[zero_index==1,j] )+np.sum(self.Y[zero_index==0,j] )


        # print(B)
        inv_b=np.linalg.inv(B)

        muj = inv_b.dot(C)
        self.A = muj[:K, :].transpose()
        self.Mu = muj[K, :]



        # Then optimize sigma
        EXM = np.zeros([N, D])  # have to figure these out  after updating mu.
        EM = np.zeros([N, D])
        EM2 = np.zeros([N, D])

        tiled_mus = np.tile(self.Mu.transpose(), [N, 1])
        tiled_A = np.tile(np.resize(self.A, [1, D, K]), [N, 1, 1])

        EXM = (tiled_A * self.EXZ).sum(axis=2) + tiled_mus * self.EX
        test_sum = (tiled_A * tiled_EZ).sum(axis=2)
        A_product = np.tile(np.reshape(self.A, [1, D, K]), [K, 1, 1]) * (np.tile(np.reshape(self.A, [1, D, K]), [K, 1, 1]).T)

        for i in range(N):
            EM[i, :] = (np.dot(A, self.EZ[i, :].transpose()) + self.Mu.transpose())  # this should be correct
            EZZT_tiled = np.tile(np.reshape(self.EZZT[i, :, :], [K, 1, K]), [1, D, 1])
            ezzt_sum = (EZZT_tiled * A_product).sum(axis=2).sum(axis=0)
            EM2[i, :] = ezzt_sum + 2 * test_sum[i, :] * tiled_mus[i, :] + tiled_mus[i, :] ** 2

        sigmas = (Zero_index * (self.EX2 - 2 * EXM + EM2) + (1 - Zero_index) * (self.Y ** 2 - 2 * self.Y * EM + EM2)).sum(axis=0)

        decay_coef = minimize(lambda x: decayCoefObjectiveFn(x, self.Y, self.EX2), self.Lambda, jac=True, bounds=[[1e-8, np.inf]])
        decay_coef = decay_coef.x[0]
        self.Lambda=decay_coef

        pass

    def init_em(self):
        #self.A,self.Mu,self.Sigma,self.Lambda
        self.N,self.D=self.Y.shape

        self.Mu=self.Y.mean(axis=0)
        FA = FactorAnalysis(n_components=self.K, random_state=0)
        FA.fit(self.Y-self.Mu)
        self.A=FA.components_.transpose()
        self.W=np.diag(FA.noise_variance_)

        p0=(np.abs(self.Y)<EPS).mean(axis=1)
        y_non=copy.deepcopy(self.Y)
        y_non[np.abs(y_non<EPS)]=np.nan
        mu_nonzero=np.nanmean(y_non,axis=1)
        self.Lambda, _ = curve_fit(doubel_exp, mu_nonzero, p0, p0=.05,)


        pass

    def get_conditional_dist(self,y):
        # print(y.shape)
        zero_index=np.abs(y)<EPS
        N,D,K=self.N,self.D,self.K

        mu_xz=np.zeros([D+K, 1])
        # print(self.Mu.shape)
        mu_xz[K:,0]=self.Mu
        cov_xz=np.zeros([D+K,D+K])
        cov_xz[0:K,0:K]=np.eye(K)
        cov_xz[K:,K:]=self.A.dot(self.A.transpose())+self.W
        cov_xz[K:,:K]=self.A
        cov_xz[:K,K:]=self.A.transpose()

        o_index=np.append(np.ones(K,dtype=int),zero_index,)
        # print(o_index.shape)
        mu_o=mu_xz[o_index==1]
        mu_p=mu_xz[o_index!=1]

        
        cov_oo=index2d(cov_xz,o_index,o_index)
        cov_op=index2d(cov_xz,o_index,1-o_index)
        cov_po=index2d(cov_xz,1-o_index,o_index)
        cov_pp=index2d(cov_xz,1-o_index,1-o_index)

        y_p=np.atleast_2d(y[o_index[K:]!=1]).transpose()


        c_mu=mu_o+cov_op.dot(np.linalg.inv(cov_pp) ).dot(y_p-mu_p)
        c_cov=cov_oo-cov_op.dot(np.linalg.inv(cov_pp)).dot(cov_po)

        Ix=np.diag(np.ones( np.sum(o_index) )) # TODO: check the correctness!!!!

        cov=np.linalg.inv(np.linalg.inv(c_cov)+2*self.Lambda*Ix)
        mean= cov.dot(c_cov).dot(c_mu)


        cov=set_value2d(np.eye(K+D),cov,o_index,o_index)
        new_mean=np.zeros((K+D,1))
        new_mean[o_index==1]= mean


        return new_mean,cov

        pass


if __name__=="__main__":
    Z,Y=gen_data()
    data=pd.read_csv("CORTEX data\gene_expression.txt",sep=" ",header=None)
    data=data.values
    zifa=ZIFA(10)
    zifa.fit(data)