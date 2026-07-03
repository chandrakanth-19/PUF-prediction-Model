import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


def my_fit( X_train, y_train ):

	train_features = my_map(X_train)
	model = LogisticRegression(solver='liblinear',fit_intercept=True,tol=0.1)  
	model.fit(train_features, y_train)

	return model.coef_, model.intercept_



def my_map( X ):

	D = 1-2*X 
	patterns = [
        [0], [1], [2], [3], [4], [5], [6], [7],  
        [0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [0,7],
        [1,2], [1,3], [1,4], [1,5], [1,6], [1,7],
        [2,3], [2,4], [2,5], [2,6], [2,7],
        [3,4], [3,5], [3,6], [3,7],
        [4,5], [4,6], [4,7],
        [5,6], [5,7],
        [6,7],
        [0,1,2], [0,6,7], [1,2,3], [1,6,7], [2,3,4], [2,6,7], [3,4,5], [3,6,7],
        [4,5,6], [4,5,7], [4,6,7], [5,6,7],
        [0,1,2,3], [0,5,6,7], [1,2,3,4], [1,5,6,7], [2,3,4,5], [2,5,6,7],
        [3,4,5,6], [3,4,5,7], [3,4,6,7], [3,5,6,7], [4,5,6,7],
        [0,1,2,3,4], [0,4,5,6,7], [1,2,3,4,5], [1,4,5,6,7],
        [2,3,4,5,6], [2,3,4,5,7], [2,3,4,6,7], [2,3,5,6,7], [2,4,5,6,7], [3,4,5,6,7],
        [0,1,2,3,4,5], [0,3,4,5,6,7], [1,2,3,4,5,6], [1,2,3,4,5,7], [1,2,3,4,6,7],
        [1,2,3,5,6,7], [1,2,4,5,6,7], [1,3,4,5,6,7], [2,3,4,5,6,7],
        [0,1,2,3,4,5,6], [0,1,2,3,4,5,7], [0,1,2,3,4,6,7], [0,1,2,3,5,6,7],
        [0,1,2,4,5,6,7], [0,1,3,4,5,6,7], [0,2,3,4,5,6,7], [1,2,3,4,5,6,7],
        [0,1,2,3,4,5,6,7]
    ]

	features = []
	for pattern in patterns:
		product = np.prod(D[:, pattern], axis=1)  
		features.append(product)

	return np.stack(features, axis=1)  



def my_decode( w ):
	
	k = 64
	A = np.zeros((k+1, 4*k))
	b = np.zeros(k+1)    

	i = np.arange(1, k) 
	A[0, 0:4] = [0.5, -0.5, 0.5, -0.5]
	b[0] = w[0]

	A[i, 4*i] += 0.5  
	A[i, 4*i+1] -= 0.5  
	A[i, 4*i+2] += 0.5   
	A[i, 4*i+3] -= 0.5  

	A[i, 4*(i-1)] += 0.5  
	A[i, 4*(i-1)+1] -= 0.5  
	A[i, 4*(i-1)+2] -= 0.5  
	A[i, 4*(i-1)+3] += 0.5   

	b[1:k] = w[1:k]

	A[k, 4*(k-1):4*k] = [0.5, -0.5, -0.5, 0.5]
	b[k] = w[k]

	model = LinearRegression(positive=True, fit_intercept=False)
	model.fit(A, b)
	x = model.coef_
	x_reshaped = x.reshape(-1, 4)  
	p = x_reshaped[:, 0]
	q = x_reshaped[:, 1]
	r = x_reshaped[:, 2]
	s = x_reshaped[:, 3]

	return p, q, r, s
