import numpy as np
from sklearn.linear_model import LogisticRegression

def my_map(X):
    mapped_features = []
    for challenge in X:
        n = len(challenge)
        d = np.ones(n)
        for i in range(n):
            product = 1
            for j in range(i, n):
                product *= (1 - 2 * challenge[j])
            d[i] = product
        
        feature_vector = [1] + d.tolist()
        for k in range(len(d) - 1):
            feature_vector.append(d[k] * d[k + 1])
        
        mapped_features.append(feature_vector)
    
    return np.array(mapped_features)

def my_fit(X_train, y0_train, y1_train):
    # Map the challenges to higher-dimensional features
    X_transformed = my_map(X_train)

    # Train models for Response0 and Response1
    model0 = LogisticRegression(fit_intercept=True, max_iter=3000, C=120, random_state=0)
    model1 = LogisticRegression(fit_intercept=True, max_iter=2500, C=120, random_state=0)
    model0.fit(X_transformed, y0_train)
    model1.fit(X_transformed, y1_train)

    W0 = model0.coef_.flatten()
    b0 = model0.intercept_[0]
    W1 = model1.coef_.flatten()
    b1 = model1.intercept_[0]

    return W0, b0, W1, b1
