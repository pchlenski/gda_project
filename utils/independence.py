from itertools import chain, combinations
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from fcit import fcit
import numpy as np

def powerset(iterable):
    """ Taken from powertools recipe on itertools page:
    https://docs.python.org/3/library/itertools.html#itertools-recipes """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def linear_independence(x, y, z, cutoff=0.01, return_statistic=False):
    """ Check if (X ind Y | Z) by regressing Y on Z, then regressing the
    residuals on X """

    # Get shape parameters
    length = y.shape[0]
    z = z.reshape(length, -1)
    y = y.reshape(length, -1)
    x = x.reshape(length, -1)

    # Fit Y ~ Z
    if z.size > 0:
        reg1 = LinearRegression()
        reg1.fit(z, y)
        y_from_z = reg1.predict(z)
        residuals = y - y_from_z
    else:
        y_from_z = y

    # Fit R(Y|Z) ~ X
    reg2 = LinearRegression()
    reg2.fit(x, y_from_z)
    
    # Test goodness-of-fit
    score = reg2.score(x, y_from_z)
    if return_statistic:
        return score
    elif score < cutoff:
        return True
    else:
        return False

def independent(x, y, z, threshold = .005, return_statistic=False):
    """ Cribbed from CI 1 homework 3 """

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    if z.size > 0:
        # z = np.stack(z).T
        regy = LinearRegression().fit(z, y)
        y = y - regy.predict(z)
        regx = LinearRegression().fit(z, x)
        x = x - regx.predict(z)
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()

    score = abs((x*y).mean() - x.mean()*y.mean())
    if return_statistic:
        return score
    else:
        return score < threshold

def boolean_independence(x, y, z, threshold=0.01, return_statistic=False):
    """ Check independence of boolean variables """

    n_samples, n_conditions = z.shape

    # Type conversion
    x = x.astype(float)

    # Conditioning case
    if z.size > 0:
        # Filter down to Z
        for z_vals in np.unique(z, axis=0):
            idx = [(row == z_vals).all() for row in z]
            x_given_z = x[idx]
            y_given_z = y[idx]
            n_samples_given_z = np.sum(idx)

            # Compute P(Y, X | Z)
            for x_val in [True, False]:
                marginal_x_given_z = (x_given_z == x_val).sum() / n_samples_given_z
                for y_val in [True, False]:
                    # TODO: I think we can omit one of the innermost truth values, since there are 2^n-1 degrees of freedom
                    joint_xy_given_z = ((x_given_z == x_val) * (y_given_z == y_val)).sum() / n_samples_given_z
                    marginal_y_given_z = (y_given_z == y_val).sum() / n_samples_given_z

                    # Test P(Y,X | Z) = P(Y | Z) * P(X | Z)
                    score = np.abs(joint_xy_given_z - marginal_y_given_z * marginal_x_given_z)

    # No conditioning case
    else:
        for x_val in [True, False]:
            marginal_x = (x == x_val).sum() / n_samples
            for y_val in [True, False]:
                joint_xy = ((x == x_val) * (y == y_val)).sum() / n_samples
                marginal_y = (y == y_val).sum() / n_samples

                score = np.abs(joint_xy - marginal_x * marginal_y)

    # Return score / boolean
    if return_statistic:
        return score
    elif score > threshold:
        return False
    else:
        return True

    # If we pass all the tests, then we're good
    return True

def fcit_independence(x, y, z, threshold=0.01, return_statistic=False):
    """ Use FCIT to test conditional independence """
    
    # Reshape datasets
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Run FCIT
    if z.size > 0:
        score = fcit.test(x,y,z)
    else:
        score = fcit.test(x,y)

    # Outputs
    if return_statistic:
        return score
    elif return_statistic > threshold:
        return False
    else:
        return True


def test_independences(data, test=linear_independence, return_statistics=False):
    """ Check each linear (conditional) independence """

    n_points, n_dims = data.shape

    # Check if (A ind B | Z)
    out = []
    for i in tqdm(range(n_dims)):
        for j in range(i):
            other_dims = list(range(n_dims))
            other_dims.remove(i)
            other_dims.remove(j)
            for k in powerset(other_dims):
                ind_result = test(data[:,i], data[:,j] ,data[:,k], return_statistic=return_statistics)
                if return_statistics:
                    out.append([i, j, k, ind_result])
                elif ind_result:
                    print(f"{i} is independent of {j} given {k}")
                    out.append([i, j, k])
    return out
