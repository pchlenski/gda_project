from itertools import chain, combinations
from sklearn.linear_model import LinearRegression

def powerset(iterable):
    """ Taken from powertools recipe on itertools page:
    https://docs.python.org/3/library/itertools.html#itertools-recipes """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def linear_independence(x, y, z, cutoff=0.4):
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
    if score < cutoff:
        return True
    else:
        return False

def independent(x, y, z):
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
    return abs((x*y).mean() - x.mean()*y.mean())*100 < 1

def test_independences(data, test=linear_independence):
    """ Check each linear (conditional) independence """
    n_points, n_dims = data.shape

    # Check if (A ind B | Z)
    for i in range(n_dims):
        for j in range(i):
            other_dims = list(range(n_dims))
            other_dims.remove(i)
            other_dims.remove(j)
            for k in powerset(other_dims):
                ind_result = test(data[:,i],data[:,j],data[:,k])
                if ind_result:
                    print(f"{i} is independent of {j} given {k}")