from numpy import *
from numpy import random
from numpy.linalg import multi_dot, inv

def vectorf(VectorFunc, X):
    if len(VectorFunc)!=len(X):
        print("Vector function and its argument do not have same length")
    n = len(X)
    result = empty(n)
    for i in range(n):
        f = VectorFunc[i]
        result[i] = f(X[i])
    return result

def simulate_rate_model(pars, quench_which = [], quench_values = []):
    N   = pars['N']
    t   = pars['t']
    R0  = pars['R0']
    J   = pars['J']
    I   = pars['I']
    sigma = pars['sigma']
    f   = pars['f']
    nsteps = len(t)
    dt = t[1] - t[0]

    if len(quench_values)!=len(quench_which):
        print('Wrong sizes of quench params')

    R = empty((N,nsteps))
    R[:,0] = R0
    for i in range(nsteps-1):

        if len(quench_which): # quench neurons with indices quench_which to values quench_values
            R[quench_which, i] = quench_values

        if hasattr(f,"__len__"):
            dR = - R[:,i] + vectorf( f, J @ R[:,i] + I[:,i] + sigma * random.normal(0,1,N)/sqrt(dt) )
        else:
            dR = - R[:, i] + f(J @ R[:, i] + I[:, i] + sigma * random.normal(0, 1, N) / sqrt(dt))
        R[:,i+1] = R[:,i] + dt * dR

        if amax(R[:i+1])>2e2:
            print('Dynamics may be unstable! Check.')

    if len(quench_which):
        R[quench_which, nsteps-1] = quench_values

    return R






##

