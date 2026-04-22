import numpy
import autograd

def gc(L,n):
    ff_=autograd.grad(L,0)
    def ff(a,b):
        return ff_(a,b,autograd.numpy)

    mf_=autograd.hessian(L, 1)
    def mf(a,b):
        return mf_(a,b,autograd.numpy)

    cf_=autograd.jacobian(autograd.grad(L,1),0)
    def cf(a,b):
        return cf_(a,b,autograd.numpy)
        
    def cal(q,qdot,qddot):
        F = ff(q,qdot)

        M = mf(q,qdot)
        C = cf(q,qdot)
        qddot[:]=numpy.linalg.solve(M, F - C.dot(qdot))
    return cal 