import matplotlib.pyplot
import time


def check(a,b):
    def l(q,qdot,usermath):
        t=0.0
        t=t+qdot[0]*qdot[0]
        t=t+qdot[1]*qdot[1]+qdot[0]*qdot[0]+qdot[0]*qdot[1]*2.0*usermath.cos(q[0]-q[1])
        v=-2*usermath.cos(q[0])-usermath.cos(q[1])
        return t-v
    n=2
    cal=gc(l,n)


    q=numpy.zeros(n,dtype=numpy.float64)
    qdot=numpy.zeros(n,dtype=numpy.float64)
    k1=numpy.zeros(n,dtype=numpy.float64)

    q[0]=a
    qdot[0]=b

    dt=0.02
    recn=100

    r=numpy.zeros((recn,n),dtype=numpy.float64)


    for i in range(recn):
        for j in range(10):
            cal(q,qdot,k1)
            
            qdot_=qdot+0.5*dt*k1
            q_=q+0.5*dt*qdot_
            cal(q_,qdot_,k1)
            
            qdot_=qdot+0.5*dt*k1
            q_=q+0.5*dt*qdot_
            cal(q_,qdot_,k1)

            qdot_=qdot+dt*k1
            q=q+0.5*dt*(qdot_+qdot)
            qdot=qdot_
        r[i]=q
    return r

yl=check(0,0.7)
matplotlib.pyplot.plot(yl)