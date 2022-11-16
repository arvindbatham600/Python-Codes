#name : Gaurav
#rollno : 2020PHY1122

import numpy as np
import matplotlib.pyplot as plt

# POTENTIAL FUNCTION

def f(e,z):
    return 2*(e - (1/2)*(z**2))

# NUMEROV

def Numerov(a,b,n,N,f): #numerov method
    
    h = (b-a)/N     #step size
    x = np.arange(a,b+h,h)
    e = n+0.5
    u = np.zeros([N+1])

    C = 1 + ((h**2)/12)*f(e,x)
    
    if n%2!=0: #odd states
        u[0] = 0
        u[1]= h
    else :      #even states
        u[0] = 1
        u[1] = (6-5*C[0])*(u[0]/C[1])
    
    for i in range(2,N+1):
        u[i] = ((12 - 10*C[i-1])*u[i-1] - C[i-2]*u[i-2])/C[i]
    
    return x,u

#PARITY
def parity(x,u,n):
    x_minus = -x[1:]

    X = np.append(x_minus[-1::-1], x)

    if n % 2 != 0:  # odd states
        
        u_minus = -u[1:]
        U = np.append(u_minus[-1::-1], u)

    else:  # even states
        u_minus = u[1:]
        U = np.append(u[-1::-1], u_minus)
    
    return X,U

# MULTIPLYING BY FACTOR

def factor(a,b,n,N,f):
    
    if n%2==0:
        x,u=Numerov(a,b,n,N,f)
    else:
        x, u = Numerov(a, b, n, N, f)
    
    xcl1 = round(np.sqrt(2 * n + 1), 2)
    p = 0
    for i in x:
        if (round(i, 2)) == xcl1:
            break
        else:
            p += 1
    c=x[p-1]
    
    h = (b - a) / N
    N2= int((c+h - a)/h)
    
    if n%2==0:
        
        x1,u1=Numerov(a,c+h,n,N2,f)
    else:
        
        x1, u1 = Numerov(a, c+h, n, N2, f)
    N3 = int((b - (c-h)) / h)
    x2,u2=Numerov(b,c-h,n,N3,f)

    h=x2[1]-x2[0]
    k= u1[-1]/u2[-1]
    u2_new= k*u2
    u2=u2_new

    return u2[-3], u2[-2], u1[-3], c,u1[-2], u2[-1], u1[-1], h

print(factor(0,6,2,500,f))

def derivative(factor):
    num= factor[2] + factor[0] - (12*factor[3] - 10)*factor[1]
    d_der= num/factor[4]
    return d_der


def bisection(a,b,nmax,N,f,derivative,tol,factor):
    nmin=0
    d_der=derivative(factor(a,b,nmax,N,f))
    while abs(d_der)>=tol:
        if d_der<=0:
            nmin=(nmin+nmax)/2
        else:
            nmax=(nmin+nmax)/2
        n_new= (nmin+nmax)/2
        
        d_der = derivative(factor(a, b, n_new, N, f))
        
    return (nmax+nmin/2 + 0.5), d_der


#2ND APPROACH
def alternate(factor):
    num1=factor[6]-factor[2] - factor[0] + factor[5]
    
    return num1


# EXAMPLE:

x,u=Numerov(6,0,5, 100, f)  
X,U = parity(x, u, 5)
plt.plot(X,U)
plt.grid()
plt.xlabel('X')
plt.ylabel('U(X)')                         
plt.show()

n = [0,1,2,3,4,5]
e_cal = []
for i in n:
    
    e_cal.append(bisection(0,8,n[i],10000,f,alternate,10**(-5),factor)[0])

print('the first 6 energy values are : ',e_cal)