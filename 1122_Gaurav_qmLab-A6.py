#name : gaurav chandra
#rollno : 2020PHY1122

import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

def Numerov(a,b,N,n,f,e): #numerov method
    
    h = (b-a)/N     #step size
    x = np.arange(a,b+h,h)
    
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
        
    return [x,u]

def MySimp(x,y):  #x here is the array of independent variable  and y for dependent variable
    # calculating step size
    h = abs((x[-1] - x[0]) / len(x))
    
    simpint = y[0] + y[-1]
    
    for i in range(1,len(x)):
        
        if i%2 == 0:
            simpint = simpint + 2 * y[i]
        else:
            simpint = simpint + 4 * y[i]          
    
    # multiply h/2 with the obtained integration to get Simpson integration 
    simpint =simpint * h/3
    
    return simpint

def f(e,z):
    return 2*(e - (1/2)*(z**2))

def bisec(e_min, e_max, n, n_nodes, e_tol):  # bisection
    e = (e_min + e_max)/2
    h_nodes = int(n_nodes/2)

    while abs(e_min - e_max) >= e_tol:
        
        
        u = Numerov(0, 4, 100, n,f,e)[1]

        n_cross = 0

        for i in range(1, len(u)):
            if u[i-1]*u[i] < 0:
                n_cross += 1

        if n_cross > h_nodes:
            e_max = e
            
        else:
            e_min = e
            
        e = (e_min + e_max)/2
        i+=1
    return e


# 2(a)i.
n = np.array([0, 1, 2, 3, 4, 5])

e_cal = []
for i in n:
    # e_min = 0, e_max = V(x_max) = 8 if x_max = 4
    e_cal.append(bisec(0, 8, i, i, 0.5*(10**(-10))))

e_cal = np.array(e_cal)

# 2(a)ii.

e_ana = n + 0.5

table = pd.DataFrame({'e_calculated': e_cal, 'e_analytical': e_ana})

print('---first six energy eigen values---')
print(table)

# 2(a)iii.

plt.plot(n, e_cal)
plt.xlabel('n')
plt.ylabel(r'$e_n$')
plt.grid()
plt.title(r'$e_n$ as a function of n')
plt.show()

slope_1 = linregress(n, e_cal)[0]
intercept_1 = linregress(n, e_cal)[1]

# 2(b)

slope_2 = linregress(n**2, e_cal)[0]
intercept_2 = linregress(n**2, e_cal)[1]

fitted_n = slope_2*(n**2) + intercept_2

plt.scatter(n**2, e_cal, label='data points',c='r')
plt.plot(n**2, fitted_n, label='fitted curve')
plt.legend()
plt.grid()
plt.title(r'$e_n$ as a function of $n^2$')
plt.show()

# 2(c)i.

def wfn_plot(n, x_max, e, p=None):

    e = np.array([e])
    
    for i in range(len(e)):

        x = Numerov(0, x_max, 500, n,f,e)[0]
        u = Numerov(0, x_max, 500, n,f,e)[1]

        u = u/np.sqrt(MySimp(x, u**2))  # normalizing the wave fnc

        def parity(x, u):
            x_ = -x[1:]
            X = np.append(x_[-1::-1], x)

            if n % 2 != 0:  # odd states
                u_ = -u[1:]
                U = np.append(u_[-1::-1], u)

            else:  # even states
                u_ = u[1:]
                U = np.append(u[-1::-1], u_)

            return X, U

        if p != None:
            wfn = (parity(x, u)[1])**p
        else:
            wfn = (parity(x, u)[1])

        x_pts = parity(x, u)[0]

        plt.plot(x_pts, wfn, label=f'e = {np.round(e[i],3)}')
        plt.title(f'for x_max = {x_max}')

    plt.legend()
    plt.grid()
    plt.xlabel('x')
    if p == None:
        plt.ylabel('Ψ(x)')
    else:
        plt.ylabel(r'$Ψ^{p}(x)$'.format(p=p))

    plt.show()


for i in range(len(n)-1):
    wfn_plot(n[i], 10, e_cal[i])

for i in range(len(n)-1):
    wfn_plot(n[i], 10, e_cal[i], 2)

# 2(c)iii.
for i in range(len(n)-1):
    wfn_plot(n[i], 5, e_cal[i])

for i in range(len(n)-1):
    wfn_plot(n[i], 5, e_cal[i], 2)
