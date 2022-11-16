
#name : Gaurav
#rollno : 2020PHY1122

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import eigh
from scipy.special import assoc_laguerre as al


def V_net(a,r):
    return (r**2) + 2*a*(r)**3

def fin_diff(X,A): 
    #h = (b-a)/(n-1)  #n is number of grid points 
    n = len(X)
    K,V = np.zeros((n,n)),np.zeros((n,n))
    h = X[1]-X[0]
    #X = np.linspace(a,b,n)
    
    v = V_net(A, X)
    
    K[0,0] = -2;K[0,1] = 1  
    K[n-1,n-1] = -2;K[n-1,n-2] = 1
    
    for i in range(n):
        V[i,i] = v[i]        
    for i in range(1,n-1):
        K[i,i]=-2
        K[i,i-1]=1
        K[i,i+1]=1
        
    H = (-1*K)/(h**2) + V
    
    U = eigh(H)[1]
    e = eigh(H)[0]
    
    return [e,U]

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

def normalize(wavefx,wavefy,int_method = MySimp):  #this function returns list including normalisation constant and 
                                          #normalised eigen function
    I = int_method(wavefx,wavefy**2)
    A = (I)**(-1/2)
    
    return [A,A*wavefy]

def plots(x,y,title,ylab,color = None): #num defines if there would be only one plot or more
    
    for i in range(len(y)):
        plt.plot(x, y[i], label='for n = '+ str(i))
        
    plt.grid()
    plt.xlabel('x')
    plt.ylabel(ylab)
    #plt.xlim(0,10)
    plt.title(title)
    plt.legend()
    plt.show()

def pert_e(n,alpha):
    e = (2*n+1) - (1/8)*(alpha**2)*(15*(2*n+1)**2 + 7)
    
    return e

#PROGRAMMING 

r = np.linspace(-5, 5, 1000)

alpha = [0]
for i in range(5):
    alpha.append(10**(-i))

for i in range(len(alpha)):  # l= 1,2,3
    plt.plot(r, V_net(alpha[i], r),label = 'for alpha = '+str(alpha[i]))

plt.title("plot of potential vs x")
plt.xlabel("x")
plt.ylabel("v(x)")
plt.grid()
plt.legend()
plt.show()

# part a _i

E_cal , E_anal = np.array([]) , []

for i in alpha:
    sol = fin_diff(r,i)[0][0:10]
    E_cal = np.append(E_cal,sol)
    
    for n in range(10):
        E_anal.append(pert_e(n, i ))

E_cal = E_cal.reshape(len(alpha),10)
E_anal =np.array(E_anal).reshape(len(alpha),10)  
#print(E_cal)

#part a_ii

print("THE FIRST 10 EIGEN VALUES COMPUTED USING FINITE DIFFERENCE METHOD ARE : ")
for i in range(len(alpha)):
    print(pd.DataFrame({'E_calculated(alpha ='+str(alpha[i])+' )':E_cal[i],'E_analytical':E_anal[i]}))    

    
#for a harmonic oscillator
n = np.arange(0,10)
E_h = []
for i in n:
    E_h.append(2*i+1)   #bcuz for our case E0 = 0.5*h_bar*w , so e = E/E0 = 2*(n+0.5)

for i in range(len(alpha)):
    plt.plot(n,E_cal[i],label = 'for alpha = '+str(alpha[i]))
plt.plot(n,E_h,'--',label = 'harmonic oscillator')
plt.legend()
plt.grid()
plt.xlabel('n')
plt.ylabel('E_n')
plt.show()    



#part c_i

for i in [0,0.1,0.01]:
    for j in range(5):
        u_1 = fin_diff(r, i)[1][:,j]
        norm_u = normalize(r, u_1)[1]  #normalised wave using normalise function
        plt.plot(r,norm_u,label = 'n = '+str(j))
    plt.grid()
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("plot of y vs x for alpha = "+str(i))
    plt.show()

for i in range(2):
    for j in range(6):
        u_1 = fin_diff(r, alpha[j])[1][:,i]
        norm_u = normalize(r, u_1)[1]
        plt.plot(r,norm_u**2,label = 'alpha = '+str(10**(-j)))   
    plt.grid()
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y**2')
    plt.title("probability density plot for state n = "+str(i))
    plt.show()

