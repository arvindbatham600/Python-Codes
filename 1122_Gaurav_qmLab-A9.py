#name : Gaurav
#rollno : 2020PHY1122

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import eigh
from scipy.special import assoc_laguerre as al


def V(r):
    return -2/r

def V_eff(l, r):
    return l*(l+1)/(r**2) + V(r)

def fin_diff(X,l): 
    #h = (b-a)/(n-1)  #n is number of grid points 
    n = len(X)
    K,V = np.zeros((n,n)),np.zeros((n,n))
    h = X[1]-X[0]
    #X = np.linspace(a,b,n)
    
    v = V_eff(l, X)
    
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

def plots(x,y1,y2,title,color = None): #num defines if there would be only one plot or more
    
    
    for i in range(len(y1)):
        plt.plot(x, y2[i], label='analytical l = '+ str(i),c = color[i][0])
        plt.scatter(x, y1[i],s=5, label='computed l = '+ str(i),c=color[i][1])
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('u**2')
    plt.xlim(0,10)
    plt.title(title)
    plt.legend()
    plt.show()

def analytical_sol(x,n,l):
    anal = np.exp(-x/n)*(2*x/n)**l*al(2*x/n,n-l-1 ,2*l+1)
    norm_anal = normalize(x, anal)[1]
    return norm_anal



#PROGRAMMING 

#part a_i

r = np.linspace(10**(-14), 150, 1000)

for i in range(1, 4):  # l= 1,2,3
    plt.plot(r, V_eff(i, r),label = 'V_eff for l='+str(i))

plt.plot(r, V(r),label = "V(r)",c = 'y')
plt.title("plot of potential vs x")
plt.xlabel("r")
plt.ylabel("v(r)")
plt.grid()
plt.legend()
plt.show()


l0,l1,l2 = 0,1,2

sol = fin_diff(r,l0)
print("for l  =0")
print("THE FIRST 10 EIGEN VALUES COMPUTED USING FINITE DIFFERENCE METHOD FOR L = 0 ARE : ")

anal_e = []

for i in range(1,11):
    anal_e.append(-1*(i)**-2)

print(pd.DataFrame({'COMPUTED e':sol[0][1:11],'ANALYTICAL e':anal_e}))    


for i in range(1,5):
    
    u = sol[1][:, i]

    norm_u = normalize(r, u)[1]  #normalised wave using normalise function
    
    anal = analytical_sol(r, i, l0)
    sign = [1,-1,1,-1]
    plt.scatter(r,norm_u,s=5,label = 'computed',c = 'r')
    plt.plot(r,sign[i-1]*anal,label = 'analytical')
    plt.xlabel('X')
    plt.xlim(0,50)
    plt.ylabel('U')
    plt.title("PLOT FOR U VS X FOR N = "+str(i))
    plt.grid();plt.legend()
    plt.show()
    
print('')  
print("for l = 1")
sol_1 = fin_diff(r[1:],1)

print("THE FIRST 10 EIGEN VALUES COMPUTED USING FINITE DIFFERENCE METHOD FOR L = 1 ARE : ")

anal_e = []

for i in range(2,12):
    anal_e.append(-1*(i)**-2)
print(pd.DataFrame({'COMPUTED e':sol_1[0][:10],'ANALYTICAL e':anal_e}))    
print('') 
print("for l = 2")
sol_2 = fin_diff(r[1:],2)

print("THE FIRST 10 EIGEN VALUES COMPUTED USING FINITE DIFFERENCE METHOD FOR L = 2 ARE : ")

anal_e = []

for i in range(3,13):
    anal_e.append(-1*(i)**-2)
print(pd.DataFrame({'COMPUTED e':sol_2[0][0:10],'ANALYTICAL e':anal_e}))    


#("plot for probability density")
#part_i  for n=l=0
u = sol[1][:,1] #n+1 th eigen vectors 
u_norm = normalize(r, u)[1]
anal = analytical_sol(r,1,l0)

plots(r, [u_norm**2], [anal**2], title = "probability density plot for n=0",color = [['b','g']])


#part_ii  for n=1 l = 0,1
u_1 = sol[1][:,2][1:]
u_2 = sol_1[1][:,2]
u_norm_1 = normalize(r[1:], u_1)[1]
u_norm_2 = normalize(r[1:], u_2)[1]
anal_1 = analytical_sol(r[1:],2,l0)
anal_2 = analytical_sol(r[1:],2,l1)

plots(r[1:], [u_norm_1**2,u_norm_2**2], [anal_1**2,anal_2**2], title = "probability density plot for n=1",color = [['b','g'],['r','y']])


#part_iii  for n=2 l = 0,1,2
u_1 = sol[1][:,3][1:]
u_2 = sol_1[1][:,3]
u_3 = sol_2[1][:,3]
u_norm_1 = normalize(r[1:], u_1)[1]
u_norm_2 = normalize(r[1:], u_2)[1]
u_norm_3 = normalize(r[1:],u_3)[1]
anal_1 = analytical_sol(r[1:],3,l0)
anal_2 = analytical_sol(r[1:],3,l1)
anal_3 = analytical_sol(r[1:],3,l2)
plots(r[1:], [u_norm_1**2,u_norm_2**2,u_norm_3**2], [anal_1**2,anal_2**2,anal_3**2], title = "probability density plot for n=2",color = [['b','g'],['r','y'],['m','orange']])
