#NAME : GAURAV CHANDRA
#ROLL NO : 2020PHY1122

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

def f(x):               # u''(x) = -f(x)*u(x)
    return -1*(1+x**2)

def initial_cond(a,b,N):      # function to generate u(0+h) from u'(0)
    h = (b-a)/N
    u_1 = 1 + (h**2)/2 + (1/8)*h**4
    return u_1


def numerov(a,b,u_0,u_1,N,f): #numerov method
    
    h = (b-a)/N
    x = np.arange(a,b+h,h)
    u_arr = np.array([u_0,u_1])  #this array will store the value of u(x)
    
    for i in range(2,N+1):
        cons1 = 2*(1-(5/12)*(h**2)*f(a+(i-1)*h))*u_arr[i-1]
        cons2 = (1+((h**2)/12)*f(a+(i-2)*h))*u_arr[i-2]
        cons3 = 1+((h**2) /12)*f(a+i*h)
        
        u_arr = np.append(u_arr,(cons1-cons2)/cons3)
        
    return [x,u_arr]

def function(y,x):     # this function is used in inbuilt function
    
    return (y[1],(1+x**2)*y[0])
    
if __name__ == "__main__":
    a = 0
    b = 1
    #for N = 2
    u_0 = 1
    
    N = 2
    u_1 = initial_cond(a, b, N)
    
    solution_1 = numerov(a, b, u_0, u_1, N, f)
    inbuilt_1 = odeint(function,(1,0),solution_1[0])[:,0]
    
    #for N = 4
    N = 4
    u_1 = initial_cond(a, b, N)
    
    solution_2 = numerov(a, b, u_0, u_1, N, f)
    inbuilt_2 = odeint(function,(1,0),solution_2[0])[:,0]
    
    
    #part c
    
    data_1 = {"x_i":solution_1[0],'u_num':solution_1[1],'u_inbuilt':inbuilt_1,'E_i':abs(inbuilt_1-solution_1[1])}
    print("--:DATA TABLE FOR N = 2 :--")
    print(pd.DataFrame(data_1))
    print('')
    
    data_2 = {"x_i":solution_2[0],'u_num':solution_2[1],'u_inbuilt':inbuilt_2,'E_i':abs(inbuilt_2-solution_2[1])}
    print("--:DATA TABLE FOR N = 4 :--")
    print(pd.DataFrame(data_2))
    
    
    #part d
    k = np.arange(1,7)
    N_list = 2**k
    color = ['red','orange','blue','yellow','violet','green']
    for i in range(len(N_list)):
        u = initial_cond(a, b, N_list[i])
        solution = numerov(a, b, u_0, u, N_list[i], f)
        inbuilt = odeint(function,(1,0),solution[0])[:,0]
        plt.plot(solution[0],solution[1],label='computed ,N='+str(N_list[i]),c=color[i])
        plt.plot(solution[0],inbuilt,label='inbuilt ,N='+str(N_list[i]),ls = '--',c=color[i])
        
    plt.xlabel('X')
    plt.ylabel('U(X)')
    plt.grid()
    plt.legend()
    plt.title("PLOTS OF U VS X FOR DIFFERENT N")
    plt.savefig("ass4.png")
    plt.show()