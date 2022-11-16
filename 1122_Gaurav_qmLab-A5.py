
#NAME : GAURAV CHANDRA
#ROLL NO : 2020PHY1122

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import hbar,elementary_charge
from scipy import special


def f(e,z):
    return 2*(e - (1/2)*(z**2))


def Numerov(a,b,u_0,u_dash,N,n,f,del_e = 10**(-6)): #numerov method
    
    e = n+0.5+del_e   #eigen value
    h = (b-a)/N     #step size
    x = np.arange(a,b+h,h)
    
    u = np.zeros([N+1])

    C = 1 + ((h**2)/12)*f(e,x)
    
    u[0] = u_0
    
    if n%2!=0: #odd states
    
        u[1] =  u[0] + u_dash*h
    else :      #even states
        
        u[1] = (6-5*C[0])*(u[0]/C[1])
    
    for i in range(2,N+1):
        u[i] = ((12 - 10*C[i-1])*u[i-1] - C[i-2]*u[i-2])/C[i]
        
    
    x_minus = -x[1:]

    X = np.append(x_minus[-1::-1], x)

    if n % 2 != 0:  # odd states
        
        u_minus = -u[1:]
        U = np.append(u_minus[-1::-1], u)

    else:  # even states
        u_minus = u[1:]
        U = np.append(u[-1::-1], u_minus)

    return [X,U]

def anal(x,n):
    p = special.hermite(n,monic = True)
    u = np.exp(-0.5*x**2)*p(x)
    U = normalize(x, u, MySimp)[1]
    return U

def normalize(wavefx,wavefy,int_method):  #this function returns list including normalisation constant and 
                                          #normalised eigen function
    I = int_method(wavefx,wavefy**2)
    A = (I)**(-1/2)
    
    return [A,A*wavefy]

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


if __name__ == "__main__":
    
    ##############  PART A ################
    
    n = 0  #GROUND STATE
    u_0 = 1
    u_dash = 1
    x = 0
    x_max = 4
    del_e = [10**(-2),10**(-4),10**(-6),10**(-8)]
    c = ['red','yellow','blue','violet']
    
    for i in range(len(del_e)) :      #plot for different del E
    
        sol = Numerov(x, x_max, u_0, u_dash, 100, n, f,del_e[i])
        X = sol[0]
        u = sol[1]
    
        U = normalize(X, u, MySimp)[1]
        
        plt.scatter(X,U,c = c[i],s=7,label = 'e = '+str(0.5+del_e[i]))
        plt.xlabel("X")
        plt.ylabel("U(X)")
        if i == 2:       #AT INDEX 2 ,DEL E IS 10**-6 ,FOR PART B FOR PROB DENSITY
            u_array = U
        plt.title("PLOT OF U(X) VS X FOR N = 0 ")
        
    x_array = X  #FOR ANALYTICAL PLOT
    plt.plot(x_array,anal(x_array, n),linewidth = 3,c = 'pink',label = 'analytical')
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.scatter(x_array,u_array**2,c = 'red',label = 'e = '+str(0.5+10**(-6)))
    plt.xlabel("X")
    plt.ylabel("U(X)")
    plt.title("PLOT OF P(X) VS X FOR N = 0")
    plt.plot(x_array,anal(X, 0)**2,linewidth = 2.5,c = 'yellow',label = 'analytical')
    plt.grid()
    plt.legend()
    plt.show()
    
    
    ################  PART B  ##########
     
    #FOR DIFFERENT N
    n = [1,2,3]
    u_0 = [0,1,0]
    u_dash = [1,1,-1]
    
    for i in range(len(n)):
        sol = Numerov(x, x_max, u_0[i], u_dash[i], 100, n[i], f)
        X = sol[0]
        u = sol[1]
        U = normalize(X, u, MySimp)[1]
        
        plt.scatter(X,U,c='red',label = 'e = '+str(n[i]+0.5+10**(-6)))
        plt.xlabel("X")
        plt.ylabel("U(X)")
        plt.title("PLOT OF U(X) VS X FOR N =  "+str(n[i]))
        plt.plot(X,anal(X, n[i]),linewidth = 2.5,c = 'yellow',label = 'analytical')
        plt.grid()
        plt.legend()
        plt.show()
        
        plt.scatter(X,U**2,c='red',label = 'e = '+str(n[i]+0.5+10**(-6)))
        plt.xlabel("x")
        plt.ylabel("P(x)")
        plt.title("PLOT OF P(X) VS X FOR N =  "+str(n[i]))
        plt.plot(X,anal(X, n[i])**2,linewidth = 2.5,c = 'yellow',label = 'analytical')
        plt.grid()
        plt.legend()
        plt.show()
        
        #########################  PART C  ####################
        
    w = 5.5*10**(14)   #GIVEN FREQUENCY
    e_eigen = [ ]
    e_analytic = [ ]
    for i in range(4):
        e_eigen.append((i+0.5+10**(-6))*hbar*w/elementary_charge)
        e_analytic.append((i+0.5)*hbar*w/elementary_charge)
        
    print("TABLE FOR ENERGY EIGEN VALUES ")
    print(pd.DataFrame({'analtyic energy values':e_analytic,'calculated energy values':e_eigen}))
        
        #########################  PART D  ####################
        
    
    x_forbidden = np.array([])
    u_forbidden = np.array([])
    for i in range(len(x_array)):
        if x_array[i] >1 or x_array[i] < -1 :
            x_forbidden = np.append(x_forbidden,x_array[i])
            u_forbidden = np.append(u_forbidden,u_array[i])
        else : 
                
            continue
        
    print('')
    print("THE PROBABILITY OF FINDING AN ELECTRON IN THE CLASSICALLY FORBIDDEN ENERGY REGION IS ",MySimp(x_forbidden, u_forbidden **2)*100,'%')
        