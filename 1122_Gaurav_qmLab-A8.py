#name : Gaurav
#rollno : 2020PHY1122

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import eigh

def fin_diff(a,b,n):
    h = (b-a)/(n-1)  #n is number of grid points 
    
    K,V = np.zeros((n,n)),np.zeros((n,n))
    
    X = np.linspace(a,b,n)
    
    K[0,0] = -2;K[0,1] = 1  
    K[n-1,n-1] = -2;K[n-1,n-2] = 1
    
    for i in range(1,n-1):
        K[i,i]=-2
        K[i,i-1]=1
        K[i,i+1]=1
        
    H = (-1*K)/(h**2) + V
    
    U = eigh(H)[1]
    e = eigh(H)[0]
    
    return [e,U,X]

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

def plots(x,y1,y2,ylabel,title,num,color = None): #num defines if there would be only one plot or more
    
    if num == 1:
        plt.plot(x, y2,linewidth = 2.5, label='analytical',c = 'b',ls = 'dashed')
        plt.scatter(x, y1,s = 20 ,label='computed',c = 'r')
    else : 
        for i in range(len(y1)):
            plt.plot(x, y2[i], label='analytical n = '+ str(i),c = color[i][0],ls = 'dashed')
            plt.scatter(x, y1[i],s=5, label='computed n = '+ str(i),c=color[i][1])
    plt.grid()
    plt.xlabel('x')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

#PROGRAMMING 

sol = fin_diff(-0.5, 0.5, 1000)

print("THE FIRST 10 EIGEN VALUES COMPUTED USING FINITE DIFFERENCE METHOD ARE : ")

anal_e = []

for i in range(1,11):
    anal_e.append((i*np.pi)**2)

print(pd.DataFrame({'COMPUTED e':sol[0][:10],'ANALYTICAL e':anal_e}))    

U_sq_list , Anal_U_sq_list = [],[]   #these lists will carry the square of values for density plot

for i in range(4):
    
    u = sol[1][:, i]
    x  =sol[2]
    norm_u = normalize(x, u)[1]  #normalised wave using normalise function
    
    U_sq_list.append(norm_u**2)
    if i % 2 != 0 : #odd states
    
        anal =  np.sin((i+1)*np.pi*x)
        anal_norm = normalize(x, anal)[1]
        
    else : #even states
        anal =  np.cos((i+1)*np.pi*x)
        anal_norm = normalize(x, anal)[1]
    
    Anal_U_sq_list.append(anal_norm**2)

    #plot for U vs X

    plots(x, norm_u,anal_norm, ylabel='Ψ', title = "PLOT OF Ψ VS X FOR N= "+str(i),num=1)

#plot for U**2 vs X on the same plot

plots(x, U_sq_list, Anal_U_sq_list, ylabel='Ψ **2', title = "PROBABILITY DENSITY PLOT OF Ψ**2 VS X", num = len(U_sq_list),color = [['r','b'],['y','g'],['m','c'],['k','violet']])