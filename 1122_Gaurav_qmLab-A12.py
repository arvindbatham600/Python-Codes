#name : Gaurav
#rollno : 2020PHY1122

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import eigh

def V_net(r,l,ratio):
    Vef = (l * (l + 1) / (r ** 2)) - (2 / r)*np.exp(-r/ratio)
    v_r = -2 / (r)
    return Vef, v_r

def fin_diff(X,L,R): 
    #h = (b-a)/(n-1)  #n is number of grid points 
    n = len(X)
    K,V = np.zeros((n,n)),np.zeros((n,n))
    h = X[1]-X[0]
    #X = np.linspace(a,b,n)
    
    v = V_net(X,L,R )[0]
    
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



#PROGRAMMING

# PART _ A
r = np.linspace(0.01, 120, 1000)
ratio = [2,5,10,20,100]

print("For l=0")
    
for i in ratio:
    count = 0
    E  = fin_diff(r, 0,i)[0][1:]
    for k in E :
        if k < 0:
            count+=1
            print("bound state energy eigen value exists for alpha=",i,": ",k)
    print("The number of bound states for alpha = ",i,'is :',count)
    
        
print()
# PART _ B
l = 0
m = 0.511 #MeV/c^2
e = 3.795 #(eV A)^0.5
h_bar = 1973 #eV A
print("For ground state ")
for j in ratio:
    e = fin_diff(r, l, j)[0][0]
    E = e * h_bar**2/(2*m*0.529*0.529 * 10**(6))
    print("The energy value in eV for alpha = ",j,":",E)

#C,D
R = np.arange(0.01,10,0.01)
for i in ratio:
    U = fin_diff(R, 0, i)[1][:,0]

    norm_u = normalize(R, U)[1]
    plt.plot(R,norm_u,label = "for alpha = "+str(i))
plt.xlabel(" x ")
plt.ylabel(" U")
plt.title("PLOT OF U VS X ")
plt.grid()
plt.legend()
plt.show()

for i in ratio:
    U = fin_diff(R, 0, i)[1][:,0]

    norm_u = normalize(R, U)[1]
    plt.plot(R,norm_u**2,label = "for alpha = "+str(i))
plt.xlabel(" x ")
plt.ylabel(" U**2")
plt.title("PROBABILITY DENSITY PLOT OF U**2 VS X ")
plt.grid()
plt.legend()
plt.show()

#E
E= []
for i in ratio:
    E.append(fin_diff(r, 0, i)[0][0]) 
plt.scatter(ratio,E)
plt.xlabel('ratio')
plt.ylabel('Energy')
plt.title('Ground state Energy as a Function Of alpha')
plt.grid()
plt.show()
