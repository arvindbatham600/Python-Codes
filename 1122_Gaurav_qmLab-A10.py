#name:gaurav chandra
#rollno : 2020phy1122

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import assoc_laguerre as al


# 2(a)------------------------------------------------------------------------------

def Numerov(a,b,N,n,f,e,l): #numerov method
    h = (b-a)/N     #step size
    x = np.arange(a,b+h,h)
    
    u = np.zeros([N+1])

    C = 1 + ((h**2)/12)*f(e,l,x)
    
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

def v(l,x):
    return l*(l+1)/x**2 -2/x
    
def f(e,l,x):
    return (e - v(l,x))

def bisec(e_min, e_max, n, n_nodes, e_tol,l):  # bisection
    e = (e_min + e_max)/2
    h_nodes = n_nodes - l -1

    while abs(e_min - e_max) >= e_tol:
        
        u = Numerov(10**(-14),150, 1000, n,f,e,l)[1]

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


def normalize(wavefx,wavefy,int_method = MySimp):  #this function returns list including normalisation constant and 
                                          #normalised eigen function
    I = int_method(wavefx,wavefy**2)
    A = (I)**(-1/2)
    
    return [A,A*wavefy]

def analytical_sol(x,n,l):
    anal = np.exp(-x/n)*(2*x/n)**l*al(2*x/n,n-l-1 ,2*l+1)
    norm_anal = normalize(x, anal)[1]
    return norm_anal

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

# 2(a)i.----------------------------------------------------------------------------
l0 ,l1,l2= 0,1,2
e_cal = []
e_ana = []

for i in range(1,11):
    e_cal.append(bisec(-1.1,0 , 100, i+1, 0.5*(10**(-10)),l0))
    
    e_ana.append(-1*(i)**(-2))
    
e_cal = np.array(e_cal)
e_ana = np.array(e_ana)

table = pd.DataFrame({'e_calculated': e_cal, 'e_analytical': e_ana})

print("THE FIRST 10 EIGEN VALUES COMPUTED USING FINITE DIFFERENCE METHOD FOR L = 2 ARE : ")

print(table)


# 2(a)ii.----------------------------------------------------------------------------
for i in range(1,5):
    
    sol = Numerov(150,10**(-14), 1000, i, f, e_cal[i-1], l0)
    x = sol[0]
    norm_u = normalize(x, sol[1])[1]  #normalised wave using normalise function
    
    anal = analytical_sol(x, i, l0)
    
    plt.scatter(x,-1*norm_u,s=5,label = 'computed',c = 'r')
    plt.plot(x,anal,label = 'analytical')
    plt.xlabel('X')
    plt.xlim(0,60)
    plt.ylabel('U')
    plt.title("PLOT FOR U VS X FOR N = "+str(i))
    plt.grid();plt.legend()
    plt.show()
  
# 2(b).----------------------------------------------------------------------------
  
print('')  
print("for l = 1")
print("THE FIRST 10 EIGEN VALUES COMPUTED USING FINITE DIFFERENCE METHOD FOR L = 1 ARE : ")

e_cal_1 = []
e_ana_1 = []

for i in range(1,11):
    e_cal_1.append(bisec(-1.1,0 , 100, i, 0.5*(10**(-10)),l1))
    
    e_ana_1.append(-1*(i)**(-2))
    
e_cal_1 = np.array(e_cal)
e_ana_1 = np.array(e_ana)

table = pd.DataFrame({'e_calculated': e_cal, 'e_analytical': e_ana})
print(table)

print("THE FIRST 10 EIGEN VALUES COMPUTED USING FINITE DIFFERENCE METHOD FOR L = 2 ARE : ")

e_cal_2 = []
e_ana_2 = []

for i in range(1,11):
    e_cal_2.append(bisec(-1.1,0 , 100, i, 0.5*(10**(-10)),l2))
    
    e_ana_2.append(-1*(i)**(-2))
    
e_cal_2 = np.array(e_cal)
e_ana_2 = np.array(e_ana)

table = pd.DataFrame({'e_calculated': e_cal, 'e_analytical': e_ana})
print(table)

# 2(c).----------------------------------------------------------------------------

#("plot for probability density")
#part_i  for n=1,l=0
sol = Numerov(150,10**(-14), 1000, 1, f, e_cal[0], l0)
u = sol[1] #n+1 th eigen vectors
x = sol[0] 
u_norm = normalize(x, u)[1]
anal = analytical_sol(x,1,l0)

plots(x, [u_norm**2], [anal**2], title = "probability density plot for n=0",color = [['b','g']])


#part_ii  for n=2 l = 0,1
sol = Numerov(150,10**(-14), 1000, 2, f, e_cal[0], l0)
sol_1 = Numerov(150,10**(-14), 1000, 2, f, e_cal[0], l1)
u_1 = sol[1]
u_2 = sol_1[1]
x = sol[0]
u_norm_1 = normalize(x, u_1)[1]
u_norm_2 = normalize(x, u_2)[1]
anal_1 = analytical_sol(x,2,l0)
anal_2 = analytical_sol(x,2,l1)

plots(x, [u_norm_1**2,u_norm_2**2], [anal_1**2,anal_2**2], title = "probability density plot for n=1",color = [['b','g'],['r','y']])


#part_iii  for n=3 l = 0,1,2
sol = Numerov(150,10**(-14), 1000, 3, f, e_cal[0], l0)
sol_1 = Numerov(150,10**(-14), 1000, 3, f, e_cal[0], l1)
sol_2 = Numerov(150,10**(-14), 1000, 3, f, e_cal[0], l2)
u_1 = sol[1]
u_2 = sol_1[1]
u_3 = sol_2[1]
x = sol[0]
u_norm_1 = normalize(x, u_1)[1]
u_norm_2 = normalize(x, u_2)[1]
u_norm_3 = normalize(x,u_3)[1]
anal_1 = analytical_sol(x,3,l0)
anal_2 = analytical_sol(x,3,l1)
anal_3 = analytical_sol(x,3,l2)
plots(x, [u_norm_1**2,u_norm_2**2,u_norm_3**2], [anal_1**2,anal_2**2,anal_3**2], title = "probability density plot for n=2",color = [['b','g'],['r','y'],['m','orange']])
