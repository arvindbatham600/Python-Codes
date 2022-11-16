#name = gaurav chandra
#rollno = 2020phy1122

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar,pi,epsilon_0,e,m_e     #for values of constants
import pandas as pd


def fcln(n):    #frequency obtained from classical mech formula
    f = m_e*e**4/(32*(pi**3)*(epsilon_0**2)*(hbar**3)*n**3)

    return f


def fqn(n):     #frequency obtained from quantum mech formula
    f = (m_e*e**4*(2*n-1))/(64*(pi**3)*(epsilon_0**2)*(hbar**3)*(n**2)*(n-1)**2)

    return f
  
    
def En(n):
    e = -13.6/(n)**2
    
    return e


#programming 

#plot for energy levels for different n

def energy_plot():
    x = np.arange(1,11)

    for i in range(1,11):
        plt.plot(x,[En(i)]*len(x),label = 'n='+str(i))    
        

    plt.ylabel("En")
    plt.legend(loc = 'lower right')
    plt.grid()
    plt.show()



def Table(tol = 10**(-5)):
    p = np.array([0.5])
    n = 10**p

    cln_arr = np.array([fcln(n[-1])])
    qn_arr = np.array([fqn(n[-1])])

    rel_err = np.array([abs(qn_arr[-1] - cln_arr[-1])*100/qn_arr[-1]])
    
    
    while rel_err[-1] >= tol :

        p = np.append(p,p[-1] + 0.5)
        
        n = 10**p[-1]

        cln_arr = np.append(cln_arr,fcln(n))
        qn_arr = np.append(qn_arr,fqn(n))

        rel_err = np.append(rel_err,abs(qn_arr[-1] - cln_arr[-1])*100/qn_arr[-1])
    
    
    n = 10**p
    
    Data = pd.DataFrame({'n':n,'f_cln':cln_arr,'f_qn':qn_arr,"rel error":rel_err})
    

    print(Data)

    plt.plot(n,rel_err)
    plt.title("plot between relative error and ln(n)")
    plt.xlabel("rel error")
    plt.ylabel("ln(n)")
    plt.xscale('log')
    plt.grid()
    plt.show()
    
    
energy_plot()
Table()