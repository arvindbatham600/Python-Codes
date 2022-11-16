#name = gaurav chandra
#rollno = 2020phy1122

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar,m_e


def func_vector(inde,dep,e):  #inde is z and dep is [u,m] ,here m = du/dz
    
    dU_dZI = dep[1]

    dY_dZI = -1*e*dep[0]

    f_vector = np.array([dU_dZI,dY_dZI])
    
    return f_vector


def rk4(ini_cond,inde_f,func,N,e):  #RK4 METHOD for solving simulatenous differential equation
    h = (inde_f - ini_cond[0])/(N)       
    time_vect = np.array([ini_cond[0]])
    a = []
    for i in range(1,len(ini_cond)):
        a.append(ini_cond[i])
    y_vect = np.array(a).reshape(len(ini_cond)-1,1)
    
    for i in range(N):
        m1_vect = h*func(time_vect[i],y_vect[:,i],e)
        m2_vect = h*func(time_vect[i] + (h/2),y_vect[:,i] + (m1_vect/2),e)
        m3_vect = h*func(time_vect[i] + (h/2),y_vect[:,i] + (m2_vect/2),e)
        m4_vect = h*func(time_vect[i] + h,y_vect[:,i]+m3_vect,e)
        mrk4 = (1/6)*(m1_vect + 2*m2_vect + 2*m3_vect + m4_vect)
        t = time_vect[i] + h
        t_vect = np.append(time_vect,t)
        time_vect = t_vect
        y_next = y_vect[:,i] + mrk4
        Y = []
        for j in range(len(y_vect)):
            y = np.append(y_vect[j],y_next[j])
            Y.append(y)
        y_vect = np.array(Y)
    return [time_vect,y_vect]

def Rk4(ini_cond,inde_f,func,N,e):  #RK4 METHOD for solving simulatenous differential equation
    h = (inde_f - ini_cond[0])/(N)       
    indep =[ini_cond[0]]
    dep1 = [ini_cond[1]]
    dep2 = [ini_cond[2]]
    for i in range(N):
        m1 = h*func(indep[i],[dep1[i],dep2[i]],e)
        
        m2 = h*func(indep[i] + (h/2),[dep1[i] + (m1[0]/2),dep2[i]+(m1[1]/2)],e)
        
        m3 = h*func(indep[i] + (h/2),[dep1[i] + (m2[0]/2),dep2[i]+(m2[1]/2)],e)
        
        m4 = h*func(indep[i] + h,[dep1[i]+m3[0],dep2[i]+m3[1]],e)
        
        mrk4 = (1/6)*(m1 + 2*m2 + 2*m3 + m4)
        
        indep.append(indep[i]+h)
        dep1.append(dep1[i]+mrk4[0])
        dep2.append(dep2[i]+mrk4[1])

    return [indep,[dep1,dep2]]

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

def normalize(wavefx,wavefy,int_method):  #this function returns list including normalisation constant and 
                                          #normalised eigen function
    I = int_method(wavefx,wavefy**2)
    A = (I)**(-1/2)
    
    return [A,A*wavefy]

sol_1 = rk4([-0.5,0,1],0.5,func_vector,50,9.86)
sol_2 = Rk4([-0.5,0,1],0.5,func_vector,50,9.86)

plt.plot(sol_1[0],sol_1[1][0])
plt.scatter(sol_2[0],sol_2[1][0],c = 'g')
plt.show()
#PROGRAMMING 


def plots(e,title,u_dash_f = [1,5],n = 1):
    indep_f = 1/2  #final independent value
    ini_cond_1 = [-1/2,0,u_dash_f[0]]      #here -1/2 is initial z(indep),0 is u(-1/2) and 1 is u'(-1/2)
    ini_cond_2 = [-1/2,0,u_dash_f[1]]
    
    sol_1_1 = rk4(ini_cond_1,indep_f,func_vector,50,e)
    sol_1_1_norm = normalize(sol_1_1[0], sol_1_1[1][0], MySimp)
    
    sol_1_2 = rk4(ini_cond_2,indep_f,func_vector,50,e)
    sol_1_2_norm = normalize(sol_1_2[0], sol_1_2[1][0], MySimp)
    if n % 2 == 0 : #even n
    
        anal_1 =  np.sin((n)*np.pi*sol_1_1[0])
        anal_1_norm = normalize(sol_1_1[0], anal_1, MySimp)[1]
        anal_2 =  np.sin((n)*np.pi*sol_1_2[0])
        anal_2_norm = normalize(sol_1_2[0], anal_2, MySimp)[1]
    
    else : #odd n
        anal_1 =  np.cos((n)*np.pi*sol_1_1[0])
        anal_1_norm = normalize(sol_1_1[0], anal_1, MySimp)[1]
        anal_2 =  np.cos((n)*np.pi*sol_1_2[0])
        anal_2_norm = normalize(sol_1_2[0], anal_2, MySimp)[1]
    
    
    
    fig, axs = plt.subplots(1,2)
    axs[0].plot(sol_1_1[0],sol_1_1_norm[1],linewidth = 4,label = "computed")
    axs[0].plot(sol_1_1[0],anal_1_norm,linewidth = 2,label = "analytical")
    axs[1].plot(sol_1_2[0],sol_1_2_norm[1],linewidth = 4,label = "computed")
    axs[1].plot(sol_1_2[0],anal_2_norm,linewidth = 2,label = "analytical")
    axs[0].set_xlabel("X / L ")
    axs[0].set_ylabel("U(X / L)")
    axs[1].set_xlabel("X / L")
    axs[1].set_ylabel("U(X / L)")
    
    axs[0].set_title("PLOT FOR u'(-1/2) = "+str(ini_cond_1[-1]))
    axs[1].set_title("PLOT FOR u'(-1/2) = "+str(ini_cond_2[-1]))
    axs[0].grid()
    axs[1].grid()
    axs[0].legend()
    axs[1].legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
  
    
def partd(n = 1,tol = 0.5*10**(-5)):
    L = 2*10**(-10)
    E = 0
    a = (n**2-0.1) * np.pi**2
    b = (n**2+0.1) * np.pi**2
    h = (b-a)/100
    arr = np.arange(a, b+h,h)
    for e in arr:
        ini_cond = [-1/2,0,1]      #here -1/2 is initial z(indep),0 is u(-1/2) and 1 is u'(-1/2)
        indep_f = 1/2
        sol_1_1 = rk4(ini_cond,indep_f,func_vector,50,e)
        sol_1_1_norm = normalize(sol_1_1[0], sol_1_1[1][0], MySimp)[1]
        
        if abs(sol_1_1_norm[-1]) < tol:
            E = e
            break
        
    print("The Eigenvalue of the dimentionless schrodinger equation is ",E)
    if n == 1:
        plots(E,"PLOT FOR GROUND STATE NORMALISED EIGEN FUNCTION FOR e = "+str(np.round(E,3)),n= n)
    elif n==2:
        plots(E,"PLOT FOR FIRST EXCITATION STATE NORMALISED EIGEN FUNCTION FOR e = "+str(np.round(E,3)),u_dash_f = [-1,-5],n = n)
    else : 
        plots(E,"PLOT FOR " + str(n-1)+ " EXCITATION STATE NORMALISED EIGEN FUNCTION FOR e = "+str(np.round(E,3)),u_dash_f = [-1,-5],n = n)
    
    E_dash = E*(hbar)**2/(2*m_e*(L**2))
    print("The Eigenvalue of the dimentionless schrodinger equation is",E_dash," eV")
    

if __name__ == "__main__":
#part b
    e = 8
    plots(e, "PLOT FOR X VS U(X) FOR e ="+str(e))
#part c    
    e = 11
    plots(e, "PLOT FOR X VS U(X) FOR e ="+str(e))
    
#part d
    print("--:THE GROUND STATE:--")
    print('')
    partd()
    print('')
#part e
    print("--:THE FIRST EXCITATION STATE:--")
    print('')
    partd(n=2,tol = 0.5*10**(-4))
    