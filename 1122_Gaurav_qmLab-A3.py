
#name = gaurav chandra
#rollno = 2020phy1122

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar,m_e,m_p,elementary_charge
from scipy import stats
import pandas as pd
print(hbar)
def func_vector(inde,dep,e):  #inde is z and dep is [u,y] ,here y = du/dz
    
    dU_dZ = dep[1]

    dY_dZ = -1*e*dep[0]

    f = np.array([dU_dZ,dY_dZ])
    
    return f


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

def derivative(arr,h):
    array = np.zeros(len(arr))
    for i in range(len(arr)):
        if i==0:
            array[i] = (arr[1]-arr[0])/h
        elif i==(len(arr)-1):
            array[i] = (arr[-1]-arr[-2])/h
        else:
            array[i] = (arr[i+1]-arr[i-1])/h
    return array

def shooting():
    e = np.linspace(0,260,100)          #array of 100 equispaced energy values
    u = []                              #list to contain the last point of the wavefunction evaluated at each e
    
    for i in e:
        u.append(rk4([-1/2,0,1],1/2,func_vector,100,i)[1][0][-1])
        
    guess_index,guess_e,guess_u = [],[],[]   #these lists are to collect the guess indexes,e values and last element of position array
    energy_eigenvalues = []                   #this list contains the energy eigenvalues 
    
    for i in range(1,len(e)):
        if u[i]*u[i-1] < 0 :        #to ensure the change in sign of position
            guess_index.append(i-1)
            guess_e.append(e[i-1])
            guess_u.append(u[i-1])
            
        
    for i in guess_index:
        E = e[i+1] - u[i+1]*(e[i+1] - e[i])/(u[i+1] - u[i])
        energy_eigenvalues.append(E)
        
        
    print("the index for which the sign changes are : ",guess_index)
    print("")
    print("the energy eigenvalue for which the sign changes are : ",guess_e)
    print("")
    print("the endpoint value of wavefunction for which the sign changes are : ",guess_u)
    print("")
    print("the final energy eigenvalues are : ",energy_eigenvalues)
    print('')
    plt.scatter(e,u)        #scatter plot of e vs u
    plt.plot([0,260],[0,0],color = 'red')
    plt.xlabel("ENERGY EIGENVALUE")
    plt.ylabel("WAVE FUNCTION AT END POINT")
    plt.title("PLOT OF e VS U(0.5) ")
    plt.grid()
    plt.show()
    return energy_eigenvalues
    
    
#PROGRAMMING 

def plots(e,u_dash_i = 1,n = 1):    #u_dash_i is the value of the derivative of u at the start point
    indep_f = 1/2  #final independent value
    ini_cond = [-1/2,0,u_dash_i]      #here -1/2 is initial z(indep),0 is u(-1/2) and 1 is u'(-1/2)
    
    sol = rk4(ini_cond,indep_f,func_vector,50,e)
    sol_norm = normalize(sol[0], sol[1][0], MySimp)  #normalised solution
    
    
    if n % 2 == 0 : #even n
    
        anal =  np.sin((n)*np.pi*sol[0])
        anal_norm = normalize(sol[0], anal, MySimp)[1]
        
    
    else : #odd n
        anal =  np.cos((n)*np.pi*sol[0])
        anal_norm = normalize(sol[0], anal, MySimp)[1]
        
    plt.scatter(sol[0],sol_norm[1],linewidth = 1,label = "calculated")
    plt.plot(sol[0],anal_norm,linewidth = 2,label = "analytical",color = 'red')
    plt.xlabel("X / L ")
    plt.ylabel("U(X / L)")
    plt.title("PLOT FOR e = "+str(np.round(e,3)))
    plt.grid()
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    
    
#part a
    e_vals = np.array(shooting())
    u_dash = [1,-1,-1,1,1]
    for i in range(len(e_vals)):
        plots(e_vals[i],u_dash[i],i+1)

#part b
    n_sq = np.array([])
    for i in range(1,len(e_vals)+1):
        n_sq=np.append(n_sq,i**2)
        
    res = stats.linregress(n_sq,e_vals)     #for linear curve fitting
    
    print("The slope obtained from the curvefitting is :",res.slope)
    print("The actual value should be pi**2 i.e = ",np.pi**2)
    print("")
    
    plt.scatter(n_sq,e_vals,label = 'original')
    plt.xlabel('n**2')
    plt.ylabel('En')
    plt.title("PLOT OF En VS n**2")
    plt.plot(n_sq,res.slope*n_sq + res.intercept,c = 'r',label = 'fitted line')
    plt.legend()
    plt.grid()
    plt.show()

#part c
    prob_cal = []
    prob_anal = []
    
    for i in range(len(e_vals)):
        cal = rk4([-0.5,0,1], 0.5, func_vector, 50, e_vals[i])
        
        sol_norm = normalize(cal[0], cal[1][0], MySimp)
        if (i+1)%2 ==0:
            anal =  (2)**0.5*np.sin((i+1)*np.pi*cal[0])
        else :
            anal = (2)**0.5* np.cos((i+1)*np.pi*cal[0])
        
        plt.scatter(cal[0],sol_norm[1]**2,c = 'r',label = 'calculated probabilty')
        plt.plot(cal[0],anal**2,label = 'analytical probabilty')
        plt.xlabel('x/L')
        plt.ylabel('Probability Density')
        plt.title("PLOT OF PROBABILTY DENSITY VS x/L for e = "+str(np.round(e_vals[i],3)))
        plt.legend()
        plt.grid()
        plt.show()

#part d and e
    L_1 = 5*10**(-10) #metres
    L_2 = 10*10**(-10)
    L_3 = 5*10**(-12)
    def part_d(L,m):
        E_eval = e_vals*(hbar)**2/(2*m*(L**2)*elementary_charge)
        E_anal = n_sq * (np.pi*hbar)**2/(2*m*L*L*elementary_charge)
        
        lists = {'CALCULATED En (eV)':np.round(E_eval,6),'ANALYTICAL En (eV)':np.round(E_anal,6)}
        print(pd.DataFrame(lists))      #for tables
    
    print("--:THE TABLE FOR EVALUATED AND ANALYTICAL ENERGY VALUES :--")
    print('')
    print("for an electron trapped inside a well of width 5 angstrom")
    part_d(L_1, m_e)
    print('')
    print("for an electron trapped inside a well of width 10 angstrom")
    part_d(L_2,m_e)
    print('')
    print("for an proton trapped inside a well of width 5 fermimeter")
    part_d(L_3,m_p)
    
#part f
    lists = rk4([-0.5,0,1], 0.5, func_vector, 100, e_vals[0])
    
    z_ground = lists[0]
    
    u_ground =10*(lists[1][0])
    du_dz_ground = lists[1][1]
    
    doubleder = derivative(du_dz_ground, z_ground[1]-z_ground[0])
    
    exp_z = MySimp(z_ground , u_ground*(z_ground*u_ground))
    exp_z_sq = MySimp(z_ground , (u_ground)*(z_ground**2)*(u_ground))
    exp_p = MySimp(z_ground , du_dz_ground*u_ground)
    exp_p_sq = MySimp(z_ground , u_ground*(doubleder))
    
    del_z = (exp_z_sq - exp_z**2)**0.5
    del_p_dimless = (abs(exp_p_sq) - exp_p**2)**0.5
     
    if del_p_dimless*del_z >= 1/2:
        print("##CHECK FOR UNCERTAINITY PRINCIPLE##")
        print("")
        print("uncertainity in z =",del_z," and uncertainity in dimentionless p = ",del_p_dimless)
        print("uncertainity in z * uncertainity in dimentionless p = ",del_p_dimless*del_z)         
        print("which is greater than 1/2 = ",1/2)
        print('')
        print("HENCE , UNCERTAINITY PRINCIPLE IS VERIFIED")
    else :
        print("UNCERTAINITY PRINCIPLE IS NOT VERIFIED")
        
#part g
    z_new,u_new = [],np.array([])
    u_ground_norm = normalize(z_ground, u_ground, MySimp)[1]
    for i in range(len(z_ground)):
        if z_ground[i]>= -1/4 and z_ground[i]<=1/4:
            z_new.append(z_ground[i])
            u_new= np.append(u_new,u_ground_norm[i])
        else:
            continue
    print('')
    print("THE PROBABILITY OF FINDING AN ELECTRON IN THE RANGE [-L/4:L/4] IS ",MySimp(z_new, u_new**2),' = ',MySimp(z_new, u_new**2)*100,'%')