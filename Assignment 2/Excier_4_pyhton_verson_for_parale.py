import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
import os 
import copy
import time

#%%



def g_BC_func(t,PAR):
    vareps=PAR['vareps']
    gL,gR=1-np.tanh((-1/2-t)/(2*vareps)),1-np.tanh((3/2-t)/(2*vareps))
    return gL,gR



def delta_x_uniform(N):
    dx=2/(N+1)
    return dx


def System(t,U,PAR):
    vareps,N=PAR['vareps'],PAR['N']
    dU=np.zeros(N)
    
    dx=delta_x_uniform(N)
    gL,gR=g_BC_func(t,PAR)
    dU[0]=vareps*(gL-2*U[0]+U[1])/(dx**2)-U[0]*(U[0]-gL)/dx
    dU[-1]=vareps*(U[-2]-2*U[-1]+gR)/(dx**2)-U[-1]*(U[-1]-U[-2])/dx
    dU[1:-1]=vareps*(U[0:-2]-2*U[1:-1]+U[2:])/(dx**2)-U[1:-1]*(U[1:-1]-U[0:-2])/dx
    return dU


def Eta_IC_and_xvec(PAR):
    vareps=PAR['vareps']
    dx=delta_x_uniform(PAR['N'])
    U0=np.zeros(PAR['N'])
    x=np.linspace(-1,1,PAR['N']+2)
    U0[:]=1-np.tanh((x[1:-1]+1/2)/(2*vareps))
    return x,U0

def solve_system(PAR):
    t_span=PAR['t_span']
    N=PAR['N']
    r_tol=PAR['r_tol']
    a_tol=PAR['a_tol']
    xvec,U0=Eta_IC_and_xvec(PAR)
    sol=solve_ivp(System,t_span,U0,args=(PAR,),method='RK45',rtol=r_tol,atol=a_tol)
    g_L,g_R=g_BC_func(sol.t,PAR)
    U=np.zeros((N+2,len(sol.t)))
    U[0,:]=g_L
    U[-1,:]=g_R
    U[1:-1,:]=sol.y
    return sol.t,xvec,U

def Ana_lytical_sol(t,x,PAR):
    vareps=PAR['vareps']
    gL,gR=g_BC_func(t,PAR)
    U=np.zeros((len(x),len(t)))
    for i in range(len(t)):
        U[:,i]=1-np.tanh((x+1/2-t[i])/(2*vareps))
    return U

PAR={}
PAR['N']=100
PAR['vareps']=0.05
PAR['t_span']=(0,2)
PAR['r_tol']=1e-8
PAR['a_tol']=1e-9
colorvec=['b','r','g','c','m','y','DarkOrange','lawngreen','k','DarkRed','DarkGreen']
tvec,xvec,U=solve_system(PAR)
U_ana=Ana_lytical_sol(tvec,xvec,PAR)    
t_range_vec=np.array([0,1.4,1.49,1.5,1.51,1.6,2])
plt.figure(figsize=(8,6),dpi=400)
for i in range(0,len(t_range_vec)):
    index=np.min(np.where(tvec>=t_range_vec[i]))
    plt.plot(xvec,U[:,index],'.',label='t = '+str(round(tvec[index],2)),color=colorvec[i])
    plt.plot(xvec,U_ana[:,index],'--',label='t = '+str(round(tvec[index],2))+' (ana)',color=colorvec[i])   
plt.xlabel('x')
plt.ylabel('U')
plt.xlim(-1,1)
plt.ylim(0,2.1)
plt.grid(True)
plt.legend()


fig,ax=plt.subplots(2,2,figsize=(12,6),dpi=400)
axs=ax.flat


eps_vec=np.array([0.01,0.05,0.25,1])
for epps in range(0,len(eps_vec)):
    PAR['vareps']=eps_vec[epps]
    tvec,xvec,U=solve_system(PAR)
    U_ana=Ana_lytical_sol(tvec,xvec,PAR)
    for i in range(0,len(t_range_vec)):
        index=np.min(np.where(tvec>=t_range_vec[i]))
        axs[epps].plot(xvec,U[:,index],'.',label='t = '+str(round(tvec[index],2)),color=colorvec[i])
        axs[epps].plot(xvec,U_ana[:,index],'--',label='t = '+str(round(tvec[index],2))+' (ana)',color=colorvec[i])   
    axs[epps].set_xlabel('x')
    axs[epps].set_ylabel('U')
    axs[epps].set_xlim(-1,1)
    axs[epps].set_ylim(0,2.1)
    axs[epps].grid(True)
    axs[epps].set_title('$\\varepsilon $ = '+str(eps_vec[epps]))
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=7, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.05, 1, 1])    

#%%

def Convergency_test(PAR):
    start = time.time()
    N=PAR['N']
    a_tol,r_tol=PAR['a_tol'],PAR['r_tol']
    dx=delta_x_uniform(N)
    t_span=PAR['t_span']
    tvec,xvec,U=solve_system(PAR)
    U_ana=Ana_lytical_sol(tvec,xvec,PAR)
    err_vec=np.zeros_like(U_ana)
    err_vec[:,:]=np.abs(U-U_ana)
    error_val=np.max(err_vec)

    print(f"N={N} done in {round(time.time() - start, 3)} sec")
    return dx,error_val

def run_single_test(N, PAR_copy):
    # We must pass a deep copy of PAR to avoid shared state
    local_PAR = copy.deepcopy(PAR_copy)
    local_PAR['N'] = N
    dx, err = Convergency_test(local_PAR)
    return dx, err

def main():
    freeze_support()

    PAR = {
        'r_tol': 1e-8,
        'a_tol': 1e-9,
        't_span': (0, 2)
    }

    vareps_vec = np.array([0.001, 0.005])
    Nvec = np.logspace(1, 4.5, 40).astype(int)
    print('Nvec:', Nvec)

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), dpi=200)
    axs = ax.flat

    for i in range(len(vareps_vec)):
        PAR['vareps'] = vareps_vec[i]
        PAR_copy = PAR.copy()

        try:
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(run_single_test, Nvec, [PAR_copy] * len(Nvec)))
        except Exception as e:
            print(f"Error during parallel execution: {e}")
            results = []

        if results:
            dx_vec, err_vec = zip(*results)
            axs[i].loglog(dx_vec, err_vec, 'o-', label=f'$\\varepsilon$ = {vareps_vec[i]}')
            axs[i].set_xlabel('dx')
            axs[i].set_ylabel('Error')
            axs[i].set_title('Convergence test')
            axs[i].grid(True)
            axs[i].legend()
        else:
            axs[i].set_title('Error occurred')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()