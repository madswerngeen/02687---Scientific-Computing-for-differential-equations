import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
import math
import os 
import copy
import time
from matplotlib.colors import BoundaryNorm

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

def Function_to_determind_coeficents_not_centered(n,alpha,beta,h):
    #n is the order for the derivative 
    # alpha is numder of nodes to the left 
    # beta is the number of nodes to the right
    N=alpha+beta+1
    A=np.zeros([N,N])
    
    for k in range(0,alpha):
        for L in range(0,N):
            A[L,k]=(-1)**L*(alpha-k)**L
    A[0,alpha]=1
    A[1:,alpha]=0
    for k in range(alpha+1,N):
        for L in range(0,N):
            A[L,k]=(k-alpha)**L
            
    bvec=np.zeros(N)
    bvec[n]=math.factorial(n)/(h**n)
            
    acoef=np.linalg.solve(A,bvec)
    return A,acoef


def Function_to_determind_coeficents_centered(n,N,h):
    #n is the order for the derivative 
    # alpha is numder of nodes to the left 
    # beta is the number of nodes to the right
    N0=N
    N=N+1
    A=np.zeros([N,N])
    A[0,:]=np.ones(N)
    for k in range(1,N): 
        for L in range(0,N):
            A[L,k]=2*(k)**(2*L) # the 2 times is becaue asume that a_1=a_{-1} the 2L is becausesipping odd dievrivevs
          
    bvec=np.zeros(N)
    nn=int(n/2) # beacise there is skipped ood diverivatives
    bvec[nn]=math.factorial(n)/(h**n)
            
    acoef=np.linalg.solve(A,bvec)
    return A,acoef        


def Function_to_determind_coeficents(PAR,Flag=0):
    # Flag=0 for mixed 
    # Flag for centered equal stepsize
    n=PAR['n']# Order of the derivative
   
    h=PAR['h']
    if Flag==0: # this is none center method,
        alpha=PAR['alpha']# Number of nodes to the left
        beta=PAR['beta']# Number of nodes to the right
        # the order of coefricent is j=-alpha, j=-alpha+1,j=-alpha+2 .... j=0 ... j=1 ... j=beta
        A,acoef_1=Function_to_determind_coeficents_not_centered(n,alpha,beta,h)
        acoef=np.flip(acoef_1)
    elif Flag==1: 
        N=PAR['N_nodes_central']# Number to right and left
        # the order of coefices is j=0, j=1 ... j=N
        A,acoef=Function_to_determind_coeficents_centered(n,N,h)
    return A,acoef     









#%%


PAR={}
PAR['n']=2
PAR['h']=1
PAR['alpha']=6
PAR['beta']=0
A,coef=Function_to_determind_coeficents(PAR,0)
PAR['coef_advection']=coef
#print(A)
print('test ')
print(coef)
print(35/12,-26/3,19/2,-14/3,11/12)
PAR['n']=1
print('__________________________')
PAR['N_nodes_central']=2
A,coef=Function_to_determind_coeficents(PAR,1)
print('Central')
print(coef)
print('__________________________')

abe=np.array([1,2,3,4])

PAR['coef_central']=coef
# %%
PAR['N']=10
def g_BC_funcNew(t,PAR):
    gL,gR=0,0
    return gL,gR





#def Diffustion_stencils(U,PAR):
 #   coef_diff=PAR['coef_central']
  #  N_nodes_diff=int(len(coef_diff))
   # N_u=int(len(U))
    #d2Udx2=np.zeros(N_u)
    #d2Udx2[:]=U[:]*coef_diff[0]
    #for l in range(1,N_nodes_diff):
    #    if l==1:
     #       d2Udx2[l:-l]+=coef_diff[l]*(U[1+l:]+U[l-1:-l-1])
      #  elif l>1:
       #     d2Udx2[l:-l]+=coef_diff[l]*(U[1+l:-l+1]+U[l-1:-l-1])

        #    d2Udx2[1:l]+=coef_diff[l]*U[1+l:2*l]  
 #           d2Udx2[-l-1:]+=coef_diff[l]*U[-2*l:-l+1]
#
  #  return d2Udx2

def Diffustion_stencils(U,PAR):
    coef_diff=PAR['coef_central']
    N_alpha=PAR['alpha']# number of nodes points used in the backwards stencile to aprosimate the first order derivative
    N=PAR['N'] # nuber of nodes that is none ghost poitns
    N_nodes_diff=int(len(coef_diff))
    N_u=int(len(U))
    d2Udx2=np.zeros(N_u) # where U[N_alpha] is the first point that is not ghost point
    d2Udx2[N_alpha:N_alpha+N]=U[N_alpha:N_alpha+N]*coef_diff[0]
    for l in range(1,N_nodes_diff):
        d2Udx2[N_alpha:N_alpha+N]+=coef_diff[l]*(U[N_alpha+l:N_alpha+N+l]+U[N_alpha-l:N_alpha+N-l])
    return d2Udx2



def Advection_stencils(U,PAR):
    coef_ad=PAR['coef_advection']
    N_alpha=PAR['alpha']# number of nodes points used in the backwards stencile to aprosimate the first order derivative
    N=PAR['N'] # nuber of nodes that is none ghost poitns
    N_u=int(len(U))
    dudX=np.zeros(N_u)
    dudX[N_alpha:N_alpha+N]=U[N_alpha:N_alpha+N]*coef_ad[0]
    for l in range(1,len(coef_ad)):
        dudX[N_alpha:N_alpha+N]+=coef_ad[l]*U[N_alpha-l:N_alpha+N-l]
    return dudX

def Advection_stencils_v2(U,PAR):
    coef_ad=PAR['coef_advection']
    N_alpha=PAR['alpha']
    # number of nodes points used in the backwards stencile to aprosimate the first order derivative
    N=PAR['N'] # nuber of nodes that is none ghost poitns
    N_u=int(len(U))
    dudX=np.zeros(N_u)
    dudX[N_alpha:N_alpha+N]=U[N_alpha:N_alpha+N]*coef_ad[0]
    for l in range(1,len(coef_ad)):
        dudX[N_alpha:N_alpha+N]+=coef_ad[l]*U[N_alpha-l:N_alpha+N-l]
    return dudX




def System_newstencils_intial(U0,PAR):
    """
    This functuion inttaluis the sytems of PDEs
    """
    N_grid=len(U0)
    PAR['N']=N_grid
    dx=delta_x_uniform(N_grid)
    PAR['dx']=dx
    PAR['h']=1
    alpha=PAR['alpha']
    N_central=PAR['N_nodes_central']
    if N_central>alpha:
        print('Error: N_central should be less or equal than alpha')
    PAR['n']=1
    A,coef=Function_to_determind_coeficents(PAR,0) # is not centered
    PAR['coef_advection']=coef
    PAR['n']=2
    A,coef=Function_to_determind_coeficents(PAR,1)# # is centered
    PAR['coef_central']=coef

    U=np.zeros(int(N_grid+alpha+N_central))
    U[alpha:alpha+N_grid]=U0
    U[0:alpha]=0
    U[-N_central:]=0
    return U

def System_newstencils(t,U,PAR):
    vareps,N=PAR['vareps'],PAR['N']
    alpha=PAR['alpha']
    N_central=PAR['N_nodes_central']
    dx=PAR['dx']
    dU=np.zeros(len(U))
    
    
    d2Udx2=Diffustion_stencils(U,PAR)
    dUdX=Advection_stencils(U,PAR)
    dU[alpha:alpha+N]=vareps/(dx**2)*d2Udx2[alpha:alpha+N]-dUdX[alpha:alpha+N]*U[alpha:alpha+N]/dx
   
    return dU


def System_newstencils_v2(t,U,PAR):
    vareps,N=PAR['vareps'],PAR['N']
    alpha=PAR['alpha']
    N_central=PAR['N_nodes_central']
    dx=PAR['dx']
    dU=np.zeros(len(U))
    
    
    d2Udx2=Diffustion_stencils(U,PAR)
    dUdX=Advection_stencils(U**2,PAR)
    dU[alpha:alpha+N]=vareps/(dx**2)*d2Udx2[alpha:alpha+N]-dUdX[alpha:alpha+N]*1/(2*dx)
   
    return dU


def IC_and_xvec(PAR):
    N=PAR['N']
    xvec=np.linspace(-1,1,N+2)
    U0=-np.sin(np.pi*xvec[1:-1])
    return xvec,U0

def solve_system(PAR):
    xvec,U0=IC_and_xvec(PAR)
    U_new0=System_newstencils_intial(U0,PAR)

    t_span=PAR['t_span']
    r_tol=PAR['r_tol']
    a_tol=PAR['a_tol']
    N=PAR['N']
    print('N',N)
    
    alpha=PAR['alpha']
    print('alpha',PAR['alpha'])

    sol=solve_ivp(System_newstencils_v2,t_span,U_new0,args=(PAR,),method='RK45',rtol=r_tol,atol=a_tol,max_step=0.1)
    U=np.zeros((N,len(sol.t)))
    print(np.shape(sol.y))
    U[:,:]=sol.y[alpha:alpha+N,:]
    return sol.t,xvec,U


PAR={}
PAR['N']=5000
PAR['alpha']=3
PAR['beta']=0
PAR['vareps']=0.01/np.pi#0.001/np.pi
PAR['N_nodes_central']=3
PAR['r_tol']=1e-8
PAR['a_tol']=1e-10
PAR['t_span']=(0,2)

tvec,xvec,U=solve_system(PAR)

timestams=np.array([0,0.025,0.05,0.1,0.25,0.5,1])*tvec[-1]
boundaries = timestams
norm = BoundaryNorm(boundaries, ncolors=256)
cmap = cm.copper
colors=cm.viridis(np.linspace(0,np.max(tvec),len(timestams)))
fig, ax = plt.subplots(dpi=300)
for i in range(len(timestams)):
    index=np.min(np.where(tvec>=timestams[i])[0])
    color = cmap(norm(timestams[i]))
    ax.plot(xvec[1:-1],U[:,index],label='time ='+str(round(timestams[i],3)),color=color)
    
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_xlabel('x')
ax.set_ylabel('U')
ax.grid(True)    
ax.legend()
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]) 
cbar = fig.colorbar(sm,ax=ax)
plt.show()
#%%

def U_gradinent(U,xvec,tvec,xva,tvval):
    index_x=np.min(np.where(xvec>=xva)[0])
    index_t=np.min(np.where(tvec>=tvval)[0])
    dx=abs(xvec[index_x+1]-xvec[index_x])
    gradient_simple=(U[index_x+1,index_t]-U[index_x-1,index_t])/(xvec[index_x+1]-xvec[index_x-1])
    gradient_central=(1 / (60 * dx)) * ( -U[index_x+3, index_t] + 9 * U[index_x+2, index_t]- 45 * U[index_x+1, index_t] + 45 * U[index_x-1, index_t]- 9 * U[index_x-2, index_t] + U[index_x-3, index_t])
    gradient_central_simple=(U[index_x+1,index_t]-U[index_x-1,index_t])/(xvec[index_x+1]-xvec[index_x-1])

    print(gradient_central,gradient_simple,gradient_central_simple)
    return print(gradient_central,gradient_simple,gradient_central_simple)


U_gradinent(U,xvec,tvec,0,1.6037/np.pi)
print(U_gradinent)
# %%
"""
With none unifrom grid 
Here there will not be used central differnce for diffsion term
"""

def delta_x_none_uniform(grid_info):
    N_fine=grid_info['N_fine']
    N_coarse=grid_info['N_coarse']
    x0=grid_info['x0']
    r_xo=grid_info['r_xo']

    Coursepartgird_left=np.linspace(-1,x0-r_xo,N_coarse+1)
    Finepartgird=np.linspace(x0-r_xo,x0+r_xo,N_fine+1)
    Coursepartgird_right=np.linspace(x0+r_xo,1,N_coarse+1)
    xvec=np.concatenate((Coursepartgird_left,Finepartgird,Coursepartgird_right))
    dx=np.zeros(len(xvec)-1)
    dx[:]= xvec[1:]-xvec[:-1]
    return dx,xvec

grid_info={}
grid_info['N_fine']=1000
grid_info['N_coarse']=100
grid_info['x0']=0
grid_info['r_xo']=0.1
dx,xvec=delta_x_none_uniform(grid_info)


def IC_and_xvec_v2(grid_info):
    dx,xvec=delta_x_none_uniform(grid_info)
    U0=-np.sin(np.pi*xvec[1:-1])
    
    return U0,xvec,dx


def System_newstencils_intial(U0,PAR,grid_info):
    """
    This functuion inttaluis the sytems of PDEs
    """
    
    dx,xvec=delta_x_none_uniform(grid_info)

    PAR['dx']=dx
    PAR['h']=1
    alpha=PAR['alpha']
    N_central=PAR['N_nodes_central']
    if N_central>alpha:
        print('Error: N_central should be less or equal than alpha')
    PAR['n']=1
    A,coef=Function_to_determind_coeficents(PAR,0) # is not centered
    PAR['coef_advection']=coef
    PAR['n']=2
    A,coef=Function_to_determind_coeficents(PAR,1)# # is centered
    PAR['coef_central']=coef

    U=np.zeros(int(N_grid+alpha+N_central))
    U[alpha:alpha+N_grid]=U0
    U[0:alpha]=0
    U[-N_central:]=0
    return U