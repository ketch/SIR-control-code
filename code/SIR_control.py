import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp
from scipy.special import lambertw, expit

def x_inf(x,y,sigma):
    return -1./sigma * np.real(lambertw(-x*sigma*np.exp(-sigma*(x+y))))

def mu(x,y,sigma):
    return x*np.exp(-sigma*(x+y))

def dxinf_dy(x,y,sigma):
    xinf = x_inf(x,y,sigma)
    return -sigma*xinf/(1-sigma*xinf)



def SIR_forward(qfun=None, beta=0.3, gamma=0.1, x0=0.99, y0=0.01, T=100):
    """ Model the current outbreak using the SIR model."""

    du = np.zeros(3)
    u0 = np.zeros(3)
    if qfun is None:
        qfun = lambda t, u : 0.
    
    def f(t,u):
        qval = qfun(t,u)
        du[0] = -(1-qval)*beta*u[1]*u[0]
        du[1] = (1-qval)*beta*u[1]*u[0] - gamma*u[1]
        return du

    # Initial values
    u0[1] = y0 # Initial infected
    u0[0] = x0

    times = np.linspace(0,T,10000)
    solution = solve_ivp(f,[0,T],u0,t_eval=times,method='RK23',max_step=0.1)
    x = solution.y[0,:]
    y = solution.y[1,:]
    t = solution.t
    
    return x, y, t


def solve_pmp(beta=0.3, gamma=0.1, x0=0.99, y0=0.01, c1=1., c2=1.e-2, c3=0., ymax=0.1, guess=None, T=100, qmax=1., N=1000):
    sigma0 = beta/gamma
    sigma_min = (1-qmax)*sigma0

    def rhs(t, u):
        # Variables: x, y, lambda_1, lambda_2
        du = np.zeros((4,len(t)))

        alpha = expit(100*(u[1,:]-ymax))*(u[1,:]-ymax)
        
        sigma = sigma0*(1 - sigma0*(u[3,:]-u[2,:])*gamma*u[1,:]*u[0,:]/(2*c2))
        sigma = np.maximum(sigma_min,np.minimum(sigma0,sigma))

        du[0,:] = -sigma*gamma*u[1,:]*u[0,:]
        du[1,:] =  sigma*gamma*u[1,:]*u[0,:] - gamma*u[1]
        du[2,:] = (u[2,:]-u[3,:])*sigma*gamma*u[1,:]
        du[3,:] = (u[2,:]-u[3,:])*sigma*gamma*u[0,:] + u[3,:]*gamma - c3*alpha
        return du

    def bc(ua, ub):
        xT = ub[0]; yT=ub[1]
        lam2T = -c1*dxinf_dy(xT,yT,sigma0)
        lam1T = lam2T*(1-1/(xT*sigma0))
        return np.array([ua[0]-x0, ua[1]-y0, ub[2]-lam1T, ub[3]-lam2T])

    tt = np.linspace(0,T,N+1)
    uu = np.zeros((4,N+1))
    xT = 1./sigma0 + 0.05
    yT = 0.
    if guess is not None:
        #last = result.sol(tt)
        uu[0,:] = guess[0,:]
        uu[1,:] = guess[1,:]
        uu[2,:] = guess[2,:]
        uu[3,:] = guess[3,:]
    else:
        uu[0,:] = np.exp(-(beta-gamma)*tt/6)
        uu[1,:] = 0.5*np.exp(-1.e-3*(tt-15)**2)
        uu[2,:] = -c1

    result = solve_bvp(rhs, bc, tt, uu, max_nodes=100000, tol=1.e-6, verbose=0)
    if result.status != 0: 
        print('Solver failed to converge for c2 = ',c2)
        return 0, 0, 0, 0, 0, 0
        #raise Exception('solve_bvp did not converge')
    else:
        print('Solver converged for c2 = ',c2)
        
    x = result.y[0,:]
    y = result.y[1,:]
    lam1 = result.y[2,:]
    lam2 = result.y[3,:]

    sigma = sigma0*(1 - sigma0*(lam2-lam1)*gamma*y*x/(2*c2))
    sigma = np.maximum(sigma_min,np.minimum(sigma0,sigma))
    t = result.x
    # Compute objective
    dt = t[1]-t[0]
    g = lambda yy : expit(100*(yy-ymax))*(yy-ymax)
    J = -x_inf(x[-1],y[-1],sigma0) + dt*c2*np.sum((1-sigma/sigma0)**2) + dt*c3*np.sum(g(y))
    #print(result.message)
    return x, y, sigma, t, result.sol(tt), J


def plot_timeline(x,y,control,t,y2=None,t2=None):
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    ax.plot(t,x)
    ax.plot(t,y)
    ax.plot(t,control)
    if y2 is not None:
        ax.plot(t2,y2,'--r')
        ax.legend(['x','y','$\sigma/\sigma_0$','y (no intervention)']);
    else:
        ax.legend(['x','y','$\sigma/\sigma_0$']);
    plt.xlabel('t');
    ax.autoscale(enable=True, axis='x', tight=True)
    return fig

def plot_timelines(xs,ys,controls,ts,labels):
    palette = plt.get_cmap('tab10')
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    for i in range(len(xs)):
        #ax.plot(ts[i],xs[i])
        ax.plot(ts[i],ys[i],color=palette(i+1),label=labels[i])
        ax.plot(ts[i],controls[i],'--',color=palette(i+1))
    plt.legend()
    plt.xlabel('t');
    ax.autoscale(enable=True, axis='x', tight=True)
    return fig

def plot_phaseplane(xs=None,ys=None,beta=0.3,gamma=0.1,color=None,labels=None,x2=None,y2=None):
    sigma0 = beta/gamma
    N1 = 10; N2=5
    Y, X = np.mgrid[0:1:100j, 0:1:100j]
    U = -beta*X*Y
    V = beta*X*Y - gamma*Y
    x_points = list(np.linspace(0,1,N1)) + list(np.linspace(1./sigma0,1,N2))
    y_points = list(1.-np.linspace(0,1,N1)) + [1.e-6]*N2
    seed_points = np.array([x_points, y_points])

    plt.figure(figsize=(6,6))
    plt.streamplot(X, Y, U, V, start_points=seed_points.T,integration_direction='forward',maxlength=1000,
                   broken_streamlines=False,linewidth=1)
    plt.plot([0,1],[1,0],'-k',alpha=0.5)
    if xs is not None:
        i = -1
        for x, y in zip(xs,ys):
            i += 1
            if color:
                plt.plot(x,y,'-',color=color)
            elif labels:
                plt.plot(x,y,'-',label=labels[i])
            else:
                plt.plot(x,y,'-')
    if x2 is not None:
        plt.plot(x2,y2,'--r')
    plt.plot([gamma/beta, gamma/beta],[0,1-gamma/beta],'--k',alpha=0.5)
    plt.xlim(0,1); plt.ylim(0,1);
    plt.xlabel('x'); plt.ylabel('y');
    fig = plt.gcf()
    return fig
