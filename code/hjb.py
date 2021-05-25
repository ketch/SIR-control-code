#!/usr/bin/env python
# encoding: utf-8
r"""
Two-dimensional HJB equation
============================
This code solves the Hamilton-Jacobi-Bellman equation for optimal
control of an SIR epidemic via control of the contact rate.
The time here corresponds to reverse time in the SIR model.

One difficulty is that we solve on a square, but the feasible domain
of the model is a triangle.
"""

from __future__ import absolute_import
import numpy as np
from clawpack import riemann
from scipy.special import lambertw, expit
from scipy.integrate import solve_ivp
from scipy.interpolate import interpn


beta = 0.3
gamma = 0.1
sigma0 = beta/gamma

def x_inf(x,y,sigma):
    return np.where((y>0)+(x!=1/sigma0),-1./sigma * np.real(lambertw(-x*sigma*np.exp(-sigma*(x+y)))),x)


def qinit(state,c1):
    """
    The initial data is the value of the objective at the final (real) time.
    """
    X, Y = state.grid.p_centers
    state.q[0,:,:] = -c1*x_inf(X,Y,sigma0)

    # Deal with the infeasible upper triangle
    for j in range(2,state.q.shape[2]):
    #    state.q[0,-j,j+1:] = state.q[0,-j,j]
        state.q[0,-j+1:,j] = state.q[0,-j,j]
    #state.q[0,:,:] = np.where(X+Y<1,state.q[0,:,:],0*state.q[0,:,:])
                

def before_step(solver,state):
    """
    Compute the optimal control value.
    """
    c2 = state.problem_data['c2']
    min_fac = state.problem_data['min_fac']
    num_ghost = solver.num_ghost
    q = solver.qbc[0,:,:].squeeze()
    dx, dy = state.grid.delta
    # Left-difference in x
    dudx = (q[num_ghost:-num_ghost,num_ghost:-num_ghost]-q[num_ghost-1:-num_ghost-1,num_ghost:-num_ghost])/dx
    # Downward difference in y?
    dudy_down = (q[num_ghost:-num_ghost,num_ghost:-num_ghost]-q[num_ghost:-num_ghost,num_ghost-1:-num_ghost-1])/dy
    dudy_up = (q[num_ghost:-num_ghost,num_ghost+1:-num_ghost+1]-q[num_ghost:-num_ghost,num_ghost:-num_ghost])/dy
    # First try centered
    dudy = (q[num_ghost:-num_ghost,num_ghost+1:-num_ghost+1]-q[num_ghost:-num_ghost,num_ghost-1:-num_ghost-1])/(2*dy)

    if c2>0:
        # Optimal control with running cost
        v = 1-sigma0*gamma*state.aux[0,:,:]*state.aux[1,:,:]*(dudy-dudx)/(2.*c2)
        sigma_star = sigma0*np.minimum(1.,np.maximum(0.,v))

        dydt = sigma_star*state.aux[0,:,:]-1.
        dudy = np.where(dydt<0, dudy_down, dudy_up)

        v = 1-sigma0*gamma*state.aux[0,:,:]*state.aux[1,:,:]*(dudy-dudx)/(2.*c2)
        state.aux[2,:,:] = sigma0*np.minimum(1.,np.maximum(min_fac,v))

    else:
        # No cost, minimum contact level
        state.aux[2,:,:] = sigma0*(dudy<dudx) + min_fac*sigma0*(dudx<=dudy)

    # Deal with the infeasible upper triangle
    X, Y = state.grid.p_centers
    for j in range(2,state.q.shape[2]):
    #    state.q[0,-j,j+1:] = state.q[0,-j,j]
        state.q[0,-j+1:,j] = state.q[0,-j,j]
        state.aux[2,-j:,j] = min_fac*sigma0
    #state.aux[2,:,0] = min_fac*sigma0
    #state.q[0,:,:] = np.where(X+Y<1,state.q[0,:,:],0*state.q[0,:,:])


def dq_src(solver,state,dt):
    X, Y = state.grid.p_centers
    c2 = state.problem_data['c2']
    c3 = state.problem_data['c3']
    ymax = state.problem_data['ymax']
    ng = solver.num_ghost
    dq = np.empty(state.q.shape)
    # Take average value of sigma (averaged to right/up) -- this would be WRONG
    #sigcen = 0.5*(solver.auxbc[2,ng:-ng,ng:-ng]+solver.auxbc[2,ng:-ng,ng+1:-ng+1])
    #dq[0,:,:] = dt * c2 * (1-sigcen/sigma0)**2
    # No minus sign here because we integrate backward in time
    dq[0,:,:] = dt * (c2 * (1-state.aux[2,:,:]/sigma0)**2 + c3*(Y-ymax)*expit(100*(Y-ymax)))#/(1+np.exp(-100.*(Y-ymax))))
    return dq

def step_src(solver,state,dt):
    X, Y = state.grid.p_centers
    c2 = state.problem_data['c2']
    c3 = state.problem_data['c3']
    ymax = state.problem_data['ymax']
    ng = solver.num_ghost
    sigcen = 0.5*(solver.auxbc[2,ng:-ng,ng:-ng]+solver.auxbc[2,ng:-ng,ng+1:-ng+1])
    state.q[0,:,:] = state.q[0,:,:] + dt * (c2 * (1-sigcen/sigma0)**2 + c3*(Y-ymax)*expit(100*(Y-ymax))) #/(1+np.exp(-100.*(Y-ymax))))

def setup(outdir='./_output',solver_type='classic',mx=50,my=50,c1=1.,c2=0.,c3=0.,min_fac=0.,ymax=1.):

    from clawpack import pyclaw
    import hjb_nocost

    assert(mx==my) # Current restriction due to how the top of the triangle is handled

    if solver_type == 'classic':
        solver = pyclaw.ClawSolver2D(hjb_nocost)
        solver.dimensional_split = 1
        solver.limiters = pyclaw.limiters.tvd.minmod
        if (c2 != 0) or (c3 != 0):
            solver.step_source = step_src
            solver.source_split = 1
    elif solver_type == 'sharpclaw':
        solver = pyclaw.SharpClawSolver2D(hjb_nocost)
        solver.time_integrator = 'SSP33'
        solver.cfl_max= 1.8
        solver.cfl_desired = 1.7
        solver.lim_type=1
        solver.call_before_step_each_stage = True
        if (c2 != 0) or (c3 != 0):
            solver.dq_src = dq_src

    # Set BCs
    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.bc_lower[1] = pyclaw.BC.extrap
    solver.bc_upper[1] = pyclaw.BC.extrap
    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[1] = pyclaw.BC.extrap
    solver.aux_bc_upper[1] = pyclaw.BC.extrap

    #solver.cfl_max = 0.9
    #solver.cfl_desired = 0.8
    solver.num_waves = 1
    solver.num_eqn = 1
    solver.before_step = before_step

    # Domain:
    x = pyclaw.Dimension(0.0,1.0,mx,name='x')
    y = pyclaw.Dimension(0.0,1.0,my,name='y')
    domain = pyclaw.Domain([x,y])

    num_aux = 3
    state = pyclaw.State(domain,solver.num_eqn,num_aux)

    X, Y = domain.grid.p_nodes
    state.aux[0,:,:] = X[:-1,:-1]
    state.aux[1,:,:] = Y[:-1,:-1]
    state.aux[2,:,:] = 0.

    state.problem_data['beta'] = beta
    state.problem_data['gam'] = gamma
    state.problem_data['c2'] = c2
    state.problem_data['c3'] = c3
    state.problem_data['ymax'] = ymax
    state.problem_data['min_fac'] = min_fac

    qinit(state,c1)

    claw = pyclaw.Controller()
    claw.tfinal = 10.0
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.outdir = outdir
    claw.setplot = setplot
    claw.keep_copy = True

    return claw


def solve_hjb(beta=0.3, gamma=0.1, x0=0.99, y0=0.01, c1=1., c2=1.e-2, c3=0., ymax=1.0, T=100, qmax=1., mx=100, my=100):
    sigma0 = beta/gamma
    min_fac = (1-qmax)

    # Solve HJB PDE
    claw = setup(solver_type='sharpclaw',mx=mx,my=my,c1=c1,c2=c2,c3=c3,ymax=ymax,min_fac=min_fac)
    claw.tfinal = T
    #claw.verbosity = 0
    claw.run()

    # Now solve the SIR system with the control obtained from HJB
    sigmaopt = np.zeros((mx,my,claw.num_output_times+1))
    ttt = np.zeros(claw.num_output_times+1)
    for i in range(claw.num_output_times+1):
        sigmaopt[:,:,i] = claw.frames[i].aux[2,:,:]
        ttt[i] = claw.frames[i].t
        
    x = claw.grid.x.centers
    y = claw.grid.y.centers

    def rhs(t, v):
        dv = np.zeros(2)
        if v[1]<np.min(y):
            sigma = 0.
        else:
            sigma = interpn((x.squeeze(),y.squeeze(),ttt.squeeze()),sigmaopt,(v[0],v[1],T-t))

        dv[0] = -sigma*gamma*v[1]*v[0]
        dv[1] =  sigma*gamma*v[1]*v[0] - gamma*v[1]

        return dv

    y0 = y0 # Initial infected
    x0 = x0
    v0 = np.array((x0,y0))
    times = np.arange(0,T)

    solution = solve_ivp(rhs,[0,T],v0,t_eval=times,method='RK23',rtol=1.e-10,atol=1.e-10,max_step=1e-1)


    xsol = solution.y[0,:]
    ysol = solution.y[1,:]

    sigmastar = np.zeros_like(xsol)
    for i in range(len(sigmastar)):
        # This is a hack just because interpn won't extrapolate
        if ysol[i]<np.min(y[:]):
            sigmastar[i]=0.
        else:
            sigmastar[i] = interpn((x.squeeze(),y.squeeze(),ttt.squeeze()),sigmaopt,(xsol[i],ysol[i],T-times[i]))

    return xsol, ysol, sigmastar, times



def setplot(plotdata):
    """ 
    Plot solution using VisClaw.
    """ 
    from clawpack.visclaw import colormaps

    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Figure for pcolor plot
    plotfigure = plotdata.new_plotfigure(name='q[0]', figno=0)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = 'q[0]'
    plotaxes.scaled = True

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = 0
    plotitem.pcolor_cmap = colormaps.yellow_red_blue
    #plotitem.pcolor_cmin = 0.0
    #plotitem.pcolor_cmax = 1.0
    plotitem.add_colorbar = True
    
    # Figure for contour plot
    plotfigure = plotdata.new_plotfigure(name='contour', figno=1)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = 'q[0]'
    plotaxes.scaled = True

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    plotitem.plot_var = 0
    plotitem.contour_nlevels = 20
    plotitem.contour_min = 0.01
    plotitem.contour_max = 0.99
    plotitem.amr_contour_colors = ['b','k','r']
    
    return plotdata

    
if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
