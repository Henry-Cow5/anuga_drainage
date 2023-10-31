#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:05:31 2023

@author: jimmy
"""

import anuga
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pipedream_solver.hydraulics import SuperLink
import sys

def Newtons_Coupling(superlink, inlet, P, P_new, H, H_new, Q, Q_new, dt, k):
    
    # Query/declare general info
    rho = 1000
    A_R = inlet.inlet.get_area()
    g = 9.81
    
    # Get neighboring information
    
    
    if k == 'uk':
        # Get params
        dx = superlink._dx_uk
        A = superlink.A_uk
        Hj = superlink.H_j[0]
        theta = superlink._theta_uk
        z_inv = superlink._z_inv_uk
        S_o = superlink._S_o_uk
        S_f = superlink._Sf_method_uk
        S_L = superlink._C_uk * Q_new / 2 / g / A**2
        
        # Recursively solve coupled equations
        epsilon = 0.05
        i=0
        P_new_temp = P_new +1
        while P_new-P_new_temp >= epsilon:
            P_new_temp = P_new + rho*Q**2/A*dt + rho*A_R*(Q/A*(H_new - H) + H/A*(Q_new - Q))
            Q_new = Q + (P_new_temp - P)/rho/dx + g*A*dt/dx*(theta*(Hj - z_inv) - H_new +
                                                        dx*S_o - dx*(S_f + S_L))
            P_new = P_new + rho*Q**2/A*dt + rho*A_R*(Q/A*(H_new - H) + H/A*(Q_new - Q))
            i += 1
            if i >= 50:
                print('50 iterations of Newton\'s Method exceeded in momentum coupling. Failing')
                sys.exit(0)
        
    else:
        # Get params
        dx = superlink._dx_dk
        A = superlink.A_dk
        Hj = superlink.H_j[1]
        theta = superlink._theta_dk
        z_inv = superlink._z_inv_dk
        S_o = superlink._S_o_dk
        S_f = superlink._Sf_method_dk
        S_L = superlink._C_dk * Q_new / 2 / g / A**2
        
        # Recursively solve coupled equations
        epsilon = 0.05
        i=0
        P_new_temp = P_new +1
        while P_new-P_new_temp >= epsilon:
            P_new_temp = P_new - rho*Q**2/A*dt + rho*A_R*(Q/A*(H_new - H) + H/A*(Q_new - Q))
            Q_new = Q + (P_new_temp - P)/rho/dx - g*A*dt/dx*(theta*(Hj - z_inv) + H_new +
                                                        dx*S_o - dx*(S_f + S_L))
            P_new = P_new - rho*Q**2/A*dt + rho*A_R*(Q/A*(H_new - H) + H/A*(Q_new - Q))
            i += 1
            if i >= 50:
                print('50 iterations of Newton\'s Method exceeded in momentum coupling. Failing')
                sys.exit(0)

    return(P_new, Q_new)
    
superjunctions = pd.DataFrame({'name' : [0, 1], 'id' : [0, 1], 'z_inv' : [0, 0], 'h_0' : 2*[1e-5], 'bc' : 2*[True], 'storage' : 2*['functional'], 'a' : 2*[0.], 'b' : 2*[0.], 'c' : 2*[1.], 'max_depth' : 2*[np.inf], 'map_x' : 2*[0], 'map_y' : 2*[0]})

superlinks = pd.DataFrame({'name' : [0], 'id' : [0], 'sj_0' : [0], 'sj_1' : [1], 'in_offset' : [0.], 'out_offset' : [0.], 'dx' : [4.], 'n' : [0.01], 'shape' : ['rect_closed'], 'g1' : [0.5], 'g2' : [0.5], 'g3' : [0.], 'g4' : [0.], 'Q_0' : [0.], 'h_0' : [1e-5], 'ctrl' : [False], 'A_s' : [1e-3], 'A_c' : [0.], 'C' : [0.], 'C_uk' : [0.1], 'C_dk' : [0.1], 'dx_uk': [1], 'dx_dk': [1] })

superlink = SuperLink(superlinks, superjunctions, internal_links=10, mobile_elements=True)


length = 11.
width = 6.
dx = dy = 0.2  # .1           # Resolution: Length of subdivisions on both axes

domain = anuga.rectangular_cross_domain(int(length / dx), int(width / dy),
                                               len1=length, len2=width)

domain.set_name('momentum_horizontal')

def topography(x, y):
    """Complex topography defined by a function of vectors x and y."""

    z = 0 * x

    # left pool
    id = x < 4
    z[id] = 0

    # wall
    id = (4 < x) & (x < 8)
    z[id] = 10
    return z

domain.set_quantity('elevation', topography, location='centroids')  # elevation is a function
domain.set_quantity('friction', 0.01)  # Constant friction

# Set initial stage
LHS = [[0,0], [4,0], [4,6], [0,6], [0,0]] # Left hand side polygon

domain.set_quantity('stage', expression='elevation', location='centroids')  # Dry initial condition
domain.add_quantity('stage', 1, polygon=LHS)
# ------------------------------------------------------------------------------
# Setup boundaries
# ------------------------------------------------------------------------------
# Bi = anuga.Dirichlet_boundary([-2.75, 0, 0])  # Inflow
Br = anuga.Reflective_boundary(domain)  # Solid reflective wall
Bo = anuga.Dirichlet_boundary([-5, 0, 0])  # Outflow
domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

# ------------------------------------------------------------------------------
# Setup inject water
# ------------------------------------------------------------------------------

input_velocity=1

inlet_tri_id = domain.get_triangle_containing_point((3.98, 3)) # slightly offset to avoid edge
outlet_tri_id = domain.get_triangle_containing_point((8.02, 3))

inlet_anuga_region = anuga.Region(domain, indices=[inlet_tri_id])
outlet_anuga_region = anuga.Region(domain, indices=[outlet_tri_id])

inlet_anuga_inlet_op = anuga.Inlet_operator(domain, inlet_anuga_region, Q=0.0, zero_velocity=False)
outlet_anuga_inlet_op = anuga.Inlet_operator(domain, outlet_anuga_region, Q=0.0, zero_velocity=False)

anuga_depths = np.array([inlet_anuga_inlet_op.inlet.get_average_stage(),
                          outlet_anuga_inlet_op.inlet.get_average_stage()])

# Initial condition
superjunctions.h_0 = anuga_depths

dplotter1 = anuga.Domain_plotter(domain)

def coupling_function(domain, coupled_model, Q_s, H_s, P_s):
    
    dt = domain.timestep
    
    # Why no work!?
    bound_flow = domain.compute_boundary_flows
    print(bound_flow)
    # Q_AN_uk = domain.compute_boundary_flows[-2]
    # Q_AN_dk = domain.compute_boundary_flows[-1]
    
    # coupled_model.step(Q=[Q_AN_uk, Q_AN_dk], dt=dt)
    Q_uk = superlink.Q_uk
    Q_dk = superlink.Q_dk
    
    inlet_anuga_inlet_op.set_Q(-Q_uk)
    outlet_anuga_inlet_op.set_Q(Q_dk)
    
    # Use domain.quantities dictionary with maybe domain.neighbors?
    # dt = domain.timestep
    # rho = 1000
    # g = 9.8
    
    # anuga_depths = np.array([inlet_anuga_inlet_op.inlet.get_average_stage(),
    #                           outlet_anuga_inlet_op.inlet.get_average_stage()])
    # H_in = anuga_depths[0]
    # H_out = anuga_depths[1]

    # Q_uk = superlink.Q_uk
    # Q_dk = superlink.Q_dk
    
    # # Upstream
    # dx = superlink._dx_uk
    # A = superlink.A_uk
    # Hj = superlink.H_j[0]
    # theta = superlink._theta_uk
    # z_inv = superlink._z_inv_uk
    # S_o = superlink._S_o_uk
    # S_f = superlink._Sf_method_uk
    # S_L = superlink._C_uk * Q_uk / 2 / g / A**2
    
    # Q_uk += dt/dx*(g*A*dx*S_o - g*A*(huk - theta*(Hj - z_inv)) - g*A*dx*(S_f + S_L))
    
    # # Downstream
    # dx = superlink._dx_dk
    # A = superlink.A_dk
    # Hj = superlink.H_j[1]
    # theta = superlink._theta_dk
    # z_inv = superlink._z_inv_dk
    # S_o = superlink._S_o_dk
    # S_f = superlink._Sf_method_dk
    # S_L = superlink._C_dk * Q_dk/ 2 / g / A**2
    
    # Old code!!!!!!!!
    # coupled_model.step(H_bc = anuga_depths, dt=dt)
    # coupled_model.reposition_junctions()
    
    # Q_in_old = inlet_anuga_inlet_op.Q
    # Q_out_old = outlet_anuga_inlet_op.Q
    
    # H_in_old = H_s[-1][0]
    # H_out_old = H_s[-1][1]
    
    # # Query new flows
    # Q_in = coupled_model.Q_uk
    # Q_out = coupled_model.Q_dk

    # # Get t-dt momentum
    # Px_in_old = P_s[-1][0][0]
    # Px_out_old = P_s[-1][1][0]
    
    # # Get current momentum
    # Px_in = inlet_anuga_inlet_op.inlet.get_average_xmom()
    # Px_out = outlet_anuga_inlet_op.inlet.get_average_xmom()

    # Px_in, Q_in = Newtons_Coupling(coupled_model, inlet_anuga_inlet_op, Px_in_old, Px_in, H_in_old, H_in, Q_in_old, Q_in, dt, 'uk')
    # Px_out, Q_out = Newtons_Coupling(coupled_model, outlet_anuga_inlet_op, Px_out_old, Px_out, H_out_old, H_out, Q_out_old, Q_out, dt, 'dk')
    
    # # Update momentum
    # inlet_anuga_inlet_op.inlet.set_xmoms(Px_in)
    # outlet_anuga_inlet_op.inlet.set_xmoms(Px_out)
    
    # # Update flows
    # inlet_anuga_inlet_op.set_Q(-Q_in)
    # outlet_anuga_inlet_op.set_Q(Q_out)
    return()

dt = 1

H0 = np.array([inlet_anuga_inlet_op.inlet.get_average_stage(),
                         outlet_anuga_inlet_op.inlet.get_average_stage()])
Q0 = [np.asarray([0]), np.asarray([0])]

losses = []
H_s = [H0]
Q_s = [Q0]
P_s = [[(0, 0), (0,0)]]

for t in domain.evolve(yieldstep=dt, finaltime=120.0,
                       coupling_function=coupling_function,
                       coupling_args=(domain, superlink, Q_s, H_s, P_s)):    
    # Compute volumes
    link_volume = ((superlink._A_ik * superlink._dx_ik).sum() +
                   (superlink._A_SIk * superlink._h_Ik).sum())
    sewer_volume = link_volume
    total_volume_correct = 4*6*1
    total_volume_real = domain.get_water_volume() + sewer_volume
    loss = total_volume_real - total_volume_correct
    
    anuga_depths = np.array([inlet_anuga_inlet_op.inlet.get_average_stage(),
                             outlet_anuga_inlet_op.inlet.get_average_stage()])
    Px_in = inlet_anuga_inlet_op.inlet.get_average_xmom()
    Px_out = outlet_anuga_inlet_op.inlet.get_average_xmom()

    # Append data
    
    losses.append(loss)
    H_s.append(anuga_depths)
    Q_s.append([inlet_anuga_inlet_op.Q, outlet_anuga_inlet_op.Q])
    P_s.append([(Px_in, 0), (Px_out, 0)])

    # Plot
    dplotter1.save_depth_frame(vmin=0.0,vmax=1.0)
    
H = np.vstack(H_s)
P = np.vstack([[x[0], y[0]] for x, y in P_s])
del Q_s[0]
Q = np.vstack(Q_s)

anim = dplotter1.make_depth_animation()
anim.save('anim_test.gif',
          writer = 'ffmpeg', fps = 30)

plt.figure()
plt.plot(H[:,0], label='Inlet 1')
plt.plot(H[:,1], label='Outlet')
plt.legend()
plt.title('Water Stage at Junctions')
plt.xlabel('Time (s)')
plt.ylabel('Stage (m)')
plt.show()

plt.figure()
plt.plot(Q[0,:], label='Inlet 1')
plt.plot(Q[1,:], label='Outlet')
plt.legend()
plt.title('Flows')
plt.xlabel('Time (s)')
plt.ylabel('Flow (m$^3$/s)')
plt.show()

plt.figure()
plt.plot(P[:,0], label='Inlet 1')
plt.plot(P[:,1], label='Outlet')
plt.legend()
plt.title('Momentum at Junctions')
plt.xlabel('Timestep')
plt.ylabel('Momentum')
plt.show()

plt.figure()
plt.plot(losses)
plt.xlabel('Time (s)')
plt.ylabel('Losses (m$^3$)')
plt.title('Loss')
plt.show()