"""
    This test is trying to build a Y-shape pipe system in anuga-swmm
    It has two inlet nodes in the same altitude and connect to one outlet node which is in lower altitude
    and see how water flows in the simple system.
    Using the same domain structure like previous testing "test_volume_inlet_op.py"
"""
# ------------------------------------------------------------------------------
# Import necessary modules
# ------------------------------------------------------------------------------
from anuga import Dirichlet_boundary
from anuga import Domain
from anuga import Reflective_boundary
from anuga import Rate_operator
from anuga import Inlet_operator
from anuga import Region
from anuga import rectangular_cross_domain

import anuga
import numpy as np
import math
import time

# ------------------------------------------------------------------------------
# Setup computational domain
# ------------------------------------------------------------------------------

length = 20.
width = 6.
dx = dy = 0.2  # .1           # Resolution: Length of subdivisions on both axes

domain = rectangular_cross_domain(int(length / dx), int(width / dy),
                                               len1=length, len2=width)
domain.set_name('Y_shape_pipedream')  # Output name based on script name. You can add timestamp=True


# print(domain.statistics())


# ------------------------------------------------------------------------------
# Setup initial conditions
# ------------------------------------------------------------------------------
def topography(x, y):
    """Complex topography defined by a function of vectors x and y."""

    z = 0 * x - 5

    # higher pools
    id = x < 10
    z[id] = -3

    # wall
    id = (10 < x) & (x < 15)
    z[id] = 3

    # wall, spilt two inlet areas
    id = (10 > x) & (2.5 < y) & (y < 3.5)
    z[id] = 3

    '''
    not consider the pit's physical depth, just note the physical location
    to setup the depth, just need to modify z[id] = <x>, x represent x meters depth
    '''

    # first pit, locate at (7, 1) radius=1
    id = (x - 7) ** 2 + (y - 1) ** 2 < 1. ** 2
    z[id] -= 0.0

    # second pit, locate at (7, 5) radius=1
    id = (x - 7) ** 2 + (y - 5) ** 2 < 1. ** 2
    z[id] -= 0.0

    # out pit, locate at (17, 3) radius=1
    id = (x - 17) ** 2 + (y - 3) ** 2 < 1 ** 2
    z[id] -= 0.0

    return z


# ------------------------------------------------------------------------------
# Setup initial quantity
# ------------------------------------------------------------------------------

domain.set_quantity('elevation', topography, location='centroids')  # elevation is a function
domain.set_quantity('friction', 0.01)  # Constant friction
domain.set_quantity('stage', expression='elevation', location='centroids')  # Dry initial condition

# ------------------------------------------------------------------------------
# Setup boundaries
# ------------------------------------------------------------------------------
Bi = Dirichlet_boundary([-2.75, 0, 0])  # Inflow
Br = Reflective_boundary(domain)  # Solid reflective wall
Bo = Dirichlet_boundary([-5, 0, 0])  # Outflow
domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

# ------------------------------------------------------------------------------
# Setup inject water
# ------------------------------------------------------------------------------

inlet1_anuga_region = Region(domain, radius=1.0, center=(7., 1.))
inlet2_anuga_region = Region(domain, radius=1.0, center=(7., 5.))
outlet_anuga_region = Region(domain, radius=1.0, center=(17., 3.))

input1_anuga_region = Region(domain, radius=1.0, center=(2., 1.))

inlet1_anuga_inlet_op = Inlet_operator(domain, inlet1_anuga_region, Q=0.0, zero_velocity=True)
inlet2_anuga_inlet_op = Inlet_operator(domain, inlet2_anuga_region, Q=0.0, zero_velocity=True)
outlet_anuga_inlet_op = Inlet_operator(domain, outlet_anuga_region, Q=0.0, zero_velocity=False)

input1_anuga_inlet_op = Inlet_operator(domain, input1_anuga_region, Q=1) #only inlet1 has water inflow.

x = domain.centroid_coordinates[:, 0]
y = domain.centroid_coordinates[:, 1]
indices = np.where(x < 10)

dplotter1 = anuga.Domain_plotter(domain)

from pipedream_solver.hydraulics import SuperLink
import matplotlib.pyplot as plt
import pandas as pd

superjunctions = pd.DataFrame({'name' : [0, 1, 2], 'id' : [0, 1, 2], 'z_inv' : [-4., -4., -6.], 'h_0' : 3*[1e-5], 'bc' : 3*[False], 'storage' : 3*['functional'], 'a' : 3*[0.], 'b' : 3*[0.], 'c' : 3*[1.], 'max_depth' : 3*[np.inf], 'map_x' : 3*[0], 'map_y' : 3*[0]})

superlinks = pd.DataFrame({'name' : [0, 1], 'id' : [0, 1], 'sj_0' : [0, 1], 'sj_1' : [1, 2], 'in_offset' : 2*[0.], 'out_offset' : 2*[0.], 'dx' : [4., 11.], 'n' : 2*[0.01], 'shape' : 2*['circular'], 'g1' : [0.5, 0.25], 'g2' : [0.05, 0.05], 'g3' : 2*[0.], 'g4' : 2*[0.], 'Q_0' : 2*[0.], 'h_0' : 2*[1e-5], 'ctrl' : 2*[False], 'A_s' : 2*[1e-3], 'A_c' : 2*[0.], 'C' : 2*[0.], 'C_uk' : 2*[0.1], 'C_dk' : 2*[0.1] })

superlink = SuperLink(superlinks, superjunctions, internal_links=10, mobile_elements=True, min_depth=0.)

surface_elev = np.array([-3, -3, -5])

input_velocity = 1
dt = 1.
H_js = []
Q_s = []
losses = []

def coupling_function(domain, coupled_model):
    anuga_depths = np.array([inlet1_anuga_inlet_op.inlet.get_average_depth(),
                             inlet2_anuga_inlet_op.inlet.get_average_depth(),
                             outlet_anuga_inlet_op.inlet.get_average_depth()])

    # Compute inflow/outflow to sewer
    C_w = 0.67
    L_w = 0.1 * 2 * np.pi
    Q_in = np.where(coupled_model.H_j <= surface_elev,
                    C_w * L_w * np.sqrt(anuga_depths) * anuga_depths,
                    C_w * L_w * np.sqrt(np.abs(anuga_depths - (coupled_model.H_j - surface_elev)))
                    * anuga_depths - (coupled_model.H_j - surface_elev))
    # Simulate sewer with flow input
    coupled_model.step(Q_in=Q_in, dt=domain.timestep)
    coupled_model.reposition_junctions()

    # Add/remove flows from surface domain
    inlet1_anuga_inlet_op.set_Q(-Q_in[0])
    inlet2_anuga_inlet_op.set_Q(-Q_in[1])
    outlet_anuga_inlet_op.set_Q(-Q_in[2])

start_t = time.time()
for t in domain.evolve(yieldstep=dt, finaltime=100.0,
                       coupling_function=coupling_function,
                       coupling_args=(domain, superlink)):
    print('\n')
    domain.print_timestepping_statistics()

    # Compute volumes
    link_volume = ((superlink._A_ik * superlink._dx_ik).sum() +
                   (superlink._A_SIk * superlink._h_Ik).sum())
    node_volume = (superlink._A_sj * (superlink.H_j - superlink._z_inv_j)).sum()
    sewer_volume = link_volume + node_volume
    total_volume_correct = t * input_velocity
    total_volume_real = domain.get_water_volume() + sewer_volume
    loss = total_volume_real - total_volume_correct

    # Append data
    losses.append(loss)
    H_js.append(superlink.H_j.copy())
    Q_s.append(superlink.Q_ik.copy())

    # Plot
    dplotter1.save_depth_frame(vmin=0.0,vmax=1.0)
end_t = time.time()

H_j = np.vstack(H_js)
Q_ik = np.vstack(Q_s)

anim = dplotter1.make_depth_animation()
anim.save('anim_test.mp4',
          writer = 'ffmpeg', fps = 30)

plt.plot(H_j[:,0], label='Inlet 1')
plt.plot(H_j[:,1], label='Inlet 2')
plt.plot(H_j[:,2], label='Outlet')
plt.legend()
plt.title('Head at junctions')
plt.xlabel('Time (s)')
plt.ylabel('Head (m)')
plt.axhline(-4 + 0.5, linestyle='--', c='k')
plt.axhline(-4 + 0.25, linestyle='--', c='k')
plt.axhline(-6 + 0.5, linestyle='--', c='k')
plt.axhline(-6 + 0.25, linestyle='--', c='k')
plt.show()


