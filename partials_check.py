import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver, KSComp
from openmdao.utils.visualization import partial_deriv_plot

from mooring_offset import MooringOffset

freqs = {\
'omega' : np.linspace(0.001,4.5,70), \
'omega_wave': np.linspace(0.12,4.5,50)}

EC = {\
'N_EC' : 1}

prob = Problem()
ivc = IndepVarComp()
ivc.add_input('thrust_0', val=831906.22657822)
ivc.add_input('F0_tower_drag', val=74600.40459915)
ivc.add_output('z_moor', val=-77.2)
ivc.add_output('water_depth', val=320.)
ivc.add_output('EA_moor', val=4.42811342e8)
ivc.add_output('mass_dens_moor', val=94.07123018)
ivc.add_output('len_hor_moor', val=700.0)
ivc.add_output('len_tot_moor', val=927.326)

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check', MooringOffset(), promotes_inputs=['z_moor', 'water_depth', 'EA_moor', 'mass_dens_moor', 'len_hor_moor', 'len_tot_moor'], promotes_outputs=['moor_tension_offset_ww', 'eff_length_offset_ww', 'moor_tension_offset_lw', 'eff_length_offset_lw', 'moor_offset'])

prob.setup(force_alloc_complex=True)
prob.run_model()
comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('stddev_tower_stress', 'resp_tower_stress', check_partials_data, binary=True)