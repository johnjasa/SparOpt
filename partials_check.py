import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from max_mooring_offset import MaxMooringOffset

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,50), \
'omega_wave': np.linspace(0.12,6.28,50)}

EC = {\
'N_EC' : 2}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('z_moor', val=-77.)
ivc.add_output('water_depth', val=320.)
ivc.add_output('EA_moor', val=7.3e8)
ivc.add_output('mass_dens_moor', val=155.)
ivc.add_output('len_hor_moor', val=848.)
ivc.add_output('len_tot_moor', val=902.)

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check',  MaxMooringOffset(), promotes_inputs=['z_moor', 'water_depth', 'EA_moor', 'mass_dens_moor', 'len_hor_moor', 'len_tot_moor'], \
	promotes_outputs=['moor_tension_max_offset_ww', 'eff_length_max_offset_ww', 'max_moor_offset'])

prob.setup(force_alloc_complex=True)
comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('hull_mom_acc_bend', 'Z_spar', check_partials_data, binary=True)