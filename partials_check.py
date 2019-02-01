import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from heave_period import HeavePeriod

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,50), \
'omega_wave': np.linspace(0.12,6.28,50)}

EC = {\
'N_EC' : 1}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('tot_M_spar', val=np.random.rand()*1e6)
ivc.add_output('M_turb', val=np.random.rand()*1e6)
ivc.add_output('M_ball', val=np.random.rand()*1e6)
ivc.add_output('D_spar', val=np.random.rand(10)*1e2)

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check', HeavePeriod(), promotes_inputs=['tot_M_spar', 'M_turb', 'M_ball', 'D_spar'], promotes_outputs=['T_heave'])

prob.setup(force_alloc_complex=True)
comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('stddev_tower_stress', 'resp_tower_stress', check_partials_data, binary=True)