import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from spar_cost import SparCost

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,50), \
'omega_wave': np.linspace(0.12,6.28,50)}

EC = {\
'N_EC' : 1}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('D_spar', val=np.random.rand(10)*1e2)
ivc.add_output('D_spar_p', val=np.random.rand(11)*1e2)
ivc.add_output('wt_spar', val=np.random.rand(10)*1e1)
ivc.add_output('L_spar', val=np.random.rand(10)*1e2)
ivc.add_output('l_stiff', val=np.random.rand(10)*1e2)
ivc.add_output('h_stiff', val=np.random.rand(10)*1e2)
ivc.add_output('t_f_stiff', val=np.random.rand(10)*1e2)
ivc.add_output('A_R', val=np.random.rand(10)*1e2)
ivc.add_output('r_f', val=np.random.rand(10)*1e2)
ivc.add_output('r_e', val=np.random.rand(10)*1e2)
ivc.add_output('tot_M_spar', val=np.random.rand()*1e6)

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check', SparCost(), promotes_inputs=['D_spar', 'D_spar_p', 'wt_spar', 'L_spar', 'l_stiff', 'h_stiff', 't_f_stiff', 'A_R', 'r_f', 'r_e', 'tot_M_spar'], promotes_outputs=['spar_cost'])

prob.setup(force_alloc_complex=True)
comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('stddev_tower_stress', 'resp_tower_stress', check_partials_data, binary=True)