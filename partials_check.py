import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from prob_max_moor_ten import ProbMaxMoorTen

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,50), \
'omega_wave': np.linspace(0.12,6.28,50)}

EC = {\
'N_EC' : 2}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('v_z_moor_ten', val=np.random.rand())
ivc.add_output('mean_moor_ten', val=np.random.rand())
ivc.add_output('stddev_moor_ten', val=np.random.rand())
ivc.add_output('gamma_F_moor_mean', val=np.random.rand())
ivc.add_output('gamma_F_moor_dyn', val=np.random.rand())

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check',  ProbMaxMoorTen(), promotes_inputs=['v_z_moor_ten', 'mean_moor_ten', 'stddev_moor_ten', 'gamma_F_moor_mean', 'gamma_F_moor_dyn'], \
	promotes_outputs=['prob_max_moor_ten'])

prob.setup(force_alloc_complex=True)
comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('hull_mom_acc_bend', 'Z_spar', check_partials_data, binary=True)