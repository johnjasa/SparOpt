import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from long_term_moor_ten_cdf import LongTermMoorTenCDF

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,50), \
'omega_wind': np.linspace(0.12,6.28,50)}

EC = {\
'N_EC' : 2}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('short_term_moor_ten_CDF0', val=np.random.rand())
ivc.add_output('short_term_moor_ten_CDF1', val=np.random.rand())
ivc.add_output('p0', val=np.random.rand())
ivc.add_output('p1', val=np.random.rand())

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check',  LongTermMoorTenCDF(EC=EC), promotes_inputs=['short_term_moor_ten_CDF0', 'short_term_moor_ten_CDF1', 'p0', 'p1'], \
	promotes_outputs=['long_term_moor_ten_CDF'])

prob.setup(force_alloc_complex=True)
comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('hull_mom_acc_pitch', 'Z_spar', check_partials_data, binary=True)