import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from transfer_function import TransferFunction

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,70), \
'omega_wave': np.linspace(0.12,6.28,50)}

EC = {\
'N_EC' : 1}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('A_feedbk', val=np.random.rand(11,11))
ivc.add_output('B_feedbk', val=np.random.rand(11,6))

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check', TransferFunction(freqs=freqs), promotes_inputs=['A_feedbk', 'B_feedbk'], promotes_outputs=['Re_H_feedbk', 'Im_H_feedbk'])

prob.setup(force_alloc_complex=True)
#comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('stddev_tower_stress', 'resp_tower_stress', check_partials_data, binary=True)
