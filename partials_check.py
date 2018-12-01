import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from hull_I_xh import HullIXh

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,10), \
'omega_wave': np.linspace(0.12,6.28,50)}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('tau', val=np.random.rand(10))
ivc.add_output('r_0', val=np.random.rand(10))
ivc.add_output('spar_draft', val=np.random.rand())
ivc.add_output('wt_spar_p', val=np.random.rand(11))
ivc.add_output('l_stiff', val=np.random.rand(10))

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check', HullIXh(), promotes_inputs=['tau', 'r_0', 'spar_draft', 'wt_spar_p', 'l_stiff'], promotes_outputs=['I_xh'])

prob.setup(force_alloc_complex=True)
#comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('Re_H_feedbk','B_feedbk', check_partials_data, binary=True)
