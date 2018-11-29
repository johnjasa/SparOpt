import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from bending_damping import BendingDamping

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,10), \
'omega_wave': np.linspace(0.12,6.28,50)}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('x_dd_sparelem', val=np.random.rand(13))
ivc.add_output('x_dd_towerelem', val=np.random.rand(10))
ivc.add_output('EI_mode_elem', val=np.random.rand(23))
ivc.add_output('z_sparnode', val=np.linspace(-120,10,14))
ivc.add_output('z_towernode', val=np.linspace(10,119,11))
ivc.add_output('alpha_damp', val=0.007)

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check', BendingDamping(), promotes_inputs=['x_dd_sparelem', 'x_dd_towerelem', 'EI_mode_elem', 'z_sparnode', 'z_towernode', 'alpha_damp'], promotes_outputs=['B_struct_77'])

prob.setup(force_alloc_complex=True)
comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('Re_H_feedbk','B_feedbk', check_partials_data, binary=True)
