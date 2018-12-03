import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from tower_stress_spectrum import TowerStressSpectrum

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,50), \
'omega_wave': np.linspace(0.12,6.28,50)}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('resp_tower_moment', val=np.random.rand(50,11))
ivc.add_output('D_tower_p', val=np.random.rand(11))
ivc.add_output('wt_tower_p', val=np.random.rand(11))

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check', TowerStressSpectrum(freqs=freqs), promotes_inputs=['resp_tower_moment', 'D_tower_p', 'wt_tower_p'], promotes_outputs=['resp_tower_stress'])

prob.setup(force_alloc_complex=True)
#comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('Re_RAO_Mwind_tower_moment', 'Re_RAO_Mwind_vel_surge', check_partials_data, binary=True)