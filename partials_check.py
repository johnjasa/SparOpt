import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from modeshape_elem_mass import ModeshapeElemMass

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,50), \
'omega_wave': np.linspace(0.12,6.28,50)}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('D_spar', val=np.random.rand(10))
ivc.add_output('L_spar', val=np.random.rand(10))
ivc.add_output('M_spar', val=np.random.rand(10))
ivc.add_output('Z_spar', val=np.linspace(-120,10,11))
ivc.add_output('L_tower', val=np.random.rand(10))
ivc.add_output('M_tower', val=np.random.rand(10))
ivc.add_output('spar_draft', val=120.)
ivc.add_output('M_ball_elem', val=np.random.rand(10))
ivc.add_output('L_ball_elem', val=np.random.rand(10))
ivc.add_output('L_ball', val=34.)
ivc.add_output('z_sparnode', val=np.linspace(-120,11,14))
ivc.add_output('L_mode_elem', val=np.random.rand(23))

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check', ModeshapeElemMass(), promotes_inputs=['D_spar', 'L_spar', 'M_spar', 'Z_spar', 'L_tower', 'M_tower', 'spar_draft', 'M_ball_elem', 'L_ball_elem', 'L_ball', 'z_sparnode', 'L_mode_elem'], promotes_outputs=['mel'])

prob.setup(force_alloc_complex=True)
#comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('Re_RAO_Mwind_tower_moment', 'Re_RAO_Mwind_vel_surge', check_partials_data, binary=True)