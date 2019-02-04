import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from taper_hull import TaperHull

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,50), \
'omega_wave': np.linspace(0.12,6.28,50)}

EC = {\
'N_EC' : 1}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('D_spar_p', val=np.random.rand(11)*1e2)
ivc.add_output('L_spar', val=np.random.rand(10)*1e2)

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check', TaperHull(), promotes_inputs=['D_spar_p', 'L_spar'], promotes_outputs=['taper_angle_hull'])

prob.setup(force_alloc_complex=True)
comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('stddev_tower_stress', 'resp_tower_stress', check_partials_data, binary=True)