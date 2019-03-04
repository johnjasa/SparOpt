import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver, KSComp
from openmdao.utils.visualization import partial_deriv_plot

from dyn_tower_drag import DynTowerDrag

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,70), \
'omega_wave': np.linspace(0.12,6.28,50)}

EC = {\
'N_EC' : 1}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('D_tower', val=np.random.rand(10)*1e2)
ivc.add_output('Z_tower', val=np.linspace(-120.,10.,11))
ivc.add_output('L_tower', val=np.random.rand(10)*1e2)
ivc.add_output('windspeed_0', val=50.)
ivc.add_output('Cd_tower', val=0.7)
ivc.add_output('CoG_rotor', val=119.)
ivc.add_output('rho_wind', val=1.25)

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check', DynTowerDrag(), promotes_inputs=['D_tower', 'Z_tower', 'L_tower', 'windspeed_0', 'Cd_tower', 'CoG_rotor', 'rho_wind'], promotes_outputs=['Fdyn_tower_drag', 'Mdyn_tower_drag'])

prob.setup(force_alloc_complex=True)
comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('stddev_tower_stress', 'resp_tower_stress', check_partials_data, binary=True)