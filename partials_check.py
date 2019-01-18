import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from mean_pitch import MeanPitch

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,50), \
'omega_wind': np.linspace(0.12,6.28,50)}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('thrust_0', val=np.random.rand())
ivc.add_output('F0_tower_drag', val=np.random.rand())
ivc.add_output('Z0_tower_drag', val=np.random.rand())
ivc.add_output('CoG_rotor', val=np.random.rand())
ivc.add_output('z_moor', val=np.random.rand())
ivc.add_output('buoy_spar', val=np.random.rand())
ivc.add_output('CoB', val=np.random.rand())
ivc.add_output('M_turb', val=np.random.rand())
ivc.add_output('tot_M_spar', val=np.random.rand())
ivc.add_output('M_ball', val=np.random.rand())
ivc.add_output('CoG_total', val=np.random.rand())

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check',  MeanPitch(), promotes_inputs=['thrust_0', 'F0_tower_drag', 'Z0_tower_drag', 'CoG_rotor', 'z_moor', 'buoy_spar', 'CoB', 'M_turb', 'tot_M_spar', 'M_ball', 'CoG_total'], \
	promotes_outputs=['mean_pitch'])

prob.setup(force_alloc_complex=True)
comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('hull_mom_acc_pitch', 'Z_spar', check_partials_data, binary=True)