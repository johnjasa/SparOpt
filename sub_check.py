import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, ExecComp, BalanceComp, DirectSolver, NonlinearBlockGS, NonlinearRunOnce, LinearRunOnce, BroydenSolver
from openmdao.utils.visualization import partial_deriv_plot
"""
from A_contrl_notch import Acontrl
from B_contrl_nf_notch import Bcontrl
from C_contrl_notch import Ccontrl

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,50), \
'omega_wave': np.linspace(0.12,6.28,50)}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('omega_lowpass', val=60.)
ivc.add_output('omega_notch', val=4.)
ivc.add_output('bandwidth_notch', val=0.1)
ivc.add_output('windspeed_0', val=15.)
ivc.add_output('rotspeed_0', val=1.)
ivc.add_output('k_i', val=0.)
ivc.add_output('k_p', val=1.)
ivc.add_output('k_t', val=-0.)
ivc.add_output('gain_corr_factor', val=1.)

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

prob.model.add_subsystem('A_contrl', Acontrl(), promotes_inputs=['omega_lowpass', 'omega_notch', 'bandwidth_notch'], promotes_outputs=['A_contrl'])

prob.model.add_subsystem('B_contrl', Bcontrl(), promotes_inputs=['omega_lowpass', 'k_t'], promotes_outputs=['B_contrl'])

prob.model.add_subsystem('C_contrl', Ccontrl(), promotes_inputs=['windspeed_0', 'rotspeed_0', 'k_i', 'k_p', 'gain_corr_factor'], promotes_outputs=['C_contrl'])

prob.setup()

prob.run_model()

import control

A = prob['A_contrl']
B = prob['B_contrl']
C = prob['C_contrl']
D = np.zeros((2,2))

TF = control.ss2tf(A,B,C,D)

omega = np.linspace(0.1,8.,1000)

mag, phase, omega = control.freqresp(TF,omega)

plt.plot(omega,mag[1,0])
plt.ylim(0,1.1)
plt.show()
"""

from mean_tower_drag import MeanTowerDrag

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('D_tower', val=np.array([8.3, 8.02166998, 7.74333996, 7.46500994, 7.18667992, 6.9083499, 6.63001988, 6.35168986, 6.07335984, 5.79502982]))
ivc.add_output('Z_tower', val=np.linspace(10.,119.,11))
ivc.add_output('L_tower', val=np.array([10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 11.13]))
ivc.add_output('windspeed_0', val=36., units='m/s')
ivc.add_output('Cd_tower', val=0.7)
ivc.add_output('CoG_rotor', val=119.)
ivc.add_output('rho_wind', val=1.25)

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

prob.model.add_subsystem('sub', MeanTowerDrag(), promotes_inputs=['D_tower', 'Z_tower', 'L_tower', 'windspeed_0', 'Cd_tower', 'CoG_rotor', 'rho_wind'], \
	promotes_outputs=['F0_tower_drag', 'Z0_tower_drag'])

prob.setup()

prob.run_model()

print prob['F0_tower_drag']
print prob['Z0_tower_drag']