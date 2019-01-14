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

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('D_spar_p', val=np.array([12., 12., 12., 12., 12., 12., 12., 12., 12., 8.3, 8.3]), units='m')
ivc.add_output('wt_spar_p', val=np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]), units='m')
ivc.add_output('Z_spar', val=np.array([-120., -108., -96., -84., -72., -60., -48., -36., -24., -12., 10.]), units='m')
ivc.add_output('M_spar', val=np.array([12e5, 12e5, 12e5, 12e5, 12e5, 12e5, 12e5, 12e5, 12e5, 8.3e5]), units='kg')
ivc.add_output('M_ball', val=1e6, units='kg')
ivc.add_output('L_ball', val=33., units='m')
ivc.add_output('spar_draft', val=120., units='m')
ivc.add_output('z_moor', val=-77.2, units='m')
ivc.add_output('M_moor', val=300000., units='kg')
ivc.add_output('angle_hull', val=0., units='rad')
ivc.add_output('dthrust_dv', val=1e5, units='N*s/m')
ivc.add_output('dmoment_dv', val=1e6, units='N*s')
ivc.add_output('l_stiff', val=1.0*np.ones(10), units='m')
ivc.add_output('t_f_stiff', val=0.02*np.ones(10), units='m')
ivc.add_output('h_stiff', val=0.6*np.ones(10), units='m')
ivc.add_output('t_w_stiff', val=0.02*np.ones(10), units='m')
ivc.add_output('b_stiff', val=0.8*np.ones(10), units='m')
ivc.add_output('buck_len', val=1.)
ivc.add_output('A_R', val=np.ones(10), units='m**2')
ivc.add_output('f_y', val=350., units='MPa')

#ivc.add_output('My_hull', val=np.array([-15.4e8, -15.4e8, -15.4e8, -15.4e8, -15.4e8, -15.4e8, -15.4e8, -15.4e8, -15.4e8, -8.4e8]), units='N*m')

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

bal = prob.model.add_subsystem('balance', BalanceComp())
bal.add_balance('My_hull', val=-np.ones(10), units='N*m')

from hull_buckling_group import HullBuckling

hull_buckling_group = HullBuckling()

prob.model.add_subsystem('hull_buckling', hull_buckling_group, promotes_inputs=['My_hull', 'D_spar_p', 'wt_spar_p', 'Z_spar', 'M_spar', 'M_ball', 'L_ball', 'spar_draft', 'M_moor', 'z_moor', 'angle_hull', \
	'dthrust_dv', 'dmoment_dv', 'l_stiff', 't_f_stiff', 'h_stiff', 'f_y', 't_w_stiff', 'b_stiff', 'buck_len', 'A_R'], promotes_outputs=['shell_buckling'])

prob.model.connect('balance.My_hull', 'My_hull')
prob.model.connect('shell_buckling', 'balance.lhs:My_hull')

prob.setup()

prob.model.hull_buckling.nonlinear_solver = NonlinearRunOnce()
prob.model.hull_buckling.linear_solver = LinearRunOnce()
#make group with balance and hull_buckling in run_spar
prob.model.linear_solver = DirectSolver()
#prob.model.nonlinear_solver = NonlinearBlockGS(maxiter=50, atol=1e-8, rtol=1e-8)
prob.model.nonlinear_solver = BroydenSolver(maxiter=50, atol=1e-8)

prob.run_model()

print prob['My_hull']
print prob['shell_buckling']