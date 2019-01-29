import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from norm_resp_wind_bldpitch import NormRespWindBldpitch

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,50), \
'omega_wave': np.linspace(0.12,6.28,50)}

EC = {\
'N_EC' : 1}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('thrust_wind', val=np.random.rand(50))
ivc.add_output('torque_wind', val=np.random.rand(50))
ivc.add_output('Re_H_feedbk', val=np.random.rand(50,11,6))
ivc.add_output('Im_H_feedbk', val=np.random.rand(50,11,6))
ivc.add_output('k_i', val=np.random.rand())
ivc.add_output('k_p', val=np.random.rand())
ivc.add_output('gain_corr_factor', val=np.random.rand())
ivc.add_output('windspeed_0', val=3.)

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check',  NormRespWindBldpitch(freqs=freqs), promotes_inputs=['thrust_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'gain_corr_factor', 'windspeed_0'], \
	promotes_outputs=['Re_RAO_wind_bldpitch', 'Im_RAO_wind_bldpitch'])

prob.setup(force_alloc_complex=True)
#comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('hull_mom_acc_bend', 'Z_spar', check_partials_data, binary=True)