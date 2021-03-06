import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss

from openmdao.api import Problem, IndepVarComp, LinearRunOnce, DirectSolver, LinearBlockGS, ScipyKrylov, NonlinearBlockGS, NewtonSolver, SqliteRecorder, ParallelGroup, BroydenSolver, ExecComp, KSComp

blades = {\
'Rtip' : 89.165, \
'Rhub' : 2.8, \
'N_b_elem' : 20, \
'indfile' : 'DTU10MW_indfacs.dat', \
'bladefile' : 'DTU10MWblade.dat', \
'foilnames' : ['foil1', 'foil11', 'foil12', 'foil13', 'foil14', 'foil15', 'foil16', 'foil17', 'foil18', 'foil19'], \
'foilfolder' : 'Airfoils/', \
'windfolder' : 'Windspeeds/'}

freqs = {\
'omega' : np.linspace(0.001,4.5,450), \
'omega_wave': np.linspace(0.1,4.5,50)}

#EC = {\
#'N_EC' : 1, \
#'ECfile' : 'prob_bins_test.dat'}

EC_fat = {\
'N_EC' : 7, \
'ECfile' : 'prob_bins_fatigue_eq_control.dat'}

#EC_ext = {\
#'N_EC' : 3, \
#'ECfile' : 'prob_bins_extreme.dat'}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('D_spar_p', val=np.array([12., 12., 12., 12., 12., 12., 12., 12., 12., 8.3, 8.3]), units='m')
ivc.add_output('wt_spar_p', val=np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]), units='m')
ivc.add_output('L_spar', val=np.array([13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 8., 14.]), units='m')
ivc.add_output('D_tower_p', val=np.array([8.3, 8.02166998, 7.74333996, 7.46500994, 7.18667992, 6.9083499, 6.63001988, 6.35168986, 6.07335984, 5.79502982, 5.5])*1.3, units='m')
ivc.add_output('wt_tower_p', val=np.array([0.038, 0.038, 0.034, 0.034, 0.030, 0.030, 0.026, 0.026, 0.022, 0.022, 0.018])*1.5, units='m')
ivc.add_output('L_tower', val=np.array([10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 11.13]), units='m')
ivc.add_output('rho_ball', val=2600., units='kg/m**3')
ivc.add_output('wt_ball', val=0.06, units='m')
ivc.add_output('M_nacelle', val=4.46e5, units='kg')
ivc.add_output('CoG_rotor', val=119., units='m')
ivc.add_output('CoG_nacelle', val=118.08, units='m')
ivc.add_output('I_rotor', val=7.808e7, units='kg*m**2')
ivc.add_output('M_rotor', val=2.307e5, units='kg')
ivc.add_output('water_depth', val=320., units='m')

ivc.add_output('z_moor', val=-77.2, units='m')
ivc.add_output('D_moor', val=0.09, units='m')
ivc.add_output('gamma_F_moor', val=1.)
ivc.add_output('gamma_F_moor_mean', val=1.3)
ivc.add_output('gamma_F_moor_dyn', val=1.75)
ivc.add_output('len_hor_moor', val=838.67, units='m')
ivc.add_output('len_tot_moor', val=902.2, units='m')

ivc.add_output('rho_wind', val=1.25, units='kg/m**3')
ivc.add_output('I_d', val=160234250.0, units='kg*m**2')
ivc.add_output('k_p', val=0.1794, units='rad*s/rad')
ivc.add_output('k_i', val=0.0165, units='rad/rad')
ivc.add_output('k_t', val=-0., units='rad*s/m')
ivc.add_output('omega_lowpass', val=2.*np.pi/0.8, units='rad/s')
ivc.add_output('omega_notch', val=10.16, units='rad/s')
ivc.add_output('bandwidth_notch', val=0.1, units='rad/s')
ivc.add_output('Cd', val=0.7)
ivc.add_output('Cd_tower', val=0.7)
ivc.add_output('struct_damp_ratio', val=0.5*0.007*2.*np.pi/2.395)

ivc.add_output('t_w_stiff', val=0.02*np.ones(10), units='m')
ivc.add_output('t_f_stiff', val=0.02*np.ones(10), units='m')
ivc.add_output('h_stiff', val=0.6*np.ones(10), units='m')
ivc.add_output('b_stiff', val=0.8*np.ones(10), units='m')
ivc.add_output('l_stiff', val=1.0*np.ones(10), units='m')

ivc.add_output('f_y', val=355., units='MPa')

ivc.add_output('gamma_M_tower', val=1.1)
ivc.add_output('gamma_F_tower', val=1.35)

ivc.add_output('DFF_tower', val=2.)

ivc.add_output('maxval_surge', val=32., units='m')
ivc.add_output('maxval_pitch', val=15.*np.pi/180., units='rad')

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

from ECs_fat import ECsFat
#from ECs_ext import ECsExt
from condition_group_reduced_fat import ConditionFat
#from condition_group_reduced_ext import ConditionExt
from total_std_dev_rotspeed import TotalStdDevRotspeed

prob.model.add_subsystem('ECs_fat', ECsFat(EC=EC_fat), promotes_outputs=['windspeed_0', 'Hs', 'Tp', 'p'])

#prob.model.add_subsystem('ECs_ext', ECsExt(EC=EC_ext), promotes_outputs=['windspeed_0_ext', 'Hs_ext', 'Tp_ext'])

parallel_fat = prob.model.add_subsystem('parallel_fat', ParallelGroup(), promotes_inputs=['D_spar_p', 'wt_spar_p', 'L_spar', 'D_tower_p', \
	'wt_tower_p', 'L_tower', 'rho_ball', 'wt_ball', 'M_nacelle', 'CoG_rotor', 'CoG_nacelle', 'I_rotor', 'M_rotor', 'water_depth', \
	'z_moor', 'D_moor', 'gamma_F_moor', 'len_hor_moor', 'len_tot_moor', 'rho_wind', 'I_d', 'k_p', 'k_i', \
	'k_t', 'omega_lowpass', 'omega_notch', 'bandwidth_notch', 'Cd', 'Cd_tower', 'struct_damp_ratio', \
	't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'l_stiff'])

prob.model.add_subsystem('total_std_dev_rotspeed', TotalStdDevRotspeed(EC=EC_fat))

for i in xrange(EC_fat['N_EC']):
	parallel_fat.add_subsystem('cond%d_fat' % i, ConditionFat(blades=blades, freqs=freqs), promotes_inputs=['D_spar_p', 'wt_spar_p', 'L_spar', 'D_tower_p', \
	'wt_tower_p', 'L_tower', 'rho_ball', 'wt_ball', 'M_nacelle', 'CoG_rotor', 'CoG_nacelle', 'I_rotor', 'M_rotor', 'water_depth', \
	'z_moor', 'D_moor', 'gamma_F_moor', 'len_hor_moor', 'len_tot_moor', 'rho_wind', 'I_d', 'k_p', 'k_i', \
	'k_t', 'omega_lowpass', 'omega_notch', 'bandwidth_notch', 'Cd', 'Cd_tower', 'struct_damp_ratio', \
	't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'l_stiff'])

	prob.model.connect('windspeed_0', 'parallel_fat.cond%d_fat.windspeed_0' % i, src_indices=[i])
	prob.model.connect('Hs', 'parallel_fat.cond%d_fat.Hs' % i, src_indices=[i])
	prob.model.connect('Tp', 'parallel_fat.cond%d_fat.Tp' % i, src_indices=[i])

	prob.model.connect('parallel_fat.cond%d_fat.stddev_rotspeed' % i, 'total_std_dev_rotspeed.stddev_rotspeed%d' % i)
	prob.model.connect('p', 'total_std_dev_rotspeed.p%d' % i, src_indices=[i])

prob.model.linear_solver = LinearRunOnce()

#from openmdao.api import ScipyOptimizeDriver
from openmdao.api import pyOptSparseDriver
#driver = prob.driver = ScipyOptimizeDriver()
driver = prob.driver = pyOptSparseDriver()
driver.options['optimizer'] = 'SNOPT'
driver.opt_settings['Major feasibility tolerance'] = 1e-3
driver.opt_settings['Major optimality tolerance'] = 1e-3

driver.recording_options['includes'] = []
driver.recording_options['record_objectives'] = True
driver.recording_options['record_constraints'] = True
driver.recording_options['record_desvars'] = True

recorder = SqliteRecorder("control.sql")
driver.add_recorder(recorder)

prob.model.add_design_var('k_p', lower=0., upper=5.)
prob.model.add_design_var('k_i', lower=0., upper=5.)

prob.model.add_subsystem('ks_poles0', KSComp(width=11, rho=10000.))
prob.model.add_subsystem('ks_poles1', KSComp(width=11, rho=10000.))
prob.model.add_subsystem('ks_poles2', KSComp(width=11, rho=10000.))
prob.model.add_subsystem('ks_poles3', KSComp(width=11, rho=10000.))
prob.model.add_subsystem('ks_poles4', KSComp(width=11, rho=10000.))
prob.model.add_subsystem('ks_poles5', KSComp(width=11, rho=10000.))
prob.model.add_subsystem('ks_poles6', KSComp(width=11, rho=10000.))
prob.model.connect('parallel_fat.cond0_fat.poles', 'ks_poles0.g')
prob.model.connect('parallel_fat.cond1_fat.poles', 'ks_poles1.g')
prob.model.connect('parallel_fat.cond2_fat.poles', 'ks_poles2.g')
prob.model.connect('parallel_fat.cond3_fat.poles', 'ks_poles3.g')
prob.model.connect('parallel_fat.cond4_fat.poles', 'ks_poles4.g')
prob.model.connect('parallel_fat.cond5_fat.poles', 'ks_poles5.g')
prob.model.connect('parallel_fat.cond6_fat.poles', 'ks_poles6.g')
prob.model.add_constraint('ks_poles0.KS', upper=0.)
prob.model.add_constraint('ks_poles1.KS', upper=0.)
prob.model.add_constraint('ks_poles2.KS', upper=0.)
prob.model.add_constraint('ks_poles3.KS', upper=0.)
prob.model.add_constraint('ks_poles4.KS', upper=0.)
prob.model.add_constraint('ks_poles5.KS', upper=0.)
prob.model.add_constraint('ks_poles6.KS', upper=0.)

prob.model.add_objective('total_std_dev_rotspeed.total_stddev_rotspeed')

prob.setup()

prob.run_driver()
