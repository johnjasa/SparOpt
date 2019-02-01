import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss

from openmdao.api import Problem, IndepVarComp, LinearRunOnce, DirectSolver, LinearBlockGS, ScipyKrylov, NonlinearBlockGS, NewtonSolver, SqliteRecorder, ParallelGroup, BroydenSolver, ExecComp

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
'omega' : np.linspace(0.014361566416410483,6.283185307179586,1000), \
'omega_wave': np.linspace(0.1,6.28,100)}

#EC = {\
#'N_EC' : 1, \
#'ECfile' : 'prob_bins_test.dat'}

EC_fat = {\
'N_EC' : 1, \
'ECfile' : 'prob_bins_fatigue.dat'}

EC_ext = {\
'N_EC' : 3, \
'ECfile' : 'prob_bins_extreme.dat'}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('D_spar_p', val=np.array([12., 12., 12., 12., 12., 12., 12., 12., 12., 8.3, 8.3]), units='m')
ivc.add_output('wt_spar_p', val=np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]), units='m')
ivc.add_output('L_spar', val=np.array([13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 8., 14.]), units='m')
ivc.add_output('D_tower_p', val=np.array([8.3, 8.02166998, 7.74333996, 7.46500994, 7.18667992, 6.9083499, 6.63001988, 6.35168986, 6.07335984, 5.79502982, 5.5]), units='m')
ivc.add_output('wt_tower_p', val=np.array([0.038, 0.038, 0.034, 0.034, 0.030, 0.030, 0.026, 0.026, 0.022, 0.022, 0.018]), units='m')
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
ivc.add_output('len_hor_moor', val=848.67, units='m')
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
ivc.add_output('Cd_tower', val=0.01)
ivc.add_output('struct_damp_ratio', val=0.5*0.007*2.*np.pi/2.395)

ivc.add_output('t_w_stiff', val=0.02*np.ones(10), units='m')
ivc.add_output('t_f_stiff', val=0.02*np.ones(10), units='m')
ivc.add_output('h_stiff', val=0.6*np.ones(10), units='m')
ivc.add_output('b_stiff', val=0.8*np.ones(10), units='m')
ivc.add_output('l_stiff', val=1.0*np.ones(10), units='m')

ivc.add_output('f_y', val=450., units='MPa')

ivc.add_output('gamma_M_tower', val=1.1)
ivc.add_output('gamma_F_tower', val=1.35)

ivc.add_output('DFF_tower', val=1.)

ivc.add_output('maxval_surge', val=50., units='m')
ivc.add_output('maxval_pitch', val=15.*np.pi/180., units='rad')

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

#from ECs_fat import ECsFat
from ECs_ext import ECsExt
#from condition_group_reduced_fat import ConditionFat
from condition_group_reduced_ext import ConditionExt
#from total_tower_fatigue_damage import TotalTowerFatigueDamage

#prob.model.add_subsystem('ECs_fat', ECsFat(EC=EC_fat), promotes_outputs=['windspeed_0', 'Hs', 'Tp', 'p'])

prob.model.add_subsystem('ECs_ext', ECsExt(EC=EC_ext), promotes_outputs=['windspeed_0_ext', 'Hs_ext', 'Tp_ext'])
"""
parallel_fat = prob.model.add_subsystem('parallel_fat', ParallelGroup(), promotes_inputs=['D_spar_p', 'wt_spar_p', 'L_spar', 'D_tower_p', \
	'wt_tower_p', 'L_tower', 'rho_ball', 'wt_ball', 'M_nacelle', 'CoG_rotor', 'CoG_nacelle', 'I_rotor', 'M_rotor', 'water_depth', \
	'z_moor', 'D_moor', 'gamma_F_moor', 'len_hor_moor', 'len_tot_moor', 'rho_wind', 'I_d', 'k_p', 'k_i', \
	'k_t', 'omega_lowpass', 'omega_notch', 'bandwidth_notch', 'Cd', 'Cd_tower', 'struct_damp_ratio', \
	't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'l_stiff'])
"""
parallel_ext = prob.model.add_subsystem('parallel_ext', ParallelGroup(), promotes_inputs=['D_spar_p', 'wt_spar_p', 'L_spar', 'D_tower_p', \
	'wt_tower_p', 'L_tower', 'rho_ball', 'wt_ball', 'M_nacelle', 'CoG_rotor', 'CoG_nacelle', 'I_rotor', 'M_rotor', 'water_depth', \
	'z_moor', 'D_moor', 'gamma_F_moor', 'gamma_F_moor_mean', 'gamma_F_moor_dyn', 'len_hor_moor', 'len_tot_moor', 'rho_wind', 'I_d', 'k_p', 'k_i', \
	'k_t', 'omega_lowpass', 'omega_notch', 'bandwidth_notch', 'Cd', 'Cd_tower', 'struct_damp_ratio', 'f_y', 'gamma_M_tower', 'gamma_F_tower', \
	'maxval_surge', 'maxval_pitch', 't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'l_stiff'])
"""
prob.model.add_subsystem('total_tower_fatigue_damage', TotalTowerFatigueDamage(EC=EC_fat))

for i in xrange(EC_fat['N_EC']):
	parallel_fat.add_subsystem('cond%d_fat' % i, ConditionFat(blades=blades, freqs=freqs), promotes_inputs=['D_spar_p', 'wt_spar_p', 'L_spar', 'D_tower_p', \
	'wt_tower_p', 'L_tower', 'rho_ball', 'wt_ball', 'M_nacelle', 'CoG_rotor', 'CoG_nacelle', 'I_rotor', 'M_rotor', 'water_depth', \
	'z_moor', 'D_moor', 'gamma_F_moor', 'len_hor_moor', 'len_tot_moor', 'rho_wind', 'I_d', 'k_p', 'k_i', \
	'k_t', 'omega_lowpass', 'omega_notch', 'bandwidth_notch', 'Cd', 'Cd_tower', 'struct_damp_ratio', \
	't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'l_stiff'])

	prob.model.connect('windspeed_0', 'parallel_fat.cond%d_fat.windspeed_0' % i, src_indices=[i])
	prob.model.connect('Hs', 'parallel_fat.cond%d_fat.Hs' % i, src_indices=[i])
	prob.model.connect('Tp', 'parallel_fat.cond%d_fat.Tp' % i, src_indices=[i])
	
	prob.model.connect('parallel_fat.cond%d_fat.tower_fatigue_damage' % i, 'total_tower_fatigue_damage.tower_fatigue_damage%d' % i)
	prob.model.connect('p', 'total_tower_fatigue_damage.p%d' % i, src_indices=[i])

prob.model.connect('DFF_tower', 'total_tower_fatigue_damage.DFF_tower')
"""
for i in xrange(EC_ext['N_EC']):
	parallel_ext.add_subsystem('cond%d_ext' % i, ConditionExt(blades=blades, freqs=freqs), promotes_inputs=['D_spar_p', 'wt_spar_p', 'L_spar', 'D_tower_p', \
	'wt_tower_p', 'L_tower', 'rho_ball', 'wt_ball', 'M_nacelle', 'CoG_rotor', 'CoG_nacelle', 'I_rotor', 'M_rotor', 'water_depth', \
	'z_moor', 'D_moor', 'gamma_F_moor', 'gamma_F_moor_mean', 'gamma_F_moor_dyn', 'len_hor_moor', 'len_tot_moor', 'rho_wind', 'I_d', 'k_p', 'k_i', \
	'k_t', 'omega_lowpass', 'omega_notch', 'bandwidth_notch', 'Cd', 'Cd_tower', 'struct_damp_ratio', 'f_y', 'gamma_M_tower', 'gamma_F_tower', \
	'maxval_surge', 'maxval_pitch', 't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'l_stiff'])

	prob.model.connect('windspeed_0_ext', 'parallel_ext.cond%d_ext.windspeed_0' % i, src_indices=[i])
	prob.model.connect('Hs_ext', 'parallel_ext.cond%d_ext.Hs' % i, src_indices=[i])
	prob.model.connect('Tp_ext', 'parallel_ext.cond%d_ext.Tp' % i, src_indices=[i])

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

recorder = SqliteRecorder("cases.sql")
driver.add_recorder(recorder)

#prob.model.add_design_var('D_spar_p', lower=np.ones(11), upper=20.*np.ones(11))
#prob.model.add_design_var('L_spar', lower=np.array([3., 3., 3., 3., 3., 3., 3., 3., 3., 10.]), upper=30.*np.ones(10))

#prob.model.add_design_var('D_tower_p', lower=np.ones(11), upper=20.*np.ones(11))
#prob.model.add_design_var('wt_tower_p', lower=0.005*np.ones(11), upper=0.5*np.ones(11))

prob.model.add_design_var('D_moor', lower=0.01, upper=0.5)
prob.model.add_design_var('z_moor', lower=-320., upper=0.)
prob.model.add_design_var('len_hor_moor', lower=1., upper=3000.)
prob.model.add_design_var('len_tot_moor', lower=320., upper=4000.)

prob.model.add_constraint('parallel_ext.cond0_ext.substructure.buoy_mass', lower=0.)
prob.model.add_constraint('parallel_ext.cond0_ext.substructure.lower_bound_z_moor', lower=0.)

#prob.model.add_constraint('parallel_ext.cond0_ext.constr_50_surge', lower=0.)
#prob.model.add_constraint('parallel_ext.cond0_ext.constr_50_pitch', lower=0.)

#prob.model.add_constraint('total_tower_fatigue_damage.total_tower_fatigue_damage', upper=np.ones(11))
#prob.model.add_constraint('parallel_ext.cond0_ext.constr_50_tower_stress', lower=np.zeros(10))

prob.model.add_constraint('parallel_ext.cond0_ext.constr_50_surge', lower=0.)
prob.model.add_constraint('parallel_ext.cond0_ext.constr_50_pitch', lower=0.)
prob.model.add_constraint('parallel_ext.cond0_ext.constr_50_moor_ten', lower=0.)
prob.model.add_constraint('parallel_ext.cond0_ext.constr_50_fairlead', lower=0.)
prob.model.add_constraint('parallel_ext.cond1_ext.constr_50_surge', lower=0.)
prob.model.add_constraint('parallel_ext.cond1_ext.constr_50_pitch', lower=0.)
prob.model.add_constraint('parallel_ext.cond1_ext.constr_50_moor_ten', lower=0.)
prob.model.add_constraint('parallel_ext.cond1_ext.constr_50_fairlead', lower=0.)
prob.model.add_constraint('parallel_ext.cond2_ext.constr_50_surge', lower=0.)
prob.model.add_constraint('parallel_ext.cond2_ext.constr_50_pitch', lower=0.)
prob.model.add_constraint('parallel_ext.cond2_ext.constr_50_moor_ten', lower=0.)
prob.model.add_constraint('parallel_ext.cond2_ext.constr_50_fairlead', lower=0.)

prob.model.add_objective('parallel_ext.cond0_ext.mooring_cost')

prob.setup()
#prob.set_solver_print(0)
prob.run_driver()

prob.cleanup()

"""
prob.setup()

prob.run_model()
"""
#print prob['total_tower_fatigue_damage.total_tower_fatigue_damage'] #[0.84150843 0.71742536 0.78139559 0.63796774 0.67922156 0.51527836 0.51941003 0.34625994 0.33857489 0.19130584 0.21402161] [0.20113283 0.1636332  0.1687287  0.12909339 0.12698522 0.08711082 0.07655451 0.04145436 0.02799767 0.00684233 0.00026347]
#print prob['parallel_ext.cond0_ext.constr_50_surge'] #[1.12581472] [10.10193924]
#print prob['parallel_ext.cond0_ext.constr_50_pitch'] #[0.78097484] [7.06654839]
#print prob['parallel_ext.cond0_ext.constr_50_tower_stress'] #[0.95840193 1.12343779 0.91899427 1.11934134 0.90256966 1.16872072 0.94877398 1.36136119 1.18218993 2.03930487] [4.2187535  4.69211873 4.16811865 4.7250744  4.14057684 4.83814728 4.1935415  5.1713398  4.5062743  6.24099886]
#print prob['parallel_ext.cond0_ext.constr_50_moor_ten'] #[0.47595709] [1.29681327]
#print prob['parallel_ext.cond0_ext.constr_50_fairlead'] #[-0.22662359] [3.00152786]
#print prob['parallel_ext.cond0_ext.mooring_cost']

#print prob['stddev_surge']
#print prob['stddev_pitch']
#print prob['stddev_bend']
#print prob['stddev_rotspeed']
#print prob['parallel.cond0.tower_fatigue_damage'][0]
#print prob['parallel.cond1.tower_fatigue_damage'][0]
#print prob['total_tower_fatigue_damage.total_tower_fatigue_damage']
#print prob['total_hull_fatigue_damage.total_hull_fatigue_damage']
#print prob['parallel.cond0.total_cost']