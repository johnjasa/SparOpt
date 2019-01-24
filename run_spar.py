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

EC = {\
'N_EC' : 1, \
'ECfile' : 'prob_bins_test.dat'}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('D_spar_p', val=np.array([12., 12., 12., 12., 12., 12., 12., 12., 12., 8.3, 8.3]), units='m')
ivc.add_output('wt_spar_p', val=np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]), units='m')
ivc.add_output('L_spar', val=np.array([13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 8., 14.])*1.1, units='m')
ivc.add_output('D_tower_p', val=np.array([8.3, 8.02166998, 7.74333996, 7.46500994, 7.18667992, 6.9083499, 6.63001988, 6.35168986, 6.07335984, 5.79502982, 5.5])*1.5, units='m')
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
ivc.add_output('gamma_F_moor', val=1.65)
ivc.add_output('len_hor_moor', val=848.67*0.9, units='m')
ivc.add_output('len_tot_moor', val=902.2*1.1, units='m')
ivc.add_output('rho_wind', val=1.25, units='kg/m**3')
ivc.add_output('I_d', val=160234250.0, units='kg*m**2')
#ivc.add_output('windspeed_0', val=21., units='m/s')
#ivc.add_output('Hs', val=3., units='m')
#ivc.add_output('Tp', val=10., units='s')
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

ivc.add_output('angle_hull', val=0., units='rad')
ivc.add_output('buck_len', val=1.)
ivc.add_output('f_y', val=450., units='MPa')

ivc.add_output('gamma_M_tower', val=1.1)
ivc.add_output('gamma_F_tower', val=1.35)

ivc.add_output('gamma_F_hull', val=1.35)

ivc.add_output('DFF_tower', val=1.)
ivc.add_output('DFF_hull', val=3.)

ivc.add_output('maxval_surge', val=50., units='m')
ivc.add_output('maxval_pitch', val=15.*np.pi/180., units='rad')

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

from ECs import ECs
from condition_group import Condition
from total_tower_fatigue_damage import TotalTowerFatigueDamage
from total_hull_fatigue_damage import TotalHullFatigueDamage
from long_term_surge_cdf import LongTermSurgeCDF
from long_term_pitch_cdf import LongTermPitchCDF
from long_term_tower_stress_cdf import LongTermTowerStressCDF
from long_term_My_shell_buckling_cdf import LongTermMyShellBucklingCDF
from long_term_My_hoop_stress_cdf import LongTermMyHoopStressCDF
from long_term_My_mom_inertia_cdf import LongTermMyMomInertiaCDF
from long_term_fairlead_cdf import LongTermFairleadCDF
from long_term_moor_ten_cdf import LongTermMoorTenCDF
from return_period_surge import ReturnPeriodSurge
from return_period_pitch import ReturnPeriodPitch
from return_period_tower_stress import ReturnPeriodTowerStress
from return_period_My_shell_buckling import ReturnPeriodMyShellBuckling
from return_period_My_hoop_stress import ReturnPeriodMyHoopStress
from return_period_My_mom_inertia import ReturnPeriodMyMomInertia
from return_period_fairlead import ReturnPeriodFairlead
from return_period_moor_ten import ReturnPeriodMoorTen

prob.model.add_subsystem('ECs', ECs(EC=EC), promotes_outputs=['windspeed_0', 'Hs', 'Tp', 'p'])

parallel = prob.model.add_subsystem('parallel', ParallelGroup(), promotes_inputs=['D_spar_p', 'wt_spar_p', 'L_spar', 'D_tower_p', \
	'wt_tower_p', 'L_tower', 'rho_ball', 'wt_ball', 'M_nacelle', 'CoG_rotor', 'CoG_nacelle', 'I_rotor', 'M_rotor', 'water_depth', \
	'z_moor', 'D_moor', 'gamma_F_moor', 'len_hor_moor', 'len_tot_moor', 'rho_wind', 'I_d', 'k_p', 'k_i', \
	'k_t', 'omega_lowpass', 'omega_notch', 'bandwidth_notch', 'Cd', 'Cd_tower', 'struct_damp_ratio', 't_w_stiff', 't_f_stiff', 'h_stiff', \
	'b_stiff', 'l_stiff', 'angle_hull', 'buck_len', 'f_y', 'gamma_M_tower', 'gamma_F_tower', 'gamma_F_hull', 'maxval_surge', 'maxval_pitch'])

prob.model.add_subsystem('total_tower_fatigue_damage', TotalTowerFatigueDamage(EC=EC))
prob.model.add_subsystem('total_hull_fatigue_damage', TotalHullFatigueDamage(EC=EC))

prob.model.add_subsystem('long_term_surge_cdf', LongTermSurgeCDF(EC=EC))
prob.model.add_subsystem('long_term_pitch_cdf', LongTermPitchCDF(EC=EC))
prob.model.add_subsystem('long_term_tower_stress_cdf', LongTermTowerStressCDF(EC=EC))
prob.model.add_subsystem('long_term_My_shell_buckling_cdf', LongTermMyShellBucklingCDF(EC=EC))
prob.model.add_subsystem('long_term_My_hoop_stress_cdf', LongTermMyHoopStressCDF(EC=EC))
prob.model.add_subsystem('long_term_My_mom_inertia_cdf', LongTermMyMomInertiaCDF(EC=EC))
prob.model.add_subsystem('long_term_fairlead_cdf', LongTermFairleadCDF(EC=EC))
prob.model.add_subsystem('long_term_moor_ten_cdf', LongTermMoorTenCDF(EC=EC))

prob.model.add_subsystem('return_period_surge', ReturnPeriodSurge())
prob.model.add_subsystem('return_period_pitch', ReturnPeriodPitch())
prob.model.add_subsystem('return_period_tower_stress', ReturnPeriodTowerStress())
prob.model.add_subsystem('return_period_My_shell_buckling', ReturnPeriodMyShellBuckling())
prob.model.add_subsystem('return_period_My_hoop_stress', ReturnPeriodMyHoopStress())
prob.model.add_subsystem('return_period_My_mom_inertia', ReturnPeriodMyMomInertia())
prob.model.add_subsystem('return_period_fairlead', ReturnPeriodFairlead())
prob.model.add_subsystem('return_period_moor_ten', ReturnPeriodMoorTen())

for i in xrange(EC['N_EC']):
	parallel.add_subsystem('cond%d' % i, Condition(blades=blades, freqs=freqs), promotes_inputs=['D_spar_p', 'wt_spar_p', 'L_spar', 'D_tower_p', \
	'wt_tower_p', 'L_tower', 'rho_ball', 'wt_ball', 'M_nacelle', 'CoG_rotor', 'CoG_nacelle', 'I_rotor', 'M_rotor', 'water_depth', \
	'z_moor', 'D_moor', 'gamma_F_moor', 'len_hor_moor', 'len_tot_moor', 'rho_wind', 'I_d', 'k_p', 'k_i', \
	'k_t', 'omega_lowpass', 'omega_notch', 'bandwidth_notch', 'Cd', 'Cd_tower', 'struct_damp_ratio', 't_w_stiff', 't_f_stiff', 'h_stiff', \
	'b_stiff', 'l_stiff', 'angle_hull', 'buck_len', 'f_y', 'gamma_M_tower', 'gamma_F_tower', 'gamma_F_hull', 'maxval_surge', 'maxval_pitch'])

	prob.model.connect('windspeed_0', 'parallel.cond%d.windspeed_0' % i, src_indices=[i])
	prob.model.connect('Hs', 'parallel.cond%d.Hs' % i, src_indices=[i])
	prob.model.connect('Tp', 'parallel.cond%d.Tp' % i, src_indices=[i])
	
	prob.model.connect('parallel.cond%d.tower_fatigue_damage' % i, 'total_tower_fatigue_damage.tower_fatigue_damage%d' % i)
	prob.model.connect('p', 'total_tower_fatigue_damage.p%d' % i, src_indices=[i])

	prob.model.connect('parallel.cond%d.hull_fatigue_damage' % i, 'total_hull_fatigue_damage.hull_fatigue_damage%d' % i)
	prob.model.connect('p', 'total_hull_fatigue_damage.p%d' % i, src_indices=[i])
	
	prob.model.connect('parallel.cond%d.short_term_surge_CDF' % i, 'long_term_surge_cdf.short_term_surge_CDF%d' % i)
	prob.model.connect('p', 'long_term_surge_cdf.p%d' % i, src_indices=[i])

	prob.model.connect('parallel.cond%d.short_term_pitch_CDF' % i, 'long_term_pitch_cdf.short_term_pitch_CDF%d' % i)
	prob.model.connect('p', 'long_term_pitch_cdf.p%d' % i, src_indices=[i])

	prob.model.connect('parallel.cond%d.short_term_tower_stress_CDF' % i, 'long_term_tower_stress_cdf.short_term_tower_stress_CDF%d' % i)
	prob.model.connect('p', 'long_term_tower_stress_cdf.p%d' % i, src_indices=[i])

	prob.model.connect('parallel.cond%d.short_term_My_shell_buckling_CDF' % i, 'long_term_My_shell_buckling_cdf.short_term_My_shell_buckling_CDF%d' % i)
	prob.model.connect('p', 'long_term_My_shell_buckling_cdf.p%d' % i, src_indices=[i])

	prob.model.connect('parallel.cond%d.short_term_My_hoop_stress_CDF' % i, 'long_term_My_hoop_stress_cdf.short_term_My_hoop_stress_CDF%d' % i)
	prob.model.connect('p', 'long_term_My_hoop_stress_cdf.p%d' % i, src_indices=[i])

	prob.model.connect('parallel.cond%d.short_term_My_mom_inertia_CDF' % i, 'long_term_My_mom_inertia_cdf.short_term_My_mom_inertia_CDF%d' % i)
	prob.model.connect('p', 'long_term_My_mom_inertia_cdf.p%d' % i, src_indices=[i])

	prob.model.connect('parallel.cond%d.short_term_fairlead_CDF' % i, 'long_term_fairlead_cdf.short_term_fairlead_CDF%d' % i)
	prob.model.connect('p', 'long_term_fairlead_cdf.p%d' % i, src_indices=[i])

	prob.model.connect('parallel.cond%d.short_term_moor_ten_CDF' % i, 'long_term_moor_ten_cdf.short_term_moor_ten_CDF%d' % i)
	prob.model.connect('p', 'long_term_moor_ten_cdf.p%d' % i, src_indices=[i])

prob.model.connect('DFF_hull', 'total_hull_fatigue_damage.DFF_hull')

prob.model.connect('DFF_tower', 'total_tower_fatigue_damage.DFF_tower')

prob.model.connect('long_term_surge_cdf.long_term_surge_CDF', 'return_period_surge.long_term_surge_CDF')

prob.model.connect('long_term_pitch_cdf.long_term_pitch_CDF', 'return_period_pitch.long_term_pitch_CDF')

prob.model.connect('long_term_tower_stress_cdf.long_term_tower_stress_CDF', 'return_period_tower_stress.long_term_tower_stress_CDF')

prob.model.connect('long_term_My_shell_buckling_cdf.long_term_My_shell_buckling_CDF', 'return_period_My_shell_buckling.long_term_My_shell_buckling_CDF')

prob.model.connect('long_term_My_hoop_stress_cdf.long_term_My_hoop_stress_CDF', 'return_period_My_hoop_stress.long_term_My_hoop_stress_CDF')

prob.model.connect('long_term_My_mom_inertia_cdf.long_term_My_mom_inertia_CDF', 'return_period_My_mom_inertia.long_term_My_mom_inertia_CDF')

prob.model.connect('long_term_fairlead_cdf.long_term_fairlead_CDF', 'return_period_fairlead.long_term_fairlead_CDF')

prob.model.connect('long_term_moor_ten_cdf.long_term_moor_ten_CDF', 'return_period_moor_ten.long_term_moor_ten_CDF')

prob.model.linear_solver = LinearRunOnce()


#from openmdao.api import ScipyOptimizeDriver
from openmdao.api import pyOptSparseDriver
#driver = prob.driver = ScipyOptimizeDriver()
driver = prob.driver = pyOptSparseDriver()
driver.options['optimizer'] = 'SNOPT'

driver.recording_options['includes'] = []
driver.recording_options['record_objectives'] = True
driver.recording_options['record_constraints'] = True
driver.recording_options['record_desvars'] = True

recorder = SqliteRecorder("cases.sql")
driver.add_recorder(recorder)

prob.model.add_design_var('D_spar_p', lower=np.ones(11), upper=30.*np.ones(11))
prob.model.add_design_var('L_spar', lower=np.array([3., 3., 3., 3., 3., 3., 3., 3., 3., 10.]), upper=30.*np.ones(10))
prob.model.add_design_var('D_tower_p', lower=np.ones(11), upper=30.*np.ones(11))
prob.model.add_design_var('wt_tower_p', lower=0.005*np.ones(11), upper=0.5*np.ones(11))

prob.model.add_design_var('z_moor', lower=-320., upper=0.)
prob.model.add_design_var('D_moor', lower=0.01, upper=1.)
prob.model.add_design_var('len_hor_moor', lower=1., upper=3000.)
prob.model.add_design_var('len_tot_moor', lower=320., upper=4000.)

prob.model.add_constraint('total_tower_fatigue_damage.total_tower_fatigue_damage', upper=np.ones(11))
#prob.model.add_constraint('total_hull_fatigue_damage.total_hull_fatigue_damage', upper=np.ones(10))

prob.model.add_constraint('parallel.cond0.substructure.buoy_mass', lower=0.)
prob.model.add_constraint('parallel.cond0.substructure.lower_bound_z_moor', lower=0.)

prob.model.add_constraint('return_period_surge.T_surge', lower=50.)
prob.model.add_constraint('return_period_pitch.T_pitch', lower=50.)
prob.model.add_constraint('return_period_tower_stress.T_tower_stress', lower=50.*np.ones(10))
#prob.model.add_constraint('return_period_My_shell_buckling.T_My_shell_buckling', lower=50.*np.ones(10))
#prob.model.add_constraint('return_period_My_hoop_stress.T_My_hoop_stress', lower=50.*np.ones(10))
#prob.model.add_constraint('return_period_My_mom_inertia.T_My_mom_inertia', lower=50.*np.ones(10))
prob.model.add_constraint('return_period_fairlead.T_fairlead', lower=50.)
prob.model.add_constraint('return_period_moor_ten.T_moor_ten', lower=50.)

prob.model.add_objective('parallel.cond0.total_cost')

prob.setup()
#prob.set_solver_print(0)
prob.run_driver()

prob.cleanup()

"""
prob.setup()

prob.run_model()
"""
#print prob['total_tower_fatigue_damage.total_tower_fatigue_damage']
#print prob['return_period_surge.T_surge']
#print prob['return_period_pitch.T_pitch']
#print prob['return_period_tower_stress.T_tower_stress']
#print prob['return_period_fairlead.T_fairlead']
#print prob['return_period_moor_ten.T_moor_ten']

#print prob['stddev_surge']
#print prob['stddev_pitch']
#print prob['stddev_bend']
#print prob['stddev_rotspeed']
#print prob['parallel.cond0.tower_fatigue_damage'][0]
#print prob['parallel.cond1.tower_fatigue_damage'][0]
#print prob['total_tower_fatigue_damage.total_tower_fatigue_damage']
#print prob['total_hull_fatigue_damage.total_hull_fatigue_damage']
#print prob['parallel.cond0.total_cost']