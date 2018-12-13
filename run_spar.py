import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss

from openmdao.api import Problem, IndepVarComp, LinearRunOnce, DirectSolver, LinearBlockGS, ScipyKrylov, NonlinearBlockGS, NewtonSolver, SqliteRecorder, ParallelGroup, ExecComp

blades = {\
'Rtip' : 89.165, \
'Rhub' : 2.8, \
'N_b_elem' : 20, \
'indfile' : 'M:/PhD/Integrated optimization/Aerodynamics/BEM/BEMpy/DTU10MW_indfacs.dat', \
'bladefile' : 'M:/PhD/Integrated optimization/Aerodynamics/BEM/BEMpy/DTU10MWblade.dat', \
'foilnames' : ['foil1', 'foil11', 'foil12', 'foil13', 'foil14', 'foil15', 'foil16', 'foil17', 'foil18', 'foil19'], \
'foilfolder' : 'M:/PhD/Integrated optimization/Aerodynamics/BEM/BEMpy/Airfoils/', \
'cohfolder' : 'M:/PhD/Integrated optimization/Aerodynamics/CoherenceCoeffs/'}

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,1000), \
'omega_wave': np.linspace(0.1,6.28,100)}

EC = {\
'N_EC' : 2, \
'ECfile' : 'C:/Code/prob_bins_test.dat'}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('D_spar_p', val=np.array([12., 12., 12., 12., 12., 12., 12., 12., 12., 8.3, 8.3]), units='m')
ivc.add_output('wt_spar_p', val=np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]), units='m')
#ivc.add_output('D_spar', val=np.array([12., 12., 12., 12., 12., 12., 12., 12., np.sqrt(1./3. * (12.**2. + 8.3**2. + 12. * 8.3)), 8.3]), units='m')
#ivc.add_output('wt_spar', val=np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]), units='m')
ivc.add_output('L_spar', val=np.array([13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 8., 14.]), units='m')
ivc.add_output('D_tower_p', val=np.array([8.3, 8.02166998, 7.74333996, 7.46500994, 7.18667992, 6.9083499, 6.63001988, 6.35168986, 6.07335984, 5.79502982, 5.5]), units='m')
ivc.add_output('wt_tower_p', val=np.array([0.038, 0.038, 0.034, 0.034, 0.030, 0.030, 0.026, 0.026, 0.022, 0.022, 0.018]), units='m')
#ivc.add_output('D_tower', val=np.array([8.16083499,7.88250497, 7.60417495, 7.32584493, 7.04751491, 6.76918489, 6.49085487, 6.21252485, 5.93419483, 5.64751491]), units='m')
#ivc.add_output('wt_tower', val=np.array([0.038, 0.036, 0.034, 0.032, 0.030, 0.028, 0.026, 0.024, 0.022, 0.020]), units='m')
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
ivc.add_output('EA_moor', val=384243000., units='N')
ivc.add_output('mass_dens_moor', val=155.41, units='kg/m')
ivc.add_output('len_hor_moor', val=848.67, units='m')
ivc.add_output('len_tot_moor', val=902.2, units='m')
ivc.add_output('rho_wind', val=1.25, units='kg/m**3')
ivc.add_output('I_d', val=160234250.0, units='kg*m**2')
#ivc.add_output('windspeed_0', val=np.array([15., 21.]), units='m/s')
#ivc.add_output('Hs', val=3., units='m')
#ivc.add_output('Tp', val=10., units='s')
ivc.add_output('k_p', val=0.1794, units='rad*s/rad')
ivc.add_output('k_i', val=0.0165, units='rad/rad')
ivc.add_output('k_t', val=-0., units='rad*s/m')
ivc.add_output('omega_lowpass', val=2.*np.pi/0.8, units='rad/s')
#ivc.add_output('K_moor', val=71000., units='N/m')
#ivc.add_output('M_moor', val=330000., units='kg')
ivc.add_output('gain_corr_factor', val=0.25104)
ivc.add_output('Cd', val=0.7)
#ivc.add_output('alpha_damp', val=0.007, units='s')
ivc.add_output('struct_damp_ratio', val=0.5*0.007*2.*np.pi/2.395)

ivc.add_output('t_w_stiff', val=0.02*np.ones(10), units='m')
ivc.add_output('t_f_stiff', val=0.02*np.ones(10), units='m')
ivc.add_output('h_stiff', val=0.6*np.ones(10), units='m')
ivc.add_output('b_stiff', val=0.8*np.ones(10), units='m')
ivc.add_output('l_stiff', val=1.0*np.ones(10), units='m')

ivc.add_output('angle_hull', val=0., units='rad')
ivc.add_output('buck_len', val=1.)
ivc.add_output('f_y', val=250., units='MPa')

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

from ECs import ECs

prob.model.add_subsystem('ECs', ECs(EC=EC), promotes_outputs=['windspeed_0', 'Hs', 'Tp', 'p'])

parallel = prob.model.add_subsystem('parallel', ParallelGroup(), promotes_inputs=['D_spar_p', 'wt_spar_p', 'L_spar', 'D_tower_p', 'wt_tower_p', 'L_tower', 'rho_ball', 'wt_ball', 'M_nacelle', 'CoG_rotor', 'CoG_nacelle', 'I_rotor', 'M_rotor', 'water_depth', 'z_moor', 'EA_moor', 'mass_dens_moor', 'len_hor_moor', 'len_tot_moor', 'rho_wind', 'I_d', 'k_p', 'k_i', 'k_t', 'omega_lowpass', 'gain_corr_factor', 'Cd', 'struct_damp_ratio'])

from total_fatigue_damage import TotalFatigueDamage

prob.model.add_subsystem('total_fatigue_damage', TotalFatigueDamage(EC=EC))

from condition_group import Condition

for i in xrange(EC['N_EC']):
	parallel.add_subsystem('cond%d' % i, Condition(blades=blades, freqs=freqs), promotes_inputs=['D_spar_p', 'wt_spar_p', 'L_spar', 'D_tower_p', 'wt_tower_p', 'L_tower', 'rho_ball', 'wt_ball', 'M_nacelle', 'CoG_rotor', 'CoG_nacelle', 'I_rotor', 'M_rotor', 'water_depth', 'z_moor', 'EA_moor', 'mass_dens_moor', 'len_hor_moor', 'len_tot_moor', 'rho_wind', 'I_d', 'k_p', 'k_i', 'k_t', 'omega_lowpass', 'gain_corr_factor', 'Cd', 'struct_damp_ratio'])

	prob.model.connect('windspeed_0', 'parallel.cond%d.windspeed_0' % i, src_indices=[i])
	prob.model.connect('Hs', 'parallel.cond%d.Hs' % i, src_indices=[i])
	prob.model.connect('Tp', 'parallel.cond%d.Tp' % i, src_indices=[i])
	prob.model.connect('parallel.cond%d.fatigue_damage' % i, 'total_fatigue_damage.fatigue_damage%d' % i, src_indices=[0])
	prob.model.connect('p', 'total_fatigue_damage.p%d' % i, src_indices=[i])

#hull_buckling_group = HullBuckling()

#prob.model.add_subsystem('hull_buckling', hull_buckling_group, promotes_inputs=['D_spar_p', 'wt_spar_p', 'Z_spar', 'M_spar', 'M_ball', 'L_ball', 'spar_draft', 'M_moor', 'z_moor',\
#'dthrust_dv', 'dmoment_dv', 't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'l_stiff', 'angle_hull', 'f_y', 'buck_len'], promotes_outputs=['shell_buckling', 'ring_buckling_1', 'ring_buckling_2', 'col_buckling', 'constr_area_ringstiff', 'constr_hoop_stress', 'constr_mom_inertia_ringstiff'])

#aero_group.linear_solver = LinearRunOnce()
#mooring_group.linear_solver = DirectSolver()
#substructure_group.linear_solver = DirectSolver()
#statespace_group.linear_solver = DirectSolver()
#viscous_group.linear_solver = LinearBlockGS(maxiter=30)
#viscous_group.nonlinear_solver = NonlinearBlockGS(atol=1e-5, rtol=1e-5)
#postpro_group.linear_solver = LinearRunOnce()
#hull_buckling_group.linear_solver = LinearRunOnce()
#prob.model.linear_solver = LinearRunOnce()

from openmdao.api import ScipyOptimizeDriver#, pyOptSparseDriver
#prob.driver = ScipyOptimizeDriver()
#prob.driver = pyOptSparseDriver()
#prob.driver.options['optimizer'] = 'SNOPT'

#prob.add_recorder(SqliteRecorder("cases.sql"))

#prob.recording_options['includes'] = []
#prob.recording_options['record_objectives'] = True
#prob.recording_options['record_constraints'] = True
#prob.recording_options['record_desvars'] = True

#prob.model.add_design_var('z_moor', lower=-120., upper=0.)

#prob.model.add_objective('stddev_pitch')

prob.setup()

prob.run_model()

#prob.check_totals(['stddev_pitch'],['z_moor'])

#prob.run_driver()

#prob.record_iteration('final')
#prob.cleanup()

#print prob['B_visc_11']
#print prob['stddev_vel_distr']
#print prob['stddev_surge'] #[2.31619926] #[3.56668782] #[5.98991904]
#print prob['stddev_pitch'] #[0.02000308] #[0.02149176] #[0.01985482]
#print prob['stddev_bend'] #[0.09505861] #[0.09791108] #[0.08875727]
#print prob['stddev_rotspeed'] #[0.09418536] #[0.09707084] #[0.13249565]
print prob['parallel.cond0.fatigue_damage'][0] #0.00030962225040664926 #0.0003224123310311592 #0.00016584825655304263
print prob['parallel.cond1.fatigue_damage'][0]
print prob['total_fatigue_damage.total_fatigue_damage']

#[  2.39647226  35.29681523 115.97636051]