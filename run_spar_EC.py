import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
import scipy.signal as ssig

from openmdao.api import Problem, IndepVarComp, LinearRunOnce, DirectSolver, LinearBlockGS, ScipyKrylov, NonlinearBlockGS, NewtonSolver, SqliteRecorder, BroydenSolver, ExecComp

from steady_bldpitch import SteadyBladePitch
from steady_rotspeed import SteadyRotSpeed
from gain_schedule import GainSchedule
from mooring_chain import MooringChain
from aero_group import Aero
from towerdim_group import Towerdim
from mean_tower_drag import MeanTowerDrag
from mooring_group import Mooring
from substructure_group import Substructure
from statespace_group import StateSpace
from wave_spectrum import WaveSpectrum
from wind_spectrum import WindSpectrum
from interp_wave_forces import InterpWaveForces
from viscous_group import Viscous
from postpro_group import Postpro
from hull_buckling_balance import HullBalance
from hull_buckling_group import HullBuckling
from tower_buckling_group import TowerBuckling
from extreme_response_group import ExtremeResponse
from cost_group import Cost

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
ivc.add_output('gamma_F_moor', val=1.65)
#ivc.add_output('EA_moor', val=384243000., units='N')
#ivc.add_output('mass_dens_moor', val=155.41, units='kg/m')
ivc.add_output('len_hor_moor', val=848.67, units='m')
ivc.add_output('len_tot_moor', val=902.2, units='m')
ivc.add_output('rho_wind', val=1.25, units='kg/m**3')
ivc.add_output('I_d', val=160234250.0, units='kg*m**2')
ivc.add_output('windspeed_0', val=21., units='m/s')
ivc.add_output('Hs', val=3., units='m')
ivc.add_output('Tp', val=10., units='s')
ivc.add_output('k_p', val=0.1794, units='rad*s/rad')
ivc.add_output('k_i', val=0.0165, units='rad/rad')
ivc.add_output('k_t', val=-0., units='rad*s/m')
ivc.add_output('omega_lowpass', val=2.*np.pi/0.8, units='rad/s')
ivc.add_output('omega_notch', val=10.16, units='rad/s')
ivc.add_output('bandwidth_notch', val=0.1, units='rad/s')
#ivc.add_output('K_moor', val=71000., units='N/m')
#ivc.add_output('M_moor', val=330000., units='kg')
#ivc.add_output('gain_corr_factor', val=0.25104)
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
ivc.add_output('f_y', val=355., units='MPa')

ivc.add_output('gamma_M_tower', val=1.1)
ivc.add_output('gamma_F_tower', val=1.35)

ivc.add_output('gamma_F_hull', val=1.35)

ivc.add_output('maxval_surge', val=30., units='m')
ivc.add_output('maxval_pitch', val=10.*np.pi/180., units='rad')

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

prob.model.add_subsystem('steady_rotspeed', SteadyRotSpeed(), promotes_inputs=['windspeed_0'], promotes_outputs=['rotspeed_0'])

prob.model.add_subsystem('steady_bldpitch', SteadyBladePitch(), promotes_inputs=['windspeed_0'], promotes_outputs=['bldpitch_0'])

prob.model.add_subsystem('gain_schedule', GainSchedule(), promotes_inputs=['bldpitch_0'], promotes_outputs=['gain_corr_factor'])

prob.model.add_subsystem('mooring_chain', MooringChain(), promotes_inputs=['D_moor', 'gamma_F_moor'], promotes_outputs=['mass_dens_moor', 'EA_moor', 'maxval_moor_ten'])

aero_group = Aero(blades=blades, freqs=freqs)

prob.model.add_subsystem('aero', aero_group, promotes_inputs=['rho_wind', 'windspeed_0', 'bldpitch_0', 'rotspeed_0'], promotes_outputs=['thrust_wind', \
	'moment_wind', 'torque_wind', 'thrust_0', 'torque_0', 'dthrust_dv', 'dmoment_dv', 'dtorque_dv', 'dthrust_drotspeed', 'dtorque_drotspeed', \
	'dthrust_dbldpitch', 'dtorque_dbldpitch'])

towerdim_group = Towerdim()

prob.model.add_subsystem('towerdim', towerdim_group, promotes_inputs=['D_tower_p', 'L_tower'], promotes_outputs=['D_tower', 'Z_tower'])

prob.model.add_subsystem('mean_tower_drag', MeanTowerDrag(), promotes_inputs=['D_tower', 'Z_tower', 'L_tower', 'windspeed_0', 'Cd_tower', 'CoG_rotor', 'rho_wind'], \
	promotes_outputs=['F0_tower_drag', 'Z0_tower_drag'])

mooring_group = Mooring()

prob.model.add_subsystem('mooring', mooring_group, promotes_inputs=['z_moor', 'water_depth', 'EA_moor', 'mass_dens_moor', 'len_hor_moor', 'len_tot_moor', \
'thrust_0', 'F0_tower_drag'], promotes_outputs=['M_moor_zero', 'K_moor', 'M_moor', 'moor_offset', 'maxval_fairlead', 'mean_moor_ten'])

substructure_group = Substructure(freqs=freqs)

prob.model.add_subsystem('substructure', substructure_group, promotes_inputs=['D_spar_p', 'L_spar', 'wt_spar_p', 'L_tower', 'wt_tower_p', \
	'rho_ball', 'wt_ball', 'M_nacelle', 'M_rotor', 'CoG_nacelle', 'CoG_rotor', 'I_rotor', 'water_depth', 'z_moor', 'M_moor_zero', 'K_moor', 'M_moor', \
	'dthrust_dv', 'dmoment_dv', 'struct_damp_ratio', 't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'l_stiff', 'D_tower', 'Z_tower'], \
	promotes_outputs=['M_global', 'A_global', 'K_global', 'Re_wave_forces', 'Im_wave_forces', 'x_d_towertop', 'z_sparnode', 'x_sparelem', \
	'Z_spar', 'M_spar', 'M_ball', 'L_ball', 'spar_draft', 'D_spar', 'wt_spar', 'wt_tower', 'tot_M_spar', 'tot_M_tower', 'B_aero_11', 'B_aero_15', \
	'B_aero_17', 'B_aero_55', 'B_aero_57', 'B_aero_77', 'B_struct_77', 'A_R', 'r_e', 'buoy_spar', 'CoB', 'M_turb', 'CoG_total', 'wave_number', \
	'x_sparnode', 'M_ball_elem', 'M_tower', 'z_towernode'])

statespace_group = StateSpace(freqs=freqs)

prob.model.add_subsystem('statespace', statespace_group, promotes_inputs=['M_global', 'A_global', 'K_global', 'CoG_rotor', 'I_d', 'dthrust_dv', \
	'dmoment_dv', 'dtorque_dv', 'dthrust_drotspeed', 'dthrust_dbldpitch', 'dtorque_dbldpitch', 'omega_lowpass', 'omega_notch', 'bandwidth_notch', 'k_i', 'k_p', 'k_t', \
	'gain_corr_factor', 'x_d_towertop', 'windspeed_0', 'rotspeed_0'], promotes_outputs=['Astr_stiff', 'Astr_ext', 'A_contrl', 'BsCc', 'BcCs', 'B_feedbk'])

prob.model.add_subsystem('wave_spectrum', WaveSpectrum(freqs=freqs), promotes_inputs=['Hs', 'Tp'], promotes_outputs=['S_wave'])

prob.model.add_subsystem('wind_spectrum', WindSpectrum(freqs=freqs), promotes_inputs=['windspeed_0'], promotes_outputs=['S_wind'])

prob.model.add_subsystem('interp_wave_forces', InterpWaveForces(freqs=freqs), promotes_inputs=['Re_wave_forces', 'Im_wave_forces'], promotes_outputs=[\
	'Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend'])

viscous_group = Viscous(freqs=freqs)

prob.model.add_subsystem('viscous', viscous_group, promotes_inputs=['Cd', 'x_sparelem', 'z_sparnode', 'Z_spar', 'D_spar', 'B_aero_11', 'B_aero_15', \
	'B_aero_17', 'B_aero_55', 'B_aero_57', 'B_aero_77', 'B_struct_77', 'M_global', 'A_global', 'CoG_rotor', 'I_d', 'dtorque_dv', 'dtorque_drotspeed', \
	'Astr_stiff', 'Astr_ext', 'A_contrl', 'BsCc', 'BcCs', 'B_feedbk', 'Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', \
	'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'thrust_wind', 'moment_wind', 'torque_wind', 'S_wave', 'S_wind'], \
	promotes_outputs=['A_feedbk', 'Re_H_feedbk', 'Im_H_feedbk', 'Re_RAO_wave_surge', 'Im_RAO_wave_surge', 'Re_RAO_wave_pitch', 'Im_RAO_wave_pitch', 'Re_RAO_wave_bend', \
	'Im_RAO_wave_bend', 'Re_RAO_wind_surge', 'Im_RAO_wind_surge', 'Re_RAO_wind_pitch', 'Im_RAO_wind_pitch', 'Re_RAO_wind_bend', 'Im_RAO_wind_bend', \
	'Re_RAO_Mwind_surge', 'Im_RAO_Mwind_surge', 'Re_RAO_Mwind_pitch', 'Im_RAO_Mwind_pitch', 'Re_RAO_Mwind_bend', 'Im_RAO_Mwind_bend', 'Re_RAO_wave_vel_surge', \
	'Im_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_bend', 'Re_RAO_wind_vel_surge', \
	'Im_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_bend', 'Re_RAO_Mwind_vel_surge', \
	'Im_RAO_Mwind_vel_surge', 'Re_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_pitch', 'Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_bend', 'B_visc_11', 'stddev_vel_distr'])

postpro_group = Postpro(freqs=freqs)

prob.model.add_subsystem('postpro', postpro_group, promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', \
	'Re_wave_force_bend', 'Im_wave_force_bend', 'thrust_wind', 'moment_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'gain_corr_factor', \
	'S_wave', 'S_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'Re_RAO_wave_surge', 'Im_RAO_wave_surge', 'Re_RAO_wave_pitch', 'Im_RAO_wave_pitch', 'Re_RAO_wave_bend', \
	'Im_RAO_wave_bend', 'Re_RAO_wind_surge', 'Im_RAO_wind_surge', 'Re_RAO_wind_pitch', 'Im_RAO_wind_pitch', 'Re_RAO_wind_bend', 'Im_RAO_wind_bend', \
	'Re_RAO_Mwind_surge', 'Im_RAO_Mwind_surge', 'Re_RAO_Mwind_pitch', 'Im_RAO_Mwind_pitch', 'Re_RAO_Mwind_bend', 'Im_RAO_Mwind_bend', 'Re_RAO_wave_vel_surge', \
	'Im_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_bend', 'Re_RAO_wind_vel_surge', \
	'Im_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_bend', 'Re_RAO_Mwind_vel_surge', \
	'Im_RAO_Mwind_vel_surge', 'Re_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_pitch', 'Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_bend', 'D_tower_p', 'wt_tower_p', \
	'Z_tower', 'dthrust_dv', 'dmoment_dv', 'dthrust_drotspeed', 'dthrust_dbldpitch', 'M_tower', 'M_nacelle', 'M_rotor', 'I_rotor', 'CoG_nacelle', 'CoG_rotor', \
	'z_towernode', 'x_towerelem', 'x_towernode', 'x_d_towertop', 'D_spar', 'L_spar', 'Z_spar', 'wave_number', 'water_depth', 'moor_offset', 'z_moor', 'K_moor', \
	'thrust_0', 'buoy_spar', 'CoB', 'M_turb', 'tot_M_spar', 'M_ball', 'CoG_total', 'M_spar', 'stddev_vel_distr', 'z_sparnode', 'x_sparnode', 'x_sparelem', 'spar_draft', \
	'L_ball', 'M_ball_elem', 'F0_tower_drag', 'Z0_tower_drag', 'D_spar_p', 'wt_spar_p'], promotes_outputs=['stddev_surge', 'stddev_pitch', 'stddev_bend', 'stddev_rotspeed', \
	'stddev_bldpitch', 'stddev_tower_stress', 'stddev_hull_moment', 'stddev_fairlead', 'stddev_moor_ten', 'mean_surge', 'mean_pitch', 'mean_tower_stress', 'mean_hull_moment', 'v_z_surge', \
	'v_z_pitch', 'v_z_tower_stress', 'v_z_hull_moment', 'v_z_fairlead', 'v_z_moor_ten', 'tower_fatigue_damage', 'hull_fatigue_damage'])

hull_buckling_balance = HullBalance()

prob.model.add_subsystem('hull_balance', hull_buckling_balance, promotes_inputs=['D_spar_p', 'wt_spar_p', 'Z_spar', 'M_spar', 'M_ball', 'L_ball', 'spar_draft', \
'M_moor', 'z_moor', 'dthrust_dv', 'dmoment_dv', 't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'l_stiff', 'angle_hull', 'f_y', 'A_R'], \
promotes_outputs=['My_shell_buckling', 'My_constr_hoop_stress', 'My_constr_mom_inertia_ringstiff', 'shell_buckling', 'constr_hoop_stress', 'constr_mom_inertia_ringstiff', 'r_f'])

prob.model.add_subsystem('shell_buckling_comp', ExecComp(['maxval_My_shell_buckling = My_shell_buckling / gamma_F_hull'], \
	maxval_My_shell_buckling={'value': np.zeros(10), 'units': 'N*m'}, My_shell_buckling={'value': np.zeros(10), 'units': 'N*m'}, \
	gamma_F_hull={'value': 0.}), promotes_inputs=['My_shell_buckling', 'gamma_F_hull'], promotes_outputs=['maxval_My_shell_buckling'])

prob.model.add_subsystem('hoop_stress_comp', ExecComp(['maxval_My_hoop_stress = My_constr_hoop_stress / gamma_F_hull'], \
	maxval_My_hoop_stress={'value': np.zeros(10), 'units': 'N*m'}, My_constr_hoop_stress={'value': np.zeros(10), 'units': 'N*m'}, \
	gamma_F_hull={'value': 0.}), promotes_inputs=['My_constr_hoop_stress', 'gamma_F_hull'], promotes_outputs=['maxval_My_hoop_stress'])

prob.model.add_subsystem('mom_inertia_comp', ExecComp(['maxval_My_mom_inertia = My_constr_mom_inertia_ringstiff / gamma_F_hull'], \
	maxval_My_mom_inertia={'value': np.zeros(10), 'units': 'N*m'}, My_constr_mom_inertia_ringstiff={'value': np.zeros(10), 'units': 'N*m'}, \
	gamma_F_hull={'value': 0.}), promotes_inputs=['My_constr_mom_inertia_ringstiff', 'gamma_F_hull'], promotes_outputs=['maxval_My_mom_inertia'])

hull_buckling_group = HullBuckling()

prob.model.add_subsystem('hull_buckling', hull_buckling_group, promotes_inputs=['D_spar_p', 'wt_spar_p', 'spar_draft', 't_w_stiff', 'h_stiff', 'b_stiff', 'l_stiff', 'f_y', 'buck_len', 'A_R'], \
promotes_outputs=['ring_buckling_1', 'ring_buckling_2', 'col_buckling', 'constr_area_ringstiff'])

tower_buckling_group = TowerBuckling()

prob.model.add_subsystem('tower_buckling', tower_buckling_group, promotes_inputs=['L_tower', 'D_tower_p', 'wt_tower_p', 'f_y', 'gamma_M_tower', 'gamma_F_tower'], \
	promotes_outputs=['maxval_tower_stress'])

extreme_response_group = ExtremeResponse()

prob.model.add_subsystem('extreme_response', extreme_response_group, promotes_inputs=['maxval_surge', 'maxval_pitch', 'maxval_tower_stress', \
	'maxval_My_shell_buckling', 'maxval_My_hoop_stress', 'maxval_My_mom_inertia', 'maxval_fairlead', 'maxval_moor_ten', 'stddev_surge', 'stddev_pitch', 'stddev_tower_stress', \
	'stddev_hull_moment', 'stddev_fairlead', 'stddev_moor_ten', 'mean_surge', 'mean_pitch', 'mean_tower_stress', 'mean_hull_moment', 'moor_offset', 'mean_moor_ten', 'v_z_surge', 'v_z_pitch', 'v_z_tower_stress', \
	'v_z_hull_moment', 'v_z_fairlead', 'v_z_moor_ten'], promotes_outputs=['short_term_surge_CDF', 'short_term_pitch_CDF', 'short_term_tower_stress_CDF', 'short_term_My_shell_buckling_CDF', \
	'short_term_My_hoop_stress_CDF', 'short_term_My_mom_inertia_CDF', 'short_term_fairlead_CDF', 'short_term_moor_ten_CDF'])

cost_group = Cost()

prob.model.add_subsystem('cost', cost_group, promotes_inputs=['D_spar', 'D_spar_p', 'wt_spar', 'L_spar', 'l_stiff', 'h_stiff', 't_f_stiff', 'A_R', 'r_f', 'r_e', \
	'tot_M_spar', 'D_tower', 'D_tower_p', 'wt_tower', 'L_tower', 'tot_M_tower', 'len_tot_moor', 'mass_dens_moor'], promotes_outputs=['spar_cost', 'tower_cost', \
	'mooring_cost', 'total_cost'])

#aero_group.linear_solver = LinearRunOnce()
#mooring_group.linear_solver = DirectSolver()
#substructure_group.linear_solver = DirectSolver()
#statespace_group.linear_solver = DirectSolver()
#viscous_group.linear_solver = LinearBlockGS(maxiter=30)
viscous_group.nonlinear_solver = NonlinearBlockGS(maxiter=50, atol=1e-5, rtol=1e-5)
#postpro_group.linear_solver = LinearRunOnce()
hull_buckling_balance.linear_solver = DirectSolver()
hull_buckling_balance.nonlinear_solver = BroydenSolver(maxiter=50, atol=1e-8)
#hull_buckling_group.linear_solver = LinearRunOnce()
#prob.model.linear_solver = LinearRunOnce()

from openmdao.api import ScipyOptimizeDriver#, pyOptSparseDriver
prob.driver = ScipyOptimizeDriver()
#prob.driver = pyOptSparseDriver()
#prob.driver.options['optimizer'] = 'SNOPT'
#prob.driver.opt_settings['Major iterations limit'] = 200 #SNOPT
#prob.driver.options['maxiter'] = 2

#prob.add_recorder(SqliteRecorder("cases.sql"))

#prob.recording_options['includes'] = []
#prob.recording_options['record_objectives'] = True
#prob.recording_options['record_constraints'] = True
#prob.recording_options['record_desvars'] = True

#prob.model.add_design_var('z_moor', lower=-120., upper=0.)

#prob.model.add_objective('stddev_pitch')

prob.setup()

prob.run_model()

#print prob['My_shell_buckling']
#print prob['My_constr_hoop_stress']
#print prob['My_constr_mom_inertia_ringstiff']
#print prob['shell_buckling']
#print prob['constr_hoop_stress']
#print prob['constr_mom_inertia_ringstiff']
#print prob['ring_buckling_1']
#print prob['ring_buckling_2']
#print prob['col_buckling']
#print prob['constr_area_ringstiff']

#prob.check_totals(['stddev_pitch'],['z_moor'])

#prob.run_driver()

#prob.record_iteration('final')
#prob.cleanup()
#print prob['mean_tower_stress']
#print prob['stddev_tower_stress']
#print prob['v_z_tower_stress']
#print prob['maxval_tower_stress']
#print prob['short_term_tower_stress_CDF']

#print prob['B_visc_11']
#print prob['stddev_vel_distr']
print prob['stddev_surge'][0] #[2.31806726]
print prob['stddev_pitch'][0] #[0.02006235]
#print prob['stddev_bend'] #[0.09510153]
print prob['stddev_rotspeed'][0] #[0.0941907]
print prob['stddev_bldpitch'][0] #[0.01095254]
print prob['tower_fatigue_damage'][0] #0.00030961246865275963
print prob['hull_fatigue_damage'][-1] #0.00030961246865275963
print prob['total_cost'][0]
print prob['spar_cost'][0]/prob['total_cost'][0]
print prob['tower_cost'][0]/prob['total_cost'][0]
print prob['mooring_cost'][0]/prob['total_cost'][0]

#[  2.39647226  35.29681523 115.97636051]