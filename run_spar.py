import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss

from openmdao.api import Problem, IndepVarComp, LinearRunOnce, DirectSolver, LinearBlockGS, ScipyKrylov, NonlinearBlockGS, NewtonSolver, SqliteRecorder

from steady_bldpitch import SteadyBladePitch
from steady_rotspeed import SteadyRotSpeed
from gain_schedule import GainSchedule
from aero_group import Aero
from mooring_group import Mooring
from substructure_group import Substructure
from statespace_group import StateSpace
from wave_spectrum import WaveSpectrum
from wind_spectrum import WindSpectrum
from interp_wave_forces import InterpWaveForces
from viscous_group import Viscous
from postpro_group import Postpro

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

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('D_spar', val=np.array([12., 12., 12., 12., 12., 12., 12., 12., np.sqrt(1./3. * (12.**2. + 8.3**2. + 12. * 8.3)), 8.3]), units='m')
ivc.add_output('L_spar', val=np.array([13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 8., 14.]), units='m')
ivc.add_output('wt_spar', val=np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]), units='m')
ivc.add_output('D_tower', val=np.array([8.16083499,7.88250497, 7.60417495, 7.32584493, 7.04751491, 6.76918489, 6.49085487, 6.21252485, 5.93419483, 5.64751491]), units='m')
ivc.add_output('L_tower', val=np.array([10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 11.13]), units='m')
ivc.add_output('wt_tower', val=np.array([0.038, 0.036, 0.034, 0.032, 0.030, 0.028, 0.026, 0.024, 0.022, 0.020]), units='m')
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
ivc.add_output('windspeed_0', val=21., units='m/s')
ivc.add_output('Hs', val=3., units='m')
ivc.add_output('Tp', val=10., units='s')
ivc.add_output('k_p', val=0.1794, units='rad*s/rad')
ivc.add_output('k_i', val=0.0165, units='rad/rad')
ivc.add_output('omega_lowpass', val=2.*np.pi/0.8, units='rad/s')
#ivc.add_output('K_moor', val=71000., units='N/m')
#ivc.add_output('M_moor', val=330000., units='kg')
ivc.add_output('gain_corr_factor', val=0.25104)
ivc.add_output('Cd', val=0.7)
ivc.add_output('alpha_damp', val=0.007, units='s')

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

prob.model.add_subsystem('steady_rotspeed', SteadyRotSpeed(), promotes_inputs=['windspeed_0'], promotes_outputs=['rotspeed_0'])

prob.model.add_subsystem('steady_bldpitch', SteadyBladePitch(), promotes_inputs=['windspeed_0'], promotes_outputs=['bldpitch_0'])

#prob.model.add_subsystem('gain_schedule', GainSchedule(), promotes_inputs=['bldpitch_0'], promotes_outputs=['gain_corr_factor'])

aero_group = Aero(blades=blades, freqs=freqs)

prob.model.add_subsystem('aero', aero_group, promotes_inputs=['rho_wind', 'windspeed_0', 'bldpitch_0', 'rotspeed_0'], promotes_outputs=['thrust_wind', \
	'moment_wind', 'torque_wind', 'thrust_0', 'torque_0', 'dthrust_dv', 'dmoment_dv', 'dtorque_dv', 'dthrust_drotspeed', 'dtorque_drotspeed', \
	'dthrust_dbldpitch', 'dtorque_dbldpitch'])

mooring_group = Mooring()

prob.model.add_subsystem('mooring', mooring_group, promotes_inputs=['z_moor', 'water_depth', 'EA_moor', 'mass_dens_moor', 'len_hor_moor', 'len_tot_moor', \
'thrust_0'], promotes_outputs=['K_moor', 'M_moor', 'moor_offset'])

substructure_group = Substructure(freqs=freqs)

prob.model.add_subsystem('substructure', substructure_group, promotes_inputs=['D_spar', 'L_spar', 'wt_spar', 'D_tower', 'L_tower', 'wt_tower', \
	'rho_ball', 'wt_ball', 'M_nacelle', 'M_rotor', 'CoG_nacelle', 'CoG_rotor', 'I_rotor', 'water_depth', 'z_moor', 'K_moor', 'M_moor', \
	'dthrust_dv', 'dmoment_dv', 'alpha_damp'], promotes_outputs=['M_global', 'A_global', 'K_global', 'Re_wave_forces', 'Im_wave_forces', 'x_d_towertop', \
	'z_sparnode', 'x_sparelem', 'Z_spar', 'B_aero_11', 'B_aero_15', 'B_aero_17', 'B_aero_55', 'B_aero_57', 'B_aero_77', 'B_struct_77'])

statespace_group = StateSpace(freqs=freqs)

prob.model.add_subsystem('statespace', statespace_group, promotes_inputs=['M_global', 'A_global', 'K_global', 'CoG_rotor', 'I_d', 'dthrust_dv', \
	'dmoment_dv', 'dtorque_dv', 'dthrust_drotspeed', 'dthrust_dbldpitch', 'dtorque_dbldpitch', 'omega_lowpass', 'k_i', 'k_p', \
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
	'Im_RAO_Mwind_vel_surge', 'Re_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_pitch', 'Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_bend'], \
	promotes_outputs=['stddev_surge', 'stddev_pitch', 'stddev_bend', 'stddev_rotspeed'])


aero_group.linear_solver = LinearRunOnce()
mooring_group.linear_solver = DirectSolver()
substructure_group.linear_solver = DirectSolver()
statespace_group.linear_solver = DirectSolver()
viscous_group.linear_solver = LinearBlockGS(maxiter=20)
viscous_group.nonlinear_solver = NonlinearBlockGS(atol=1e-5, rtol=1e-5)
postpro_group.linear_solver = LinearBlockGS()
prob.model.linear_solver = LinearRunOnce()

#2.315413453135903 0.019966294432131052 55402666.15073847 0.09418224139718613
#0.28435090406814023
#0.00030560768476726487

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

prob.check_totals(['stddev_pitch'],['z_moor'])

#prob.run_driver()

#prob.record_iteration('final')
#prob.cleanup()

#print prob['B_visc_11']
#print prob['stddev_vel_distr']
print prob['stddev_surge'] #[2.31503107]
print prob['stddev_pitch'] #[0.01996003]
print prob['stddev_bend'] #[0.095082]
print prob['stddev_rotspeed'] #[0.09418025]

#[  2.39647226  35.29681523 115.97636051]
"""
omega = freqs['omega']
omega_wave = freqs['omega_wave']
N_omega = len(omega)
mag = np.abs(prob['Re_H_feedbk'] + 1j * prob['Im_H_feedbk'])
phase = np.angle(prob['Re_H_feedbk'] + 1j * prob['Im_H_feedbk'])

Xcal = prob['Re_wave_forces'] + 1j * prob['Im_wave_forces']

Xcal1_FD = np.interp(omega, omega_wave, Xcal[:,0,0])
Xcal5_FD = np.interp(omega, omega_wave, Xcal[:,1,0])
Xcal7_FD = np.interp(omega, omega_wave, Xcal[:,2,0])

RAO_wave_surge = mag[:,0,3] * np.exp(1j * phase[:,0,3]) * Xcal1_FD + mag[:,0,4] * np.exp(1j * phase[:,0,4]) * Xcal5_FD + mag[:,0,5] * np.exp(1j * phase[:,0,5]) * Xcal7_FD
RAO_wave_pitch = mag[:,1,3] * np.exp(1j * phase[:,1,3]) * Xcal1_FD + mag[:,1,4] * np.exp(1j * phase[:,1,4]) * Xcal5_FD + mag[:,1,5] * np.exp(1j * phase[:,1,5]) * Xcal7_FD
RAO_wave_bend = mag[:,2,3] * np.exp(1j * phase[:,2,3]) * Xcal1_FD + mag[:,2,4] * np.exp(1j * phase[:,2,4]) * Xcal5_FD + mag[:,2,5] * np.exp(1j * phase[:,2,5]) * Xcal7_FD
RAO_wave_rotspeed = mag[:,6,3] * np.exp(1j * phase[:,6,3]) * Xcal1_FD + mag[:,6,4] * np.exp(1j * phase[:,6,4]) * Xcal5_FD + mag[:,6,5] * np.exp(1j * phase[:,6,5]) * Xcal7_FD
RAO_wave_rot_LP = mag[:,7,3] * np.exp(1j * phase[:,7,3]) * Xcal1_FD + mag[:,7,4] * np.exp(1j * phase[:,7,4]) * Xcal5_FD + mag[:,7,5] * np.exp(1j * phase[:,7,5]) * Xcal7_FD
RAO_wave_rotspeed_LP = mag[:,8,3] * np.exp(1j * phase[:,8,3]) * Xcal1_FD + mag[:,8,4] * np.exp(1j * phase[:,8,4]) * Xcal5_FD + mag[:,8,5] * np.exp(1j * phase[:,8,5]) * Xcal7_FD
RAO_wave_bldpitch = prob['gain_corr_factor'] * prob['k_i'] * RAO_wave_rot_LP + prob['gain_corr_factor'] * prob['k_p'] * RAO_wave_rotspeed_LP
RAO_wave_vel_surge = (mag[:,0,3] * np.exp(1j * phase[:,0,3]) * Xcal1_FD + mag[:,0,4] * np.exp(1j * phase[:,0,4]) * Xcal5_FD + mag[:,0,5] * np.exp(1j * phase[:,0,5]) * Xcal7_FD) * omega * 1j
RAO_wave_vel_pitch = (mag[:,1,3] * np.exp(1j * phase[:,1,3]) * Xcal1_FD + mag[:,1,4] * np.exp(1j * phase[:,1,4]) * Xcal5_FD + mag[:,1,5] * np.exp(1j * phase[:,1,5]) * Xcal7_FD) * omega * 1j
RAO_wave_vel_bend = (mag[:,2,3] * np.exp(1j * phase[:,2,3]) * Xcal1_FD + mag[:,2,4] * np.exp(1j * phase[:,2,4]) * Xcal5_FD + mag[:,2,5] * np.exp(1j * phase[:,2,5]) * Xcal7_FD) * omega * 1j
RAO_wave_acc_surge = (mag[:,0,3] * np.exp(1j * phase[:,0,3]) * Xcal1_FD + mag[:,0,4] * np.exp(1j * phase[:,0,4]) * Xcal5_FD + mag[:,0,5] * np.exp(1j * phase[:,0,5]) * Xcal7_FD) * omega**2. * (-1.)
RAO_wave_acc_pitch = (mag[:,1,3] * np.exp(1j * phase[:,1,3]) * Xcal1_FD + mag[:,1,4] * np.exp(1j * phase[:,1,4]) * Xcal5_FD + mag[:,1,5] * np.exp(1j * phase[:,1,5]) * Xcal7_FD) * omega**2. * (-1.)
RAO_wave_acc_bend = (mag[:,2,3] * np.exp(1j * phase[:,2,3]) * Xcal1_FD + mag[:,2,4] * np.exp(1j * phase[:,2,4]) * Xcal5_FD + mag[:,2,5] * np.exp(1j * phase[:,2,5]) * Xcal7_FD) * omega**2. * (-1.)

RAO_wind_surge = mag[:,0,0] * np.exp(1j * phase[:,0,0]) * prob['thrust_wind'] + mag[:,0,2] * np.exp(1j * phase[:,0,2]) * prob['torque_wind']
RAO_wind_pitch = mag[:,1,0] * np.exp(1j * phase[:,1,0]) * prob['thrust_wind'] + mag[:,1,2] * np.exp(1j * phase[:,1,2]) * prob['torque_wind']
RAO_wind_bend = mag[:,2,0] * np.exp(1j * phase[:,2,0]) * prob['thrust_wind'] + mag[:,2,2] * np.exp(1j * phase[:,2,2]) * prob['torque_wind']
RAO_wind_rotspeed = mag[:,6,0] * np.exp(1j * phase[:,6,0]) * prob['thrust_wind'] + mag[:,6,2] * np.exp(1j * phase[:,6,2]) * prob['torque_wind']
RAO_wind_rot_LP = mag[:,7,0] * np.exp(1j * phase[:,7,0]) * prob['thrust_wind'] + mag[:,7,2] * np.exp(1j * phase[:,7,2]) * prob['torque_wind']
RAO_wind_rotspeed_LP = mag[:,8,0] * np.exp(1j * phase[:,8,0]) * prob['thrust_wind'] + mag[:,8,2] * np.exp(1j * phase[:,8,2]) * prob['torque_wind']
RAO_wind_bldpitch = prob['gain_corr_factor'] * prob['k_i'] * RAO_wind_rot_LP + prob['gain_corr_factor'] * prob['k_p'] * RAO_wind_rotspeed_LP
RAO_wind_vel_surge = (mag[:,0,0] * np.exp(1j * phase[:,0,0]) * prob['thrust_wind'] + mag[:,0,2] * np.exp(1j * phase[:,0,2]) * prob['torque_wind']) * omega * 1j
RAO_wind_vel_pitch = (mag[:,1,0] * np.exp(1j * phase[:,1,0]) * prob['thrust_wind'] + mag[:,1,2] * np.exp(1j * phase[:,1,2]) * prob['torque_wind']) * omega * 1j
RAO_wind_vel_bend = (mag[:,2,0] * np.exp(1j * phase[:,2,0]) * prob['thrust_wind'] + mag[:,2,2] * np.exp(1j * phase[:,2,2]) * prob['torque_wind']) * omega * 1j
RAO_wind_acc_surge = (mag[:,0,0] * np.exp(1j * phase[:,0,0]) * prob['thrust_wind'] + mag[:,0,2] * np.exp(1j * phase[:,0,2]) * prob['torque_wind']) * omega**2. * (-1.)
RAO_wind_acc_pitch = (mag[:,1,0] * np.exp(1j * phase[:,1,0]) * prob['thrust_wind'] + mag[:,1,2] * np.exp(1j * phase[:,1,2]) * prob['torque_wind']) * omega**2. * (-1.)
RAO_wind_acc_bend = (mag[:,2,0] * np.exp(1j * phase[:,2,0]) * prob['thrust_wind'] + mag[:,2,2] * np.exp(1j * phase[:,2,2]) * prob['torque_wind']) * omega**2. * (-1.)

RAO_Mwind_surge = mag[:,0,1] * np.exp(1j * phase[:,0,1]) * prob['moment_wind']
RAO_Mwind_pitch = mag[:,1,1] * np.exp(1j * phase[:,1,1]) * prob['moment_wind']
RAO_Mwind_bend = mag[:,2,1] * np.exp(1j * phase[:,2,1]) * prob['moment_wind']
RAO_Mwind_rotspeed = mag[:,6,1] * np.exp(1j * phase[:,6,1]) * prob['moment_wind']
RAO_Mwind_rot_LP = mag[:,7,1] * np.exp(1j * phase[:,7,1]) * prob['moment_wind']
RAO_Mwind_rotspeed_LP = mag[:,8,1] * np.exp(1j * phase[:,8,1]) * prob['moment_wind']
RAO_Mwind_bldpitch = prob['gain_corr_factor'] * prob['k_i'] * RAO_wind_rot_LP + prob['gain_corr_factor'] * prob['k_p'] * RAO_wind_rotspeed_LP
RAO_Mwind_vel_surge = (mag[:,0,1] * np.exp(1j * phase[:,0,1]) * prob['moment_wind']) * omega * 1j
RAO_Mwind_vel_pitch = (mag[:,1,1] * np.exp(1j * phase[:,1,1]) * prob['moment_wind']) * omega * 1j
RAO_Mwind_vel_bend = (mag[:,2,1] * np.exp(1j * phase[:,2,1]) * prob['moment_wind']) * omega * 1j
RAO_Mwind_acc_surge = (mag[:,0,1] * np.exp(1j * phase[:,0,1]) * prob['moment_wind']) * omega**2. * (-1.)
RAO_Mwind_acc_pitch = (mag[:,1,1] * np.exp(1j * phase[:,1,1]) * prob['moment_wind']) * omega**2. * (-1.)
RAO_Mwind_acc_bend = (mag[:,2,1] * np.exp(1j * phase[:,2,1]) * prob['moment_wind']) * omega**2. * (-1.)

X1_lin_FD = np.abs(RAO_wave_surge)**2. * prob['S_wave'] + np.abs(RAO_wind_surge)**2. * prob['S_wind'] + np.abs(RAO_Mwind_surge)**2. * prob['S_wind']

X5_lin_FD = np.abs(RAO_wave_pitch)**2. * prob['S_wave'] + np.abs(RAO_wind_pitch)**2. * prob['S_wind'] + np.abs(RAO_Mwind_pitch)**2. * prob['S_wind']

X7_lin_FD = np.abs(RAO_wave_bend)**2. * prob['S_wave'] + np.abs(RAO_wind_bend)**2. * prob['S_wind'] + np.abs(RAO_Mwind_bend)**2. * prob['S_wind']

phi_dot_lin_FD = np.abs(RAO_wave_rotspeed)**2. * prob['S_wave'] + np.abs(RAO_wind_rotspeed)**2. * prob['S_wind'] + np.abs(RAO_Mwind_rotspeed)**2. * prob['S_wind']

X1vel_lin_FD = np.abs(RAO_wave_vel_surge)**2. * prob['S_wave'] + np.abs(RAO_wind_vel_surge)**2. * prob['S_wind'] + np.abs(RAO_Mwind_vel_surge)**2. * prob['S_wind']

mom_acc_surge = 98294935.72543612
mom_acc_pitch = 10583502511.569796
mom_acc_bend = 77898403.15720505
mom_damp_surge = 16097428.734389322
mom_damp_pitch = 1919768134.6718657
mom_damp_bend = 16218043.969506808
mom_grav_pitch = 963944031.431848
mom_grav_bend = 10878305.867464177
mom_rotspeed = -202745723.95380586
mom_bldpitch = -991123196.6665341

S_wave_mom_TB = np.zeros(N_omega)
for i in xrange(N_omega):
	S_wave_mom_TB[i] = np.abs(-mom_acc_surge * RAO_wave_acc_surge[i] - mom_acc_pitch * RAO_wave_acc_pitch[i] - mom_acc_bend * RAO_wave_acc_bend[i] - mom_damp_surge * RAO_wave_vel_surge[i] - mom_damp_pitch * RAO_wave_vel_pitch[i] - mom_damp_bend * RAO_wave_vel_bend[i] + mom_grav_pitch * RAO_wave_pitch[i] + mom_grav_bend * RAO_wave_bend[i] + mom_rotspeed * RAO_wave_rotspeed[i] + mom_bldpitch * RAO_wave_bldpitch[i])**2. * prob['S_wave'][i]

S_wind_mom_TB = np.zeros(N_omega)
for i in xrange(N_omega):
	S_wind_mom_TB[i] = np.abs(-mom_acc_surge * RAO_wind_acc_surge[i] - mom_acc_pitch * RAO_wind_acc_pitch[i] - mom_acc_bend * RAO_wind_acc_bend[i] - mom_damp_surge * RAO_wind_vel_surge[i] - mom_damp_pitch * RAO_wind_vel_pitch[i] - mom_damp_bend * RAO_wind_vel_bend[i] + mom_grav_pitch * RAO_wind_pitch[i] + mom_grav_bend * RAO_wind_bend[i] + mom_rotspeed * RAO_wind_rotspeed[i] + mom_bldpitch * RAO_wind_bldpitch[i] + (119.- 10.) * prob['dthrust_dv'] * prob['thrust_wind'][i])**2. * prob['S_wind'][i]

S_Mwind_mom_TB = np.zeros(N_omega)
for i in xrange(N_omega):
	S_Mwind_mom_TB[i] = np.abs(-mom_acc_surge * RAO_Mwind_acc_surge[i] - mom_acc_pitch * RAO_Mwind_acc_pitch[i] - mom_acc_bend * RAO_Mwind_acc_bend[i] - mom_damp_surge * RAO_Mwind_vel_surge[i] - mom_damp_pitch * RAO_Mwind_vel_pitch[i] - mom_damp_bend * RAO_Mwind_vel_bend[i] + mom_grav_pitch * RAO_Mwind_pitch[i] + mom_grav_bend * RAO_Mwind_bend[i] + mom_rotspeed * RAO_Mwind_rotspeed[i] + mom_bldpitch * RAO_Mwind_bldpitch[i] + prob['dmoment_dv'] * prob['moment_wind'][i])**2. * prob['S_wind'][i]

S_mom_TB = S_wave_mom_TB + S_wind_mom_TB + S_Mwind_mom_TB

print np.sqrt(np.trapz(X1_lin_FD, omega)), np.sqrt(np.trapz(X5_lin_FD, omega)), np.sqrt(np.trapz(S_mom_TB, omega)), np.sqrt(np.trapz(phi_dot_lin_FD, omega))
print np.sqrt(np.trapz(X1vel_lin_FD, omega))

S_stress_TB = S_mom_TB / (np.pi / 64. * (8.3**4. - (8.3 - 2. * 0.038)**4.))**2. * (8.3 / 2. * 10.**(-6.))**2.

C = 10**12.164
k = 3.0

m0 = np.trapz(S_stress_TB,omega)
m1 = np.trapz(omega * S_stress_TB,omega)
m2 = np.trapz(omega**2. * S_stress_TB,omega)
m4 = np.trapz(omega**4. * S_stress_TB,omega)

sigma = np.sqrt(m0)

x_m = m1 / m0 * np.sqrt(m2 / m4)
alpha_2 = m2 / np.sqrt(m0 * m4)
v_p = 1. / (2. * np.pi) * np.sqrt(m4 / m2)

G1 = 2. * (x_m - alpha_2**2.) / (1. + alpha_2**2.)
R = (alpha_2 - x_m - G1**2.) / (1. - alpha_2 - G1 + G1**2.)
G2 = (1. - alpha_2 - G1 + G1**2.) / (1. - R)
G3 = 1. - G1 - G2
Q = 1.25 * (alpha_2 - G3 - G2 * R) / G1

D = C**(-1.) * v_p * (2. * sigma)**k * (G1 * Q**k * ss.gamma(1. + k) + np.sqrt(2.)**k * ss.gamma(1. + k / 2.) * (G2 * R**k + G3)) * (0.038 / 0.025)**(0.2 * k)

print D * 3600.

plt.plot(omega,S_mom_TB)
plt.show()
"""