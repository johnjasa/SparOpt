import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
import scipy.signal as ssig

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
from hull_buckling_group import HullBuckling

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
ivc.add_output('D_spar_p', val=np.array([12., 12., 12., 12., 12., 12., 12., 12., 12., 8.3, 8.3]), units='m') #np.array([12., 12., 12., 12., 12., 12., 12., 12., 12., 8.3, 8.3]) #np.array([12.7, 13.2, 12.7, 12.5, 12.9, 13.8, 15., 15.9, 14.6, 8.4, 8.4])
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
ivc.add_output('windspeed_0', val=21., units='m/s')
ivc.add_output('Hs', val=3., units='m')
ivc.add_output('Tp', val=10., units='s')
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

prob.model.add_subsystem('steady_rotspeed', SteadyRotSpeed(), promotes_inputs=['windspeed_0'], promotes_outputs=['rotspeed_0'])

prob.model.add_subsystem('steady_bldpitch', SteadyBladePitch(), promotes_inputs=['windspeed_0'], promotes_outputs=['bldpitch_0'])

#prob.model.add_subsystem('gain_schedule', GainSchedule(), promotes_inputs=['bldpitch_0'], promotes_outputs=['gain_corr_factor'])

aero_group = Aero(blades=blades, freqs=freqs)

prob.model.add_subsystem('aero', aero_group, promotes_inputs=['rho_wind', 'windspeed_0', 'bldpitch_0', 'rotspeed_0'], promotes_outputs=['thrust_wind', \
	'moment_wind', 'torque_wind', 'thrust_0', 'torque_0', 'dthrust_dv', 'dmoment_dv', 'dtorque_dv', 'dthrust_drotspeed', 'dtorque_drotspeed', \
	'dthrust_dbldpitch', 'dtorque_dbldpitch'])

mooring_group = Mooring()

prob.model.add_subsystem('mooring', mooring_group, promotes_inputs=['z_moor', 'water_depth', 'EA_moor', 'mass_dens_moor', 'len_hor_moor', 'len_tot_moor', \
'thrust_0'], promotes_outputs=['M_moor_zero', 'K_moor', 'M_moor', 'moor_offset'])

substructure_group = Substructure(freqs=freqs)

prob.model.add_subsystem('substructure', substructure_group, promotes_inputs=['D_spar_p', 'L_spar', 'wt_spar_p', 'D_tower_p', 'L_tower', 'wt_tower_p', \
	'rho_ball', 'wt_ball', 'M_nacelle', 'M_rotor', 'CoG_nacelle', 'CoG_rotor', 'I_rotor', 'water_depth', 'z_moor', 'M_moor_zero', 'K_moor', 'M_moor', \
	'dthrust_dv', 'dmoment_dv', 'struct_damp_ratio'], promotes_outputs=['M_global', 'A_global', 'K_global', 'Re_wave_forces', 'Im_wave_forces', 'x_d_towertop', \
	'z_sparnode', 'x_sparelem', 'Z_spar', 'Z_tower', 'M_spar', 'M_ball', 'L_ball', 'spar_draft', 'D_spar', 'B_aero_11', 'B_aero_15', 'B_aero_17', 'B_aero_55', \
	'B_aero_57', 'B_aero_77', 'B_struct_77'])

statespace_group = StateSpace(freqs=freqs)

prob.model.add_subsystem('statespace', statespace_group, promotes_inputs=['M_global', 'A_global', 'K_global', 'CoG_rotor', 'I_d', 'dthrust_dv', \
	'dmoment_dv', 'dtorque_dv', 'dthrust_drotspeed', 'dthrust_dbldpitch', 'dtorque_dbldpitch', 'omega_lowpass', 'k_i', 'k_p', 'k_t', \
	'gain_corr_factor', 'x_d_towertop', 'windspeed_0', 'rotspeed_0'], promotes_outputs=['Astr_stiff', 'Astr_ext', 'A_contrl', 'BsCc', 'BcCs', 'B_feedbk', 'BsDcCs'])

prob.model.add_subsystem('wave_spectrum', WaveSpectrum(freqs=freqs), promotes_inputs=['Hs', 'Tp'], promotes_outputs=['S_wave'])

prob.model.add_subsystem('wind_spectrum', WindSpectrum(freqs=freqs), promotes_inputs=['windspeed_0'], promotes_outputs=['S_wind'])

prob.model.add_subsystem('interp_wave_forces', InterpWaveForces(freqs=freqs), promotes_inputs=['Re_wave_forces', 'Im_wave_forces'], promotes_outputs=[\
	'Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend'])

viscous_group = Viscous(freqs=freqs)

prob.model.add_subsystem('viscous', viscous_group, promotes_inputs=['Cd', 'x_sparelem', 'z_sparnode', 'Z_spar', 'D_spar', 'B_aero_11', 'B_aero_15', \
	'B_aero_17', 'B_aero_55', 'B_aero_57', 'B_aero_77', 'B_struct_77', 'M_global', 'A_global', 'CoG_rotor', 'I_d', 'dtorque_dv', 'dtorque_drotspeed', \
	'Astr_stiff', 'Astr_ext', 'A_contrl', 'BsCc', 'BcCs', 'B_feedbk', 'BsDcCs', 'Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', \
	'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'thrust_wind', 'moment_wind', 'torque_wind', 'S_wave', 'S_wind'], \
	promotes_outputs=['A_feedbk', 'Re_H_feedbk', 'Im_H_feedbk', 'Re_RAO_wave_surge', 'Im_RAO_wave_surge', 'Re_RAO_wave_pitch', 'Im_RAO_wave_pitch', 'Re_RAO_wave_bend', \
	'Im_RAO_wave_bend', 'Re_RAO_wind_surge', 'Im_RAO_wind_surge', 'Re_RAO_wind_pitch', 'Im_RAO_wind_pitch', 'Re_RAO_wind_bend', 'Im_RAO_wind_bend', \
	'Re_RAO_Mwind_surge', 'Im_RAO_Mwind_surge', 'Re_RAO_Mwind_pitch', 'Im_RAO_Mwind_pitch', 'Re_RAO_Mwind_bend', 'Im_RAO_Mwind_bend', 'Re_RAO_wave_vel_surge', \
	'Im_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_bend', 'Re_RAO_wind_vel_surge', \
	'Im_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_bend', 'Re_RAO_Mwind_vel_surge', \
	'Im_RAO_Mwind_vel_surge', 'Re_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_pitch', 'Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_bend', 'B_visc_11', 'stddev_vel_distr'])

postpro_group = Postpro(freqs=freqs)

prob.model.add_subsystem('postpro', postpro_group, promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', \
	'Re_wave_force_bend', 'Im_wave_force_bend', 'thrust_wind', 'moment_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'k_t', 'gain_corr_factor', \
	'S_wave', 'S_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'Re_RAO_wave_surge', 'Im_RAO_wave_surge', 'Re_RAO_wave_pitch', 'Im_RAO_wave_pitch', 'Re_RAO_wave_bend', \
	'Im_RAO_wave_bend', 'Re_RAO_wind_surge', 'Im_RAO_wind_surge', 'Re_RAO_wind_pitch', 'Im_RAO_wind_pitch', 'Re_RAO_wind_bend', 'Im_RAO_wind_bend', \
	'Re_RAO_Mwind_surge', 'Im_RAO_Mwind_surge', 'Re_RAO_Mwind_pitch', 'Im_RAO_Mwind_pitch', 'Re_RAO_Mwind_bend', 'Im_RAO_Mwind_bend', 'Re_RAO_wave_vel_surge', \
	'Im_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_bend', 'Re_RAO_wind_vel_surge', \
	'Im_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_bend', 'Re_RAO_Mwind_vel_surge', \
	'Im_RAO_Mwind_vel_surge', 'Re_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_pitch', 'Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_bend', 'D_tower_p', 'wt_tower_p', \
	'Z_tower', 'dthrust_dv', 'dmoment_dv', 'dthrust_drotspeed', 'dthrust_dbldpitch', 'M_tower', 'M_nacelle', 'M_rotor', 'I_rotor', 'CoG_nacelle', 'CoG_rotor', \
	'z_towernode', 'x_towerelem', 'x_towernode', 'x_d_towertop'], promotes_outputs=['resp_surge', 'resp_tower_moment', 'stddev_surge', 'stddev_pitch', 'stddev_bend', 'stddev_rotspeed', 'stddev_bldpitch', 'fatigue_damage'])

#hull_buckling_group = HullBuckling()

#prob.model.add_subsystem('hull_buckling', hull_buckling_group, promotes_inputs=['D_spar_p', 'wt_spar_p', 'Z_spar', 'M_spar', 'M_ball', 'L_ball', 'spar_draft', 'M_moor', 'z_moor',\
#'dthrust_dv', 'dmoment_dv', 't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'l_stiff', 'angle_hull', 'f_y', 'buck_len'], promotes_outputs=['shell_buckling', 'ring_buckling_1', 'ring_buckling_2', 'col_buckling', 'constr_area_ringstiff', 'constr_hoop_stress', 'constr_mom_inertia_ringstiff'])

#aero_group.linear_solver = LinearRunOnce()
#mooring_group.linear_solver = DirectSolver()
#substructure_group.linear_solver = DirectSolver()
#statespace_group.linear_solver = DirectSolver()
#viscous_group.linear_solver = LinearBlockGS(maxiter=30)
viscous_group.nonlinear_solver = NonlinearBlockGS(maxiter=50, atol=1e-5, rtol=1e-5)
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
print prob['stddev_surge'] #[2.31806726]
print prob['stddev_pitch'] #[0.02006235]
print prob['stddev_bend'] #[0.09510153]
print prob['stddev_rotspeed'] #[0.0941907]
print prob['stddev_bldpitch'] #[0.01095254]
print prob['fatigue_damage'][0] #0.00030961246865275963

#[  2.39647226  35.29681523 115.97636051]
"""
import h5py
def cut_transients(arr,time,dt):
	cut_idx = np.linspace(0,time/dt-1,time/dt)
	arr = np.delete(arr,cut_idx)

	return arr

def fft(x, dt):
	x_fft = np.fft.fft(ssig.detrend(x, type='constant'))
	freqs = np.fft.fftfreq(x_fft.size, dt)
	NFFT = len(freqs)
	freqs = 2. * np.pi*freqs
	x_fft = abs(x_fft)**2. * (dt / NFFT) / (2. * np.pi)
	freqs = freqs[0:NFFT/2]
	x_fft = 2. * x_fft[0:NFFT/2]

	lim1 = 0
	lim2 = 0

	for i in xrange(len(freqs)):
		if freqs[i] > (2.*np.pi): #neglect frequencies higher than 1 Hz
			lim2 = i
			break
	for i in xrange(len(freqs)):
		if freqs[i] > (2.*np.pi / 500.): #neglect frequencies lower than 1/500 Hz
			lim1 = i
			break

	freqs = freqs[lim1:lim2]
	x_fft = x_fft[lim1:lim2]

	return freqs, x_fft

def readfile(file):
	f = h5py.File(file, 'r')

	model = f.keys()[0]
	condSet = f[model].keys()[0]

	surge = np.array(f[model+'/'+condSet+'/Dynamic/spar/Global pos_ (time domain)/XGtranslationTotalmotion'])
	pitch = np.array(f[model+'/'+condSet+'/Dynamic/spar/Global pos_ (time domain)/YLrotationTotalmotion'])
	rotspeed = np.array(f[model+'/'+condSet+'/Dynamic/Wind Turbine/Rotor speed'])
	bldpitch = np.array(f[model+'/'+condSet+'/Dynamic/Wind Turbine/Pitch angle blade 1, Line: bl1foil'])
	#torque = np.array(f[model+'/'+condSet+'/Dynamic/Wind Turbine/Electrical generator torque'])
	My_TB = np.array(f[model+'/'+condSet+'/Dynamic/tower/segment_1/element_1/Mom_ about local y-axis, end 1'])

	return surge, pitch, rotspeed, bldpitch, My_TB

SIMAfile = r'C:\Code\Verification\SparOpt_ver_3_10_21.h5'

surge, pitch, rotspeed, bldpitch, My_TB = readfile(SIMAfile)
rotspeed = rotspeed * np.pi / 180.
bldpitch = bldpitch * np.pi / 180.
pitch = pitch * np.pi / 180.

surge = cut_transients(surge, 200., 0.25)
pitch = cut_transients(pitch, 200., 0.25)
rotspeed = cut_transients(rotspeed, 200., 0.05)
bldpitch = cut_transients(bldpitch, 200., 0.05)
My_TB = cut_transients(My_TB, 200., 0.05)

freqs, TD_fft = fft(My_TB, 0.05)

omega = np.linspace(0.014361566416410483,6.283185307179586,1000)
plt.plot(freqs,TD_fft)
plt.plot(omega,prob['resp_tower_moment'][:,0])
plt.show()
"""