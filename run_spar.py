import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp

from substructure_group import Substructure
from aero_group import Aero
from statespace_group import StateSpace
from wave_spectrum import WaveSpectrum
from wind_spectrum import WindSpectrum

blades = {\
'Rtip' : 89.165, \
'Rhub' : 2.8, \
'N_b_elem' : 20, \
'indfile' : 'M:/PhD/Integrated optimization/Aerodynamics/BEM/BEMpy/DTU10MW_indfacs.dat', \
'bladefile' : 'M:/PhD/Integrated optimization/Aerodynamics/BEM/BEMpy/DTU10MWblade.dat', \
'foilnames' : ['foil1', 'foil11', 'foil12', 'foil13', 'foil14', 'foil15', 'foil16', 'foil17', 'foil18', 'foil19'], \
'foilfolder' : 'M:/PhD/Integrated optimization/Aerodynamics/BEM/BEMpy/Airfoils/', \
'cohfolder' : 'M:/PhD/Integrated optimization/Aerodynamics/CoherenceCoeffs/'}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('D_secs', val=np.array([12., np.sqrt(1./3. * (12.**2. + 8.3**2. + 12. * 8.3)), 8.3]), units='m')
ivc.add_output('L_secs', val=np.array([108., 8., 14.]), units='m')
ivc.add_output('Z_spar', val=np.array([-120., -12., -4., 10.]), units='m')
ivc.add_output('spar_draft', val=120., units='m')
ivc.add_output('wt_secs', val=np.array([0.06, 0.06, 0.06]), units='m')
ivc.add_output('D_tower', val=np.array([8.16083499,7.88250497, 7.60417495, 7.32584493, 7.04751491, 6.76918489, 6.49085487, 6.21252485, 5.93419483, 5.64751491]), units='m')
ivc.add_output('L_tower', val=np.array([10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 11.13]), units='m')
ivc.add_output('Z_tower', val=np.array([10.0, 20.5, 31.0, 41.5, 52.0, 62.5, 73.0, 83.5, 94.0, 104.5, 115.63]), units='m')
ivc.add_output('wt_tower', val=np.array([0.038, 0.036, 0.034, 0.032, 0.030, 0.028, 0.026, 0.024, 0.022, 0.020]), units='m')
ivc.add_output('rho_ball', val=2600., units='kg/m**3')
ivc.add_output('wt_ball', val=0.06, units='m')
ivc.add_output('M_nacelle', val=4.46e5, units='kg')
ivc.add_output('CoG_rotor', val=119., units='m')
ivc.add_output('CoG_nacelle', val=118.08, units='m')
ivc.add_output('I_rotor', val=7.808e7, units='kg*m**2')
ivc.add_output('M_rotor', val=2.307e5, units='kg')
ivc.add_output('omega_wave', val=2. * np.pi / np.linspace(40.,1.,80), units='rad/s')
ivc.add_output('omega', val=np.linspace(0.014361566416410483,6.283185307179586,3493), units='rad/s')
ivc.add_output('water_depth', val=320., units='m')
ivc.add_output('z_moor', val=-77.2, units='m')
ivc.add_output('rho_wind', val=1.25, units='kg/m**3')
ivc.add_output('windspeed_0', val=21., units='m/s')
ivc.add_output('bldpitch_0', val=19.08*np.pi/180., units='rad')
ivc.add_output('rotspeed_0', val=9.6*2.*np.pi/60., units='rad/s')
ivc.add_output('I_d', val=160234250.0, units='kg*m**2')
ivc.add_output('Hs', val=3., units='m')
ivc.add_output('Tp', val=10., units='s')

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

aero_group = Aero(blades=blades)

prob.model.add_subsystem('aero', aero_group, promotes_inputs=['rho_wind', 'windspeed_0', 'bldpitch_0', 'rotspeed_0', 'omega'], promotes_outputs=['thrust_wind', \
	'moment_wind', 'torque_wind', 'thrust_0', 'torque_0', 'dthrust_dv', 'dmoment_dv', 'dtorque_dv', 'dthrust_drotspeed', 'dtorque_drotspeed', \
	'dthrust_dbldpitch', 'dtorque_dbldpitch'])

substructure_group = Substructure()

prob.model.add_subsystem('substructure', substructure_group, promotes_inputs=['D_secs', 'L_secs', 'wt_secs', 'Z_spar', 'D_tower', 'L_tower', 'wt_tower', 'Z_tower', 'rho_ball', 'wt_ball', \
	'M_nacelle', 'M_rotor', 'CoG_nacelle', 'CoG_rotor', 'I_rotor', \
	'omega_wave', 'water_depth', 'z_moor', 'spar_draft', 'dthrust_dv', 'dmoment_dv'], promotes_outputs=['M_global', 'A_global', 'B_global', 'K_global', 'Re_wave_forces', 'Im_wave_forces', \
	'x_towermode', 'z_towermode'])

statespace_group = StateSpace()

prob.model.add_subsystem('statespace', statespace_group, promotes_inputs=['M_global', 'A_global', 'B_global', 'K_global', 'CoG_rotor', 'I_d', 'dthrust_dv', 'dmoment_dv', \
	'dtorque_dv', 'dthrust_drotspeed', 'dtorque_drotspeed', 'dthrust_dbldpitch', 'dtorque_dbldpitch', 'omega_lowpass', 'k_i', 'k_p', 'omega', 'x_towermode', 'z_towermode', 'Z_tower'], \
	promotes_outputs=['A_struct', 'Re_H_feedbk', 'Im_H_feedbk'])

prob.model.add_subsystem('wave_spectrum', WaveSpectrum(), promotes_inputs=['Hs', 'Tp', 'omega'], promotes_outputs=['S_wave'])

prob.model.add_subsystem('wind_spectrum', WindSpectrum(), promotes_inputs=['windspeed_0', 'omega'], promotes_outputs=['S_wind'])

prob.setup()

prob.run_model()
#print 2*np.pi / np.sqrt(np.linalg.eig(np.dot(np.linalg.inv(prob['M_global']+prob['A_global']),(prob['K_global'])))[0])
#print prob['M_global'] + prob['A_global'], prob['K_global']
#print prob['A_struct']

omega = prob['omega']
mag = np.abs(prob['Re_H_feedbk'] + 1j * prob['Im_H_feedbk'])
phase = np.angle(prob['Re_H_feedbk'] + 1j * prob['Im_H_feedbk'])

Xcal = prob['Re_wave_forces'] + 1j * prob['Im_wave_forces']

Xcal1_FD = np.interp(omega, prob['omega_wave'], Xcal[:,0,0])
Xcal5_FD = np.interp(omega, prob['omega_wave'], Xcal[:,1,0])
Xcal7_FD = np.interp(omega, prob['omega_wave'], Xcal[:,2,0])

RAO_wave_surge = mag[:,0,3] * np.exp(1j * phase[:,0,3]) * Xcal1_FD + mag[:,0,4] * np.exp(1j * phase[:,0,4]) * Xcal5_FD + mag[:,0,5] * np.exp(1j * phase[:,0,5]) * Xcal7_FD
RAO_wave_pitch = mag[:,1,3] * np.exp(1j * phase[:,1,3]) * Xcal1_FD + mag[:,1,4] * np.exp(1j * phase[:,1,4]) * Xcal5_FD + mag[:,1,5] * np.exp(1j * phase[:,1,5]) * Xcal7_FD
RAO_wave_bend = mag[:,2,3] * np.exp(1j * phase[:,2,3]) * Xcal1_FD + mag[:,2,4] * np.exp(1j * phase[:,2,4]) * Xcal5_FD + mag[:,2,5] * np.exp(1j * phase[:,2,5]) * Xcal7_FD
RAO_wave_rotspeed = mag[:,3,3] * np.exp(1j * phase[:,3,3]) * Xcal1_FD + mag[:,3,4] * np.exp(1j * phase[:,3,4]) * Xcal5_FD + mag[:,3,5] * np.exp(1j * phase[:,3,5]) * Xcal7_FD

RAO_wind_surge = mag[:,0,0] * np.exp(1j * phase[:,0,0]) * prob['thrust_wind'] + mag[:,0,1] * np.exp(1j * phase[:,0,1]) * prob['moment_wind'] + mag[:,0,2] * np.exp(1j * phase[:,0,2]) * prob['torque_wind']
RAO_wind_pitch = mag[:,1,0] * np.exp(1j * phase[:,1,0]) * prob['thrust_wind'] + mag[:,1,1] * np.exp(1j * phase[:,1,1]) * prob['moment_wind'] + mag[:,1,2] * np.exp(1j * phase[:,1,2]) * prob['torque_wind']
RAO_wind_bend = mag[:,2,0] * np.exp(1j * phase[:,2,0]) * prob['thrust_wind'] + mag[:,2,1] * np.exp(1j * phase[:,2,1]) * prob['moment_wind'] + mag[:,2,2] * np.exp(1j * phase[:,2,2]) * prob['torque_wind']
RAO_wind_rotspeed = mag[:,3,0] * np.exp(1j * phase[:,3,0]) * prob['thrust_wind'] + mag[:,3,1] * np.exp(1j * phase[:,3,1]) * prob['moment_wind'] + mag[:,3,2] * np.exp(1j * phase[:,3,2]) * prob['torque_wind']

X1_lin_FD = np.abs(RAO_wave_surge)**2. * prob['S_wave'] + np.abs(RAO_wind_surge)**2. * prob['S_wind']

X5_lin_FD = np.abs(RAO_wave_pitch)**2. * prob['S_wave'] + np.abs(RAO_wind_pitch)**2. * prob['S_wind']

phi_dot_lin_FD = np.abs(RAO_wave_rotspeed)**2. * prob['S_wave'] + np.abs(RAO_wind_rotspeed)**2. * prob['S_wind']

print np.std(X1_lin_FD), np.std(X5_lin_FD), np.std(phi_dot_lin_FD)

"""
#plt.plot(omega,mag[:,6,3], label='X1')
#plt.plot(omega,mag[:,6,4], label='X5')
#plt.plot(omega,mag[:,6,5], label='X7')
#plt.plot(omega,mag[:,6,0], label='Vf')
#plt.plot(omega,mag[:,6,1], label='Vm')
plt.plot(omega,mag[:,6,2], label='Vq')
plt.legend()
plt.show()
"""