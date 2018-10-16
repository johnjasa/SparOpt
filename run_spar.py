import numpy as np

from openmdao.api import Problem, IndepVarComp

from substructure_group import Substructure
from aero_group import Aero
from statespace_group import StateSpace

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
ivc.add_output('D_secs', val=np.array([8.3, np.sqrt(1./3. * (12.**2. + 8.3**2. + 12. * 8.3)), 12.]), units='m')
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

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

aero_group = Aero(blades=blades)

prob.model.add_subsystem('aero', aero_group, promotes_inputs=['rho_wind', 'windspeed_0', 'bldpitch_0', 'rotspeed_0', 'omega'], promotes_outputs=['thrust_wind', \
	'moment_wind', 'torque_wind', 'thrust_0', 'torque_0', 'dthrust_dv', 'dmoment_dv', 'dtorque_dv', 'dthrust_drotspeed', 'dtorque_drotspeed', \
	'dthrust_dbldpitch', 'dtorque_dbldpitch'])

substructure_group = Substructure()

prob.model.add_subsystem('substructure', substructure_group, promotes_inputs=['D_secs', 'L_secs', 'wt_secs', 'Z_spar', 'D_tower', 'L_tower', 'wt_tower', 'Z_tower', 'rho_ball', 'wt_ball', \
	'M_nacelle', 'M_rotor', 'CoG_nacelle', 'CoG_rotor', 'I_rotor', \
	'omega_wave', 'water_depth', 'z_moor', 'spar_draft'], promotes_outputs=['M_global', 'A_global', 'B_global', 'K_global', 'Re_wave_forces', 'Im_wave_forces'])

statespace_group = StateSpace()

prob.model.add_subsystem('statespace', statespace_group, promotes_inputs=['M_global', 'A_global', 'B_global', 'K_global', 'CoG_rotor', 'I_d', 'dthrust_dv', 'dmoment_dv', \
	'dtorque_dv', 'dthrust_drotspeed', 'dtorque_drotspeed', 'dthrust_dbldpitch', 'dtorque_dbldpitch', 'omega_lowpass', 'k_i', 'k_p', 'omega'], promotes_outputs=['Re_H_feedbk', 'Im_H_feedbk'])

prob.setup()

prob.run_model()

print prob['Re_H_feedbk'] + 1j * prob['Im_H_feedbk']