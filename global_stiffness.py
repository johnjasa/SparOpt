import numpy as np

from openmdao.api import ExplicitComponent

class GlobalStiffness(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(3), units='m')
		self.add_input('tot_M_spar', val=0., units='kg')
		self.add_input('tot_M_tower', val=0., units='kg')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('M_turb', val=0., units='kg')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('CoG_spar', val=0., units='m')
		self.add_input('CoG_tower', val=0., units='m')
		self.add_input('CoG_nacelle', val=0., units='m')
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('CoG_ball', val=0., units='m')

		self.add_input('M_moor', val=0., units='kg')
		self.add_input('K_moor', val=0., units='N/m')

		self.add_input('K17', val=0., units='N/m')
		self.add_input('K57', val=0., units='N')
		self.add_input('K77', val=0., units='N/m')

		self.add_input('buoy_spar', val=0., units='N')
		self.add_input('CoB', val=0., units='m')

		self.add_output('K_global', val=np.zeros((3,3)), units='N/m')

	def compute(self, inputs, outputs):
		D_spar = inputs['D_spar']
		tot_M_spar = inputs['tot_M_spar']
		M_turb = inputs['M_turb']
		M_ball = inputs['M_ball'] 
		CoG_spar = inputs['CoG_spar']
		CoG_turb = (inputs['tot_M_tower'] * inputs['CoG_tower'] + inputs['M_nacelle'] * inputs['CoG_nacelle'] + inputs['M_rotor'] * inputs['CoG_rotor']) / M_turb
		CoG_ball = inputs['CoG_ball']
		M_moor = inputs['M_moor']
		K_moor = inputs['K_moor']
		K17 = inputs['K17']
		K57 = inputs['K57']
		K77 = inputs['K77']
		buoy_spar = inputs['buoy_spar']
		CoB = inputs['CoB']

		CoG_tot = (M_turb * CoG_turb + tot_M_spar * CoG_spar + M_ball * CoG_ball) / (tot_M_spar + M_turb + M_ball)
		hydrostatic_pitch = buoy_spar * CoB - tot_M_spar * 9.80665 * CoG_spar - M_ball * 9.80665 * CoG_ball - M_turb * 9.80665 * CoG_turb + 1025. * 9.80665 * np.pi / 64. * D_spar[-1]**4.

		outputs['K_global'] = np.zeros((3,3))

		outputs['K_global'][0,0] += K_moor
		outputs['K_global'][0,1] += K_moor * CoG_tot
		outputs['K_global'][0,2] += K17
		outputs['K_global'][1,0] += K_moor * CoG_tot
		outputs['K_global'][1,1] += K_moor * CoG_tot**2. - M_moor * 9.80665 * CoG_tot + hydrostatic_pitch
		outputs['K_global'][1,2] += K57
		outputs['K_global'][2,0] += K17
		outputs['K_global'][2,1] += K57
		outputs['K_global'][2,2] += K77