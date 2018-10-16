import numpy as np

from openmdao.api import ExplicitComponent

class GlobalMass(ExplicitComponent):

	def setup(self):
		self.add_input('M_secs', val=np.zeros(3), units='kg')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('CoG_spar', val=0., units='m')
		self.add_input('CoG_ball', val=0., units='m')
		self.add_input('CoG_tower', val=0., units='m')
		self.add_input('CoG_nacelle', val=0., units='m')
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('I_spar', val=0., units='kg*m**2')
		self.add_input('I_tower', val=0., units='kg*m**2')
		self.add_input('I_rotor', val=0., units='kg*m**2')

		self.add_input('M17', val=0., units='kg') #TODO: calculate these for bending mode
		self.add_input('M57', val=0., units='kg*m')
		self.add_input('M77', val=0., units='kg')

		self.add_output('M_global', val=np.zeros((3,3)), units='kg')

	def compute(self, inputs, outputs):
		M_spar = np.sum(inputs['M_secs'])
		M_ball = np.sum(inputs['M_ball'])
		M_turb = np.sum(inputs['M_tower']) + inputs['M_nacelle'] + inputs['M_rotor']
		CoG_spar = inputs['CoG_spar']
		CoG_ball = inputs['CoG_ball']
		CoG_turb = (np.sum(inputs['M_tower']) * inputs['CoG_tower'] + inputs['M_nacelle'] * inputs['CoG_nacelle'] + inputs['M_rotor'] * inputs['CoG_rotor']) / M_turb
		I_spar = inputs['I_spar']
		I_turb = inputs['I_tower'] + inputs['M_nacelle'] * inputs['CoG_nacelle']**2. + inputs['M_rotor'] * inputs['CoG_rotor']**2. + inputs['I_rotor']
		M17 = inputs['M17']
		M57 = inputs['M57']
		M77 = inputs['M77']

		outputs['M_global'] = np.zeros((3,3))

		outputs['M_global'][0,0] += M_turb + M_spar + M_ball
		outputs['M_global'][0,1] += M_turb * CoG_turb + M_spar * CoG_spar + M_ball * CoG_ball
		outputs['M_global'][0,2] += M17
		outputs['M_global'][1,0] += M_turb * CoG_turb + M_spar * CoG_spar  + M_ball * CoG_ball
		outputs['M_global'][1,1] += I_turb + I_spar
		outputs['M_global'][1,2] += M57
		outputs['M_global'][2,0] += M17
		outputs['M_global'][2,1] += M57
		outputs['M_global'][2,2] += M77