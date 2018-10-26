import numpy as np

from openmdao.api import ExplicitComponent

class TurbInertia(ExplicitComponent):

	def setup(self):
		self.add_input('I_tower', val=0., units='kg*m**2')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('CoG_nacelle', val=0., units='m')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('I_rotor', val=0., units='kg*m**2')

		self.add_output('I_turb', val=0., units='kg*m**2')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['I_turb'] = inputs['I_tower'] + inputs['M_nacelle'] * inputs['CoG_nacelle']**2. + inputs['M_rotor'] * inputs['CoG_rotor']**2. + inputs['I_rotor']

	def compute_partials(self, inputs, partials):
		partials['I_turb', 'I_tower'] = 1.
		partials['I_turb', 'M_nacelle'] = inputs['CoG_nacelle']**2.
		partials['I_turb', 'CoG_nacelle'] = 2. * inputs['M_nacelle'] * inputs['CoG_nacelle']
		partials['I_turb', 'M_rotor'] = inputs['CoG_rotor']**2.
		partials['I_turb', 'CoG_rotor'] = 2. * inputs['M_rotor'] * inputs['CoG_rotor']
		partials['I_turb', 'I_rotor'] = 1.