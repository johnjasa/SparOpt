import numpy as np

from openmdao.api import ExplicitComponent

class TurbCoG(ExplicitComponent):

	def setup(self):
		self.add_input('tot_M_tower', val=0., units='kg')
		self.add_input('CoG_tower', val=0., units='m')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('CoG_nacelle', val=0., units='m')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('M_turb', val=1., units='kg')

		self.add_output('CoG_turb', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['CoG_turb'] = (inputs['tot_M_tower'] * inputs['CoG_tower'] + inputs['M_nacelle'] * inputs['CoG_nacelle'] + inputs['M_rotor'] * inputs['CoG_rotor']) / inputs['M_turb']

	def compute_partials(self, inputs, partials):
		partials['CoG_turb', 'tot_M_tower'] = inputs['CoG_tower'] / inputs['M_turb']
		partials['CoG_turb', 'CoG_tower'] = inputs['tot_M_tower'] / inputs['M_turb']
		partials['CoG_turb', 'M_nacelle'] = inputs['CoG_nacelle'] / inputs['M_turb']
		partials['CoG_turb', 'CoG_nacelle'] = inputs['M_nacelle'] / inputs['M_turb']
		partials['CoG_turb', 'M_rotor'] = inputs['CoG_rotor'] / inputs['M_turb']
		partials['CoG_turb', 'CoG_rotor'] = inputs['M_rotor'] / inputs['M_turb']
		partials['CoG_turb', 'M_turb'] = -(inputs['tot_M_tower'] * inputs['CoG_tower'] + inputs['M_nacelle'] * inputs['CoG_nacelle'] + inputs['M_rotor'] * inputs['CoG_rotor']) / inputs['M_turb']**2.