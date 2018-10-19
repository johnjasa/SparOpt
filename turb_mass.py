import numpy as np

from openmdao.api import ExplicitComponent

class TurbMass(ExplicitComponent):

	def setup(self):
		self.add_input('tot_M_tower', val=0., units='kg')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('M_rotor', val=0., units='kg')

		self.add_output('M_turb', val=0., units='kg')

		self.declare_partials('M_turb', 'tot_M_tower', val=1.)
		self.declare_partials('M_turb', 'M_nacelle', val=1.)
		self.declare_partials('M_turb', 'M_rotor', val=1.)

	def compute(self, inputs, outputs):
		outputs['M_turb'] = inputs['tot_M_tower'] + inputs['M_nacelle'] + inputs['M_rotor']