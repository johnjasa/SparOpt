import numpy as np

from openmdao.api import ExplicitComponent

class BallInertia(ExplicitComponent):

	def setup(self):
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('CoG_ball', val=0., units='m')

		self.add_output('I_ball', val=0., units='kg*m**2')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['I_ball'] = inputs['M_ball'] * inputs['CoG_ball']**2.

	def compute_partials(self, inputs, partials):
		partials['I_ball', 'M_ball'] = inputs['CoG_ball']**2.
		partials['I_ball', 'CoG_ball'] = 2. * inputs['M_ball'] * inputs['CoG_ball']