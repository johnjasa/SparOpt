import numpy as np

from openmdao.api import ExplicitComponent

class BallastLen(ExplicitComponent):

	def setup(self):
		self.add_input('L_ball_elem', val=np.zeros(10), units='m')

		self.add_output('L_ball', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['L_ball'] = np.sum(inputs['L_ball_elem'])

	def compute_partials(self, inputs, partials):
		partials['L_ball', 'L_ball_elem'] = np.ones(10)