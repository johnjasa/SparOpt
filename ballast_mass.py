import numpy as np

from openmdao.api import ExplicitComponent

class BallastMass(ExplicitComponent):

	def setup(self):
		self.add_input('M_ball_elem', val=np.zeros(10), units='kg')

		self.add_output('M_ball', val=0., units='kg')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['M_ball'] = np.sum(inputs['M_ball_elem'])

	def compute_partials(self, inputs, partials):
		partials['M_ball', 'M_ball_elem'] = np.ones(10)