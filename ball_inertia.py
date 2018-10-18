import numpy as np

from openmdao.api import ExplicitComponent

class BallInertia(ExplicitComponent):

	def setup(self):
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('CoG_ball', val=0., units='m')

		self.add_output('I_ball', val=0., units='kg*m**2')

	def compute(self, inputs, outputs):
		outputs['I_ball'] = inputs['M_ball'] * inputs['CoG_ball']**2.