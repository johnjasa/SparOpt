import numpy as np

from openmdao.api import ExplicitComponent

class Draft(ExplicitComponent):

	def setup(self):
		self.add_input('L_spar', val=np.zeros(10), units='m')

		self.add_output('spar_draft', val=0., units='m')

		self.declare_partials('spar_draft', 'L_spar', val=np.ones((1,10)))

	def compute(self, inputs, outputs):
		outputs['spar_draft'] = np.sum(inputs['L_spar']) - 10.