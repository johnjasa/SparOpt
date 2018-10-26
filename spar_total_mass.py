import numpy as np

from openmdao.api import ExplicitComponent

class SparTotalMass(ExplicitComponent):

	def setup(self):
		self.add_input('M_spar', val=np.zeros(10), units='kg')

		self.add_output('tot_M_spar', val=0., units='kg')

		self.declare_partials('tot_M_spar', 'M_spar', val=np.ones((1,10)))

	def compute(self, inputs, outputs):
		outputs['tot_M_spar'] = np.sum(inputs['M_spar'])