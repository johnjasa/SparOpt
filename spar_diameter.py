import numpy as np

from openmdao.api import ExplicitComponent

class SparDiameter(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar_p', val=np.zeros(11), units='m')

		self.add_output('D_spar', val=np.zeros(10), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_spar_p  = inputs['D_spar_p']

		outputs['D_spar'] = np.zeros(10)

		for i in range(len(D_spar)):
			outputs['D_spar'][i] = (D_spar_p[i] + D_spar_p[i+1]) / 2.

	def compute_partials(self, inputs, partials): #TODO check
		D_spar  = inputs['D_spar']

		partials['D_spar', 'D_spar_p'] = np.zeros((len(D_spar),len(D_spar_p)))

		for i in range(len(D_spar)):
			partials['D_spar', 'D_spar_p'][i,i] += 0.5
			partials['D_spar', 'D_spar_p'][i,i+1] += 0.5