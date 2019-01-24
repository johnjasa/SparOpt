import numpy as np

from openmdao.api import ExplicitComponent

class SparThickness(ExplicitComponent):

	def setup(self):
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')

		self.add_output('wt_spar', val=np.zeros(10), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		wt_spar_p  = inputs['wt_spar_p']

		outputs['wt_spar'] = np.zeros(10)

		for i in range(len(wt_spar_p)-1):
			outputs['wt_spar'][i] = (wt_spar_p[i] + wt_spar_p[i+1]) / 2.

	def compute_partials(self, inputs, partials):
		wt_spar_p  = inputs['wt_spar_p']

		partials['wt_spar', 'wt_spar_p'] = np.zeros((len(wt_spar_p)-1,len(wt_spar_p)))

		for i in range(len(wt_spar_p)-1):
			partials['wt_spar', 'wt_spar_p'][i,i] += 0.5
			partials['wt_spar', 'wt_spar_p'][i,i+1] += 0.5