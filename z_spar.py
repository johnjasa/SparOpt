import numpy as np

from openmdao.api import ExplicitComponent

class ZSpar(ExplicitComponent):

	def setup(self):
		self.add_input('L_spar', val=np.zeros(10), units='m')
		self.add_input('spar_draft', val=0., units='m')

		self.add_output('Z_spar', val=np.zeros(11), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		L_spar = inputs['L_spar']
		spar_draft = inputs['spar_draft']
		
		outputs['Z_spar'] = np.zeros(len(L_spar) + 1)

		for i in xrange(len(L_spar)):
			outputs['Z_spar'][i] = -spar_draft + np.sum(L_spar[:i]) 

		outputs['Z_spar'][-1] = 10.

	def compute_partials(self, inputs, partials):
		partials['Z_spar', 'L_spar'] = np.concatenate((np.tril(np.ones((10,10)),-1),np.zeros((1,10))),0)
		partials['Z_spar', 'spar_draft'] = np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., 0.])