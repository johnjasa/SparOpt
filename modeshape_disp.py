import numpy as np
from scipy.sparse import linalg
from scipy.linalg import det

from openmdao.api import ExplicitComponent

class ModeshapeDisp(ExplicitComponent):

	def setup(self):
		self.add_input('eig_vector', val=np.zeros(48), units='m')

		self.add_output('x_sparnode', val=np.zeros(14), units='m')
		self.add_output('x_towernode', val=np.zeros(11), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		N_sparelem = 13
		N_towerelem = 10
		N_elem = N_sparelem + N_towerelem

		x_sparnode = inputs['eig_vector'][0:(N_sparelem+1)*2:2]
		x_towernode = inputs['eig_vector'][(N_sparelem+1)*2-2:(N_elem+1)*2:2]

		outputs['x_sparnode'] = x_sparnode / x_towernode[-1]
		outputs['x_towernode'] = x_towernode / x_towernode[-1]

	def compute_partials(self, inputs, partials):
		N_sparelem = 13
		N_towerelem = 10
		N_elem = N_sparelem + N_towerelem

		x_sparnode = inputs['eig_vector'][0:(N_sparelem+1)*2:2]
		x_towernode = inputs['eig_vector'][(N_sparelem+1)*2-2:(N_elem+1)*2:2]

		partials['x_sparnode', 'eig_vector'] = np.zeros((14,48))
		partials['x_towernode', 'eig_vector'] = np.zeros((11,48))

		for i in xrange(N_sparelem+1):
			partials['x_sparnode', 'eig_vector'][i,2*i] += 1. / x_towernode[-1]
			partials['x_sparnode', 'eig_vector'][i,-2] += -x_sparnode[i] / x_towernode[-1]**2.

			if i < (N_towerelem+1):
				partials['x_towernode', 'eig_vector'][i,(N_sparelem+1)*2+2*(i-1)] += 1. / x_towernode[-1]
				partials['x_towernode', 'eig_vector'][i,-2] += -x_towernode[i] / x_towernode[-1]**2.