import numpy as np
from scipy.sparse import linalg
from scipy.linalg import det

from openmdao.api import ExplicitComponent

class ModeshapeDisp(ExplicitComponent):

	def setup(self):
		self.add_input('eig_vector', val=np.zeros(48), units='m')

		self.add_output('x_sparnode', val=np.zeros(14), units='m')
		self.add_output('x_towernode', val=np.zeros(11), units='m')

	def compute(self, inputs, outputs):
		N_sparelem = 13
		N_elem = 23

		x_sparnode = inputs['eig_vector'][0:(N_sparelem+1)*2:2]
		x_towernode = inputs['eig_vector'][(N_sparelem+1)*2-2:(N_elem+1)*2:2]

		outputs['x_sparnode'] = x_sparnode / x_towernode[-1]
		outputs['x_towernode'] = x_towernode / x_towernode[-1]