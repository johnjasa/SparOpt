import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeElemLength(ExplicitComponent):

	def setup(self):
		self.add_input('z_sparnode', val=np.zeros(13), units='m')
		self.add_input('z_towernode', val=np.zeros(11), units='m')

		self.add_output('L_mode_elem', val=np.zeros(22), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		z_sparnode = inputs['z_sparnode']
		z_towernode = inputs['z_towernode']

		N_sparelem = len(z_sparnode) - 1
		N_towerelem = len(z_towernode) - 1

		outputs['L_mode_elem'] = np.zeros(N_sparelem + N_towerelem)

		for i in xrange(N_sparelem):
			outputs['L_mode_elem'][i] = z_sparnode[i+1] - z_sparnode[i]
		
		for i in xrange(N_towerelem):
			outputs['L_mode_elem'][N_sparelem+i] = z_towernode[i+1] - z_towernode[i]

	def compute_partials(self, inputs, partials):
		z_sparnode = inputs['z_sparnode']
		z_towernode = inputs['z_towernode']

		N_sparelem = len(z_sparnode) - 1
		N_towerelem = len(z_towernode) - 1

		partials['L_mode_elem', 'z_sparnode'] = np.zeros((22,13))
		partials['L_mode_elem', 'z_towernode'] = np.zeros((22,11))

		for i in xrange(N_sparelem):
			partials['L_mode_elem', 'z_sparnode'][i,i] = -1.
			partials['L_mode_elem', 'z_sparnode'][i,i+1] = 1.

		for i in xrange(N_towerelem):
			partials['L_mode_elem', 'z_towernode'][N_sparelem+i,i] = -1.
			partials['L_mode_elem', 'z_towernode'][N_sparelem+i,i+1] = 1.