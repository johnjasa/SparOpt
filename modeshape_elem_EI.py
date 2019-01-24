import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeElemEI(ExplicitComponent):

	def setup(self):
		self.add_input('D_tower', val=np.zeros(10), units='m')
		self.add_input('wt_tower', val=np.zeros(10), units='m')

		self.add_output('EI_mode_elem', val=np.zeros(23), units='N*m**2')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_tower = inputs['D_tower']
		wt_tower = inputs['wt_tower']

		N_sparelem = 13
		N_towerelem = 10

		outputs['EI_mode_elem'] = np.zeros(N_sparelem + N_towerelem) #TODO

		for i in xrange(N_sparelem):
			outputs['EI_mode_elem'][i] = 1e15 #np.pi / 64. * (D_spar[i]**4. - (D_spar[i] - 2. * wt_spar[i])**4.) * 2.1e11
		
		for i in xrange(N_towerelem):
			outputs['EI_mode_elem'][N_sparelem+i] = np.pi / 64. * (D_tower[i]**4. - (D_tower[i] - 2. * wt_tower[i])**4.) * 2.1e11

	def compute_partials(self, inputs, partials):
		D_tower = inputs['D_tower']
		wt_tower = inputs['wt_tower']

		N_sparelem = 13
		N_towerelem = 10

		partials['EI_mode_elem', 'D_tower'] = np.zeros((23,10))
		partials['EI_mode_elem', 'wt_tower'] = np.zeros((23,10))

		for i in xrange(N_towerelem):
			partials['EI_mode_elem', 'D_tower'][N_sparelem+i,i] = np.pi / 64. * (4. * D_tower[i]**3. - 4. * (D_tower[i] - 2. * wt_tower[i])**3.) * 2.1e11
			partials['EI_mode_elem', 'wt_tower'][N_sparelem+i,i] = np.pi / 64. * 8. * (D_tower[i] - 2. * wt_tower[i])**3. * 2.1e11