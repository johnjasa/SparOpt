import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeElemNormforce(ExplicitComponent):

	def setup(self):
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('tot_M_tower', val=0., units='kg')

		self.add_output('normforce_mode_elem', val=np.zeros(10), units='N')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		M_tower = inputs['M_tower']
		M_nacelle = inputs['M_nacelle']
		M_rotor = inputs['M_rotor']
		tot_M_tower = inputs['tot_M_tower']

		N_towerelem = len(M_tower)
		
		for i in xrange(N_towerelem):
			outputs['normforce_mode_elem'][i] = (-M_nacelle - M_rotor - tot_M_tower + np.sum(M_tower[:i])) * 9.80665

	def compute_partials(self, inputs, partials):
		M_tower = inputs['M_tower']
		M_nacelle = inputs['M_nacelle']
		M_rotor = inputs['M_rotor']
		tot_M_tower = inputs['tot_M_tower']

		N_towerelem = len(M_tower)
		
		partials['normforce_mode_elem', 'M_tower'] = np.zeros((10,10))
		partials['normforce_mode_elem', 'M_nacelle'] = np.zeros(10)
		partials['normforce_mode_elem', 'M_rotor'] = np.zeros(10)
		partials['normforce_mode_elem', 'tot_M_tower'] = np.zeros(10)

		for i in xrange(N_towerelem):
			partials['normforce_mode_elem', 'M_nacelle'][i] = -9.80665
			partials['normforce_mode_elem', 'M_rotor'][i] = -9.80665
			partials['normforce_mode_elem', 'tot_M_tower'][i] = -9.80665

			for j in xrange(i):
				partials['normforce_mode_elem', 'M_tower'][i,j] += 9.80665