import numpy as np

from openmdao.api import ExplicitComponent

class TowerMass(ExplicitComponent):

	def setup(self):
		self.add_input('D_tower', val=np.zeros(10), units='m')
		self.add_input('L_tower', val=np.zeros(10), units='m')
		self.add_input('wt_tower', val=np.zeros(10), units='m')

		self.add_output('M_tower', val=np.zeros(10), units='kg')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_tower  = inputs['D_tower']
		L_tower  = inputs['L_tower']
		wt_tower  = inputs['wt_tower']

		outputs['M_tower'] = np.zeros(10)

		for i in range(10):
			outputs['M_tower'][i] += np.pi / 4. * (D_tower[i]**2. - (D_tower[i] - 2. * wt_tower[i])**2.) * L_tower[i] * 8500. #includes secondary structures in density

	def compute_partials(self, inputs, partials):
		D_tower  = inputs['D_tower']
		L_tower  = inputs['L_tower']
		wt_tower  = inputs['wt_tower']

		partials['M_tower', 'D_tower'] = np.zeros((len(D_tower),len(D_tower)))
		partials['M_tower', 'L_tower'] = np.zeros((len(D_tower),len(D_tower)))
		partials['M_tower', 'wt_tower'] = np.zeros((len(D_tower),len(D_tower)))

		for i in range(10):
			partials['M_tower', 'D_tower'][i,i] = np.pi * wt_tower[i] * L_tower[i] * 8500.
			partials['M_tower', 'L_tower'][i,i] = np.pi / 4. * (D_tower[i]**2. - (D_tower[i] - 2. * wt_tower[i])**2.) * 8500.
			partials['M_tower', 'wt_tower'][i,i] = np.pi * (D_tower[i] - 2. * wt_tower[i]) * L_tower[i] * 8500.