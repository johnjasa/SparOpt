import numpy as np

from openmdao.api import ExplicitComponent

class SparMass(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('L_spar', val=np.zeros(10), units='m')
		self.add_input('wt_spar', val=np.zeros(10), units='m')

		self.add_output('M_spar', val=np.zeros(10), units='kg')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_spar  = inputs['D_spar']
		L_spar  = inputs['L_spar']
		wt_spar  = inputs['wt_spar']

		outputs['M_spar'] = np.zeros(10)

		for i in range(10):
			outputs['M_spar'][i] += np.pi / 4. * (D_spar[i]**2. - (D_spar[i] - 2. * wt_spar[i])**2.) * L_spar[i] * 7850.

	def compute_partials(self, inputs, partials):
		D_spar  = inputs['D_spar']
		L_spar  = inputs['L_spar']
		wt_spar  = inputs['wt_spar']

		partials['M_spar', 'D_spar'] = np.zeros((3,3))
		partials['M_spar', 'L_spar'] = np.zeros((3,3))
		partials['M_spar', 'wt_spar'] = np.zeros((3,3))

		for i in range(3):
			partials['M_spar', 'D_spar'][i,i] = np.pi * wt_spar[i] * L_spar[i] * 7850.
			partials['M_spar', 'L_spar'][i,i] = np.pi / 4. * (D_spar[i]**2. - (D_spar[i] - 2. * wt_spar[i])**2.) * 7850.
			partials['M_spar', 'wt_spar'][i,i] = np.pi * (D_spar[i] - 2. * wt_spar[i]) * L_spar[i] * 7850.