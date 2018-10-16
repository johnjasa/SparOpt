import numpy as np

from openmdao.api import ExplicitComponent

class SparMass(ExplicitComponent):

	def setup(self):
		self.add_input('D_secs', val=np.zeros(3), units='m')
		self.add_input('L_secs', val=np.zeros(3), units='m')
		self.add_input('wt_secs', val=np.zeros(3), units='m')

		self.add_output('M_secs', val=np.zeros(3), units='kg')

	def compute(self, inputs, outputs):
		D_secs  = inputs['D_secs']
		L_secs  = inputs['L_secs']
		wt_secs  = inputs['wt_secs']

		outputs['M_secs'] = np.zeros(3)

		for i in range(3):
			outputs['M_secs'][i] += np.pi / 4. * (D_secs[i]**2. - (D_secs[i] - 2. * wt_secs[i])**2.) * L_secs[i] * 7850.