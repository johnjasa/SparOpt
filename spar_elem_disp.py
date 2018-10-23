import numpy as np

from openmdao.api import ExplicitComponent

class SparElemDisp(ExplicitComponent):

	def setup(self):
		self.add_input('x_sparnode', val=np.zeros(14), units='m')
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('x_d_sparnode', val=np.zeros(14), units='m/m')

		self.add_output('x_sparelem', val=np.zeros(13), units='m')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		z = inputs['z_sparnode']
		x = inputs['x_sparnode']
		x_d = inputs['x_d_sparnode']

		N_spar = len(z)

		h = np.zeros(N_spar - 1)
		for i in xrange(N_spar - 1):
			h[i] = z[i+1] - z[i]

		outputs['x_sparelem'] = np.zeros(N_spar - 1)
		
		for i in xrange(N_spar - 1):
			outputs['x_sparelem'][i] = (x[i+1] + x[i]) / 2. - 1. / 8. * h[i] * (x_d[i+1] - x_d[i])