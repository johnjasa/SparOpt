import numpy as np

from openmdao.api import ExplicitComponent

class SparElem1Deriv(ExplicitComponent):

	def setup(self):
		self.add_input('x_sparnode', val=np.zeros(14), units='m')
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('x_d_sparnode', val=np.zeros(14), units='m/m')

		self.add_output('x_d_sparelem', val=np.zeros(13), units='1/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		z = inputs['z_sparnode']
		x = inputs['x_sparnode']
		x_d = inputs['x_d_sparnode']

		N_spar = len(z)

		h = np.zeros(N_spar - 1)
		for i in xrange(N_spar - 1):
			h[i] = z[i+1] - z[i]

		outputs['x_d_sparelem'] = np.zeros(N_spar - 1)
		
		for i in xrange(N_spar - 1):
			outputs['x_d_sparelem'][i] = 3. / (2. * h[i]) * (x[i+1] - x[i]) - 1. / 4. * (x_d[i+1] + x_d[i])

	def compute_partials(self, inputs, partials):
		z = inputs['z_sparnode']
		x = inputs['x_sparnode']
		x_d = inputs['x_d_sparnode']

		N_spar = len(z)

		h = np.zeros(N_spar - 1)
		for i in xrange(N_spar - 1):
			h[i] = z[i+1] - z[i]

		partials['x_d_sparelem', 'z_sparnode'] = np.zeros((N_spar - 1, N_spar))
		partials['x_d_sparelem', 'x_sparnode'] = np.zeros((N_spar - 1, N_spar))
		partials['x_d_sparelem', 'x_d_sparnode'] = np.zeros((N_spar - 1, N_spar))
		
		for i in xrange(N_spar - 1):
			partials['x_d_sparelem', 'z_sparnode'][i,i] = 3. / (2. * h[i]**2.) * (x[i+1] - x[i])
			partials['x_d_sparelem', 'z_sparnode'][i,i+1] = -3. / (2. * h[i]**2.) * (x[i+1] - x[i])
		
			partials['x_d_sparelem', 'x_sparnode'][i,i] = -3. / (2. * h[i])
			partials['x_d_sparelem', 'x_sparnode'][i,i+1] = 3. / (2. * h[i])

			partials['x_d_sparelem', 'x_d_sparnode'][i,i] = - 1. / 4.
			partials['x_d_sparelem', 'x_d_sparnode'][i,i+1] = - 1. / 4.