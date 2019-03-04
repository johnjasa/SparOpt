import numpy as np

from openmdao.api import ExplicitComponent

class SparNode1RHS(ExplicitComponent):

	def setup(self):
		self.add_input('z_sparnode', val=np.zeros(13), units='m')
		self.add_input('x_sparnode', val=np.zeros(13), units='m')

		self.add_output('spar_spline_rhs', val=np.zeros(13), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		z = inputs['z_sparnode']
		x = inputs['x_sparnode']

		N_spar = len(z)

		h = np.zeros(N_spar - 1)
		delta = np.zeros(N_spar - 1)
		for i in xrange(N_spar - 1):
			h[i] = z[i+1] - z[i]
			delta[i] = (x[i+1] - x[i]) / (z[i+1] - z[i])

		outputs['spar_spline_rhs'] = np.zeros(N_spar)
		
		for i in xrange(1,N_spar - 1):
			outputs['spar_spline_rhs'][i] = 3. * (h[i-1] * delta[i] + h[i] * delta[i-1])

		outputs['spar_spline_rhs'][0] = ((2. * h[1] + 3. * h[0]) * h[1] * delta[0] + h[0]**2. * delta[1]) / (h[0] + h[1])
		outputs['spar_spline_rhs'][-1] = ((2. * h[-2] + 3. * h[-1]) * h[-2] * delta[-1] + h[-1]**2. * delta[-2]) / (h[-1] + h[-2])

	def compute_partials(self, inputs, partials):
		z = inputs['z_sparnode']
		x = inputs['x_sparnode']

		N_spar = len(z)

		h = np.zeros(N_spar - 1)
		delta = np.zeros(N_spar - 1)
		for i in xrange(N_spar - 1):
			h[i] = z[i+1] - z[i]
			delta[i] = (x[i+1] - x[i]) / (z[i+1] - z[i])

		partials['spar_spline_rhs', 'z_sparnode'] = np.zeros((N_spar,N_spar))
		partials['spar_spline_rhs', 'x_sparnode'] = np.zeros((N_spar,N_spar))

		for i in xrange(1,N_spar - 1):
			partials['spar_spline_rhs', 'z_sparnode'][i,i-1] = -3. * delta[i] + 3. * h[i] * delta[i-1] / h[i-1]
			partials['spar_spline_rhs', 'z_sparnode'][i,i] = 3. * (delta[i] + h[i-1] * delta[i] / h[i] - delta[i-1] - h[i] * delta[i-1] / h[i-1])
			partials['spar_spline_rhs', 'z_sparnode'][i,i+1] = -3. * h[i-1] * delta[i] / h[i] + 3. * delta[i-1]

			partials['spar_spline_rhs', 'x_sparnode'][i,i-1] = -3. * h[i] / h[i-1]
			partials['spar_spline_rhs', 'x_sparnode'][i,i] = -3. * h[i-1] / h[i] + 3. * h[i] / h[i-1]
			partials['spar_spline_rhs', 'x_sparnode'][i,i+1] = 3. * h[i-1] / h[i]

		partials['spar_spline_rhs', 'z_sparnode'][0,0] = -(3. * h[1] * delta[0] + 2. * h[0] * delta[1]) / (h[0] + h[1]) + ((2. * h[1] + 3. * h[0]) * h[1] * delta[0] + h[0]**2. * delta[1]) / (h[0] + h[1])**2. + (2. * h[1] + 3. * h[0]) * h[1] / (h[0] + h[1]) * delta[0] / h[0]
		partials['spar_spline_rhs', 'z_sparnode'][0,1] = (3. * h[1] * delta[0] + 2. * h[0] * delta[1]) / (h[0] + h[1]) - (4. * h[1] + 3. * h[0]) * delta[0] / (h[0] + h[1]) - (2. * h[1] + 3 * h[0]) * h[1] / (h[0] + h[1]) * delta[0] / h[0] + h[0]**2. / (h[0] + h[1]) * delta[1] / h[1]
		partials['spar_spline_rhs', 'z_sparnode'][0,2] = (4. * h[1] + 3. * h[0]) * delta[0] / (h[0] + h[1]) - ((2. * h[1] + 3. * h[0]) * h[1] * delta[0] + h[0]**2. * delta[1]) / (h[0] + h[1])**2. - h[0]**2. / (h[0] + h[1]) * delta[1] / h[1]

		partials['spar_spline_rhs', 'x_sparnode'][0,0] = -(2. * h[1] + 3. * h[0]) * h[1] / (h[0] + h[1]) * 1. / h[0]
		partials['spar_spline_rhs', 'x_sparnode'][0,1] = (2. * h[1] + 3 * h[0]) * h[1] / (h[0] + h[1]) * 1. / h[0] - h[0]**2. / (h[0] + h[1]) * 1. / h[1]
		partials['spar_spline_rhs', 'x_sparnode'][0,2] = h[0]**2. / (h[0] + h[1]) * 1. / h[1]

		partials['spar_spline_rhs', 'z_sparnode'][-1,-1] = (3. * h[-2] * delta[-1] + 2. * h[-1] * delta[-2]) / (h[-1] + h[-2]) - ((2. * h[-2] + 3. * h[-1]) * h[-2] * delta[-1] + h[-1]**2. * delta[-2]) / (h[-1] + h[-2])**2. - (2. * h[-2] + 3. * h[-1]) * h[-2] / (h[-1] + h[-2]) * delta[-1] / h[-1]
		partials['spar_spline_rhs', 'z_sparnode'][-1,-2] = -(3. * h[-2] * delta[-1] + 2. * h[-1] * delta[-2]) / (h[-1] + h[-2]) + (4. * h[-2] + 3. * h[-1]) * delta[-1] / (h[-1] + h[-2]) + (2. * h[-2] + 3 * h[-1]) * h[-2] / (h[-1] + h[-2]) * delta[-1] / h[-1] - h[-1]**2. / (h[-1] + h[-2]) * delta[-2] / h[-2]
		partials['spar_spline_rhs', 'z_sparnode'][-1,-3] = -(4. * h[-2] + 3. * h[-1]) * delta[-1] / (h[-1] + h[-2]) + ((2. * h[-2] + 3. * h[-1]) * h[-2] * delta[-1] + h[-1]**2. * delta[-2]) / (h[-1] + h[-2])**2. + h[-1]**2. / (h[-1] + h[-2]) * delta[-2] / h[-2]

		partials['spar_spline_rhs', 'x_sparnode'][-1,-1] = (2. * h[-2] + 3. * h[-1]) * h[-2] / (h[-1] + h[-2]) * 1. / h[-1]
		partials['spar_spline_rhs', 'x_sparnode'][-1,-2] = -(2. * h[-2] + 3 * h[-1]) * h[-2] / (h[-1] + h[-2]) * 1. / h[-1] + h[-1]**2. / (h[-1] + h[-2]) * 1. / h[-2]
		partials['spar_spline_rhs', 'x_sparnode'][-1,-3] = -h[-1]**2. / (h[-1] + h[-2]) * 1. / h[-2]