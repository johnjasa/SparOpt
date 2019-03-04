import numpy as np

from openmdao.api import ExplicitComponent

class SparNode1LHS(ExplicitComponent):

	def setup(self):
		self.add_input('z_sparnode', val=np.zeros(13), units='m')

		self.add_output('spar_spline_lhs', val=np.zeros((13,13)), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		z = inputs['z_sparnode']

		N_spar = len(z)

		h = np.zeros(N_spar - 1)
		for i in xrange(N_spar - 1):
			h[i] = z[i+1] - z[i]

		outputs['spar_spline_lhs'] = np.zeros((N_spar,N_spar))
		
		for i in xrange(1,N_spar - 1):
			outputs['spar_spline_lhs'][i,i] = 2. * (h[i] + h[i-1])
			outputs['spar_spline_lhs'][i,i-1] = h[i]
			outputs['spar_spline_lhs'][i,i+1] = h[i-1]

		outputs['spar_spline_lhs'][0,0] = h[1]
		outputs['spar_spline_lhs'][0,1] = h[0] + h[1]
		outputs['spar_spline_lhs'][-1,-1] = h[-2]
		outputs['spar_spline_lhs'][-1,-2] = h[-1] + h[-2]

	def compute_partials(self, inputs, partials):
		z = inputs['z_sparnode']

		N_spar = len(z)

		partials['spar_spline_lhs', 'z_sparnode'] = np.zeros((N_spar * N_spar,N_spar))

		for i in xrange(1,N_spar - 1):
			partials['spar_spline_lhs', 'z_sparnode'][i*N_spar-1+i,i] = -1.
			partials['spar_spline_lhs', 'z_sparnode'][i*N_spar-1+i,i+1] = 1.

			partials['spar_spline_lhs', 'z_sparnode'][i*N_spar+i,i-1] = -2.
			partials['spar_spline_lhs', 'z_sparnode'][i*N_spar+i,i] = 0.
			partials['spar_spline_lhs', 'z_sparnode'][i*N_spar+i,i+1] = 2.

			partials['spar_spline_lhs', 'z_sparnode'][i*N_spar+1+i,i-1] = -1.
			partials['spar_spline_lhs', 'z_sparnode'][i*N_spar+1+i,i] = 1.

		partials['spar_spline_lhs', 'z_sparnode'][0,1] = -1.
		partials['spar_spline_lhs', 'z_sparnode'][0,2] = 1.
		partials['spar_spline_lhs', 'z_sparnode'][1,0] = -1.
		partials['spar_spline_lhs', 'z_sparnode'][1,2] = 1.
		partials['spar_spline_lhs', 'z_sparnode'][-1,-3] = -1.
		partials['spar_spline_lhs', 'z_sparnode'][-1,-2] = 1.
		partials['spar_spline_lhs', 'z_sparnode'][-2,-3] = -1.
		partials['spar_spline_lhs', 'z_sparnode'][-2,-1] = 1.
