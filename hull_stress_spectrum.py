import numpy as np

from openmdao.api import ExplicitComponent

class HullStressSpectrum(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_hull_moment', val=np.zeros((N_omega,10)), units='(N*m)**2*s/rad')
		self.add_input('D_spar_p', val=np.zeros(11), units='m')
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')

		self.add_output('resp_hull_stress', val=np.zeros((N_omega,10)), units='MPa**2*s/rad')

		self.declare_partials('resp_hull_stress', 'resp_hull_moment', rows=np.arange(10*N_omega), cols=np.arange(10*N_omega))
		self.declare_partials('resp_hull_stress', 'D_spar_p', rows=np.arange(10*N_omega), cols=np.tile(np.arange(10),N_omega))
		self.declare_partials('resp_hull_stress', 'wt_spar_p', rows=np.arange(10*N_omega), cols=np.tile(np.arange(10),N_omega))

	def compute(self, inputs, outputs):
		D_spar_p = inputs['D_spar_p'][:-1]
		wt_spar_p = inputs['wt_spar_p'][:-1]
		
		for i in xrange(len(D_spar_p)):
			outputs['resp_hull_stress'][:,i] = inputs['resp_hull_moment'][:,i] / (np.pi / 64. * (D_spar_p[i]**4. - (D_spar_p[i] - 2. * wt_spar_p[i])**4.))**2. * (D_spar_p[i] / 2. * 10.**(-6.))**2.

	def compute_partials(self, inputs, partials):
		N_omega = len(self.omega)

		D_spar_p = inputs['D_spar_p'][:-1]
		wt_spar_p = inputs['wt_spar_p'][:-1]

		for i in xrange(N_omega):
			partials['resp_hull_stress', 'D_spar_p'][i*10:i*10+10] = inputs['resp_hull_moment'][i] / (np.pi / 64. * (D_spar_p**4. - (D_spar_p - 2. * wt_spar_p)**4.))**2. * (D_spar_p / 2. * 10.**(-6.)) * 10.**(-6.) + inputs['resp_hull_moment'][i] * (D_spar_p / 2. * 10.**(-6.))**2. * (-2.) / (np.pi / 64. * (D_spar_p**4. - (D_spar_p - 2. * wt_spar_p)**4.))**3. * np.pi / 16. * (D_spar_p**3. - (D_spar_p - 2. * wt_spar_p)**3.)
			partials['resp_hull_stress', 'wt_spar_p'][i*10:i*10+10] = -inputs['resp_hull_moment'][i] * (D_spar_p / 2. * 10.**(-6.))**2. * 2. / (np.pi / 64. * (D_spar_p**4. - (D_spar_p - 2. * wt_spar_p)**4.))**3. * np.pi / 8. * (D_spar_p - 2. * wt_spar_p)**3.

			partials['resp_hull_stress', 'resp_hull_moment'][i*10:i*10+10] = 1. / (np.pi / 64. * (D_spar_p**4. - (D_spar_p - 2. * wt_spar_p)**4.))**2. * (D_spar_p / 2. * 10.**(-6.))**2.