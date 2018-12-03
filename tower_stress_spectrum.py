import numpy as np

from openmdao.api import ExplicitComponent

class TowerStressSpectrum(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_tower_moment', val=np.zeros((N_omega,11)), units='(N*m)**2*s/rad')
		self.add_input('D_tower_p', val=np.zeros(11), units='m')
		self.add_input('wt_tower_p', val=np.zeros(11), units='m')

		self.add_output('resp_tower_stress', val=np.zeros((N_omega,11)), units='MPa**2*s/rad')

		self.declare_partials('resp_tower_stress', 'resp_tower_moment', rows=np.arange(11*N_omega), cols=np.arange(11*N_omega))
		self.declare_partials('resp_tower_stress', 'D_tower_p', rows=np.arange(11*N_omega), cols=np.tile(np.arange(11),N_omega))
		self.declare_partials('resp_tower_stress', 'wt_tower_p', rows=np.arange(11*N_omega), cols=np.tile(np.arange(11),N_omega))

	def compute(self, inputs, outputs):
		D_tower_p = inputs['D_tower_p']
		wt_tower_p = inputs['wt_tower_p']
		
		for i in xrange(len(D_tower_p)):
			outputs['resp_tower_stress'][:,i] = inputs['resp_tower_moment'][:,i] / (np.pi / 64. * (D_tower_p[i]**4. - (D_tower_p[i] - 2. * wt_tower_p[i])**4.))**2. * (D_tower_p[i] / 2. * 10.**(-6.))**2.

	def compute_partials(self, inputs, partials):
		N_omega = len(self.omega)

		D_tower_p = inputs['D_tower_p']
		wt_tower_p = inputs['wt_tower_p']

		for i in xrange(N_omega):
			partials['resp_tower_stress', 'D_tower_p'][i*11:i*11+11] = inputs['resp_tower_moment'][i] / (np.pi / 64. * (D_tower_p**4. - (D_tower_p - 2. * wt_tower_p)**4.))**2. * (D_tower_p / 2. * 10.**(-6.)) * 10.**(-6.) + inputs['resp_tower_moment'][i] * (D_tower_p / 2. * 10.**(-6.))**2. * (-2.) / (np.pi / 64. * (D_tower_p**4. - (D_tower_p - 2. * wt_tower_p)**4.))**3. * np.pi / 16. * (D_tower_p**3. - (D_tower_p - 2. * wt_tower_p)**3.)
			partials['resp_tower_stress', 'wt_tower_p'][i*11:i*11+11] = -inputs['resp_tower_moment'][i] * (D_tower_p / 2. * 10.**(-6.))**2. * 2. / (np.pi / 64. * (D_tower_p**4. - (D_tower_p - 2. * wt_tower_p)**4.))**3. * np.pi / 8. * (D_tower_p - 2. * wt_tower_p)**3.

			partials['resp_tower_stress', 'resp_tower_moment'][i*11:i*11+11] = 1. / (np.pi / 64. * (D_tower_p**4. - (D_tower_p - 2. * wt_tower_p)**4.))**2. * (D_tower_p / 2. * 10.**(-6.))**2.