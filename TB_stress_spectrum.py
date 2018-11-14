import numpy as np

from openmdao.api import ExplicitComponent

class TBStressSpectrum(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_TB_moment', val=np.zeros(N_omega), units='(N*m)**2*s/rad')
		self.add_input('D_tower', val=np.zeros(10), units='m')
		self.add_input('wt_tower', val=np.zeros(10), units='m')

		self.add_output('resp_TB_stress', val=np.zeros(N_omega), units='MPa**2*s/rad')

		self.declare_partials('resp_TB_stress', 'resp_TB_moment', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_TB_stress', 'D_tower')
		self.declare_partials('resp_TB_stress', 'wt_tower')

	def compute(self, inputs, outputs):
		D_tower = inputs['D_tower']
		wt_tower = inputs['wt_tower']
		
		outputs['resp_TB_stress'] = inputs['resp_TB_moment'] / (np.pi / 64. * (D_tower[0]**4. - (D_tower[0] - 2. * wt_tower[0])**4.))**2. * (D_tower[0] / 2. * 10.**(-6.))**2.

	def compute_partials(self, inputs, partials):
		N_omega = len(self.omega)

		D_tower = inputs['D_tower']
		wt_tower = inputs['wt_tower']

		partials['resp_TB_stress', 'D_tower'] = np.zeros((N_omega,10))
		partials['resp_TB_stress', 'wt_tower'] = np.zeros((N_omega,10))

		partials['resp_TB_stress', 'D_tower'][:,0] = inputs['resp_TB_moment'] / (np.pi / 64. * (D_tower[0]**4. - (D_tower[0] - 2. * wt_tower[0])**4.))**2. * (D_tower[0] * 10.**(-6.)) * 0.5 * 10.**(-6.) + inputs['resp_TB_moment'] * (D_tower[0] / 2. * 10.**(-6.))**2. * (-2.) / (np.pi / 64. * (D_tower[0]**4. - (D_tower[0] - 2. * wt_tower[0])**4.))**3. * np.pi / 16. * (D_tower[0]**3. - (D_tower[0] - 2. * wt_tower[0])**3.)
		partials['resp_TB_stress', 'wt_tower'][:,0] = inputs['resp_TB_moment'] * (D_tower[0] / 2. * 10.**(-6.))**2. * 2. / (np.pi / 64. * (D_tower[0]**4. - (D_tower[0] - 2. * wt_tower[0])**4.))**3. * np.pi / 8. * (D_tower[0] - 2. * wt_tower[0])**3.

		partials['resp_TB_stress', 'resp_TB_moment'] = np.ones(N_omega) * 1. / (np.pi / 64. * (D_tower[0]**4. - (D_tower[0] - 2. * wt_tower[0])**4.))**2. * (D_tower[0] / 2. * 10.**(-6.))**2.