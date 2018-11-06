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

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_tower = inputs['D_tower']
		wt_tower = inputs['wt_tower']
		
		outputs['resp_TB_stress'] = inputs['resp_TB_moment'] / (np.pi / 64. * (D_tower[0]**4. - (D_tower[0] - 2. * wt_tower[0])**4.))**2. * (D_tower[0] / 2. * 10.**(-6.))**2.