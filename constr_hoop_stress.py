import numpy as np

from openmdao.api import ExplicitComponent

class ConstrHoopStress(ExplicitComponent):

	def setup(self):
		self.add_input('sigma_hR', val=np.zeros(10), units='MPa')
		self.add_input('f_r', val=0., units='MPa')

		self.add_output('hoop_stress', val=np.zeros(10))

		self.declare_partials('hoop_stress', 'sigma_hR', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('hoop_stress', 'f_r')

	def compute(self, inputs, outputs):
		outputs['hoop_stress'] = abs(inputs['sigma_hR']) / (inputs['f_r'] / 2.) - 1. #less than 0 to satisfy constraint

	def compute_partials(self, inputs, partials):
		partials['hoop_stress', 'sigma_hR'] = inputs['sigma_hR'] / abs(inputs['sigma_hR']) * 1. / (inputs['f_r'] / 2.)
		partials['hoop_stress', 'f_r'] = -abs(inputs['sigma_hR']) / (inputs['f_r']**2. / 2.)
		