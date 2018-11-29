import numpy as np

from openmdao.api import ExplicitComponent

class HullGammaM(ExplicitComponent):

	def setup(self):
		self.add_input('lambda_s', val=np.zeros(10))

		self.add_output('gamma_M', val=np.zeros(10))

		self.declare_partials('gamma_M', 'lambda_s', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		#material factor ref DNVGL-ST-0126

		for i in xrange(len(inputs['lambda_s'])):
			if inputs['lambda_s'][i] < 0.5:
				gamma_M = 1.1
			elif inputs['lambda_s'][i] > 1.0:
				gamma_M = 1.4
			else:
				gamma_M = 0.8 + 0.6 * inputs['lambda_s'][i]

			outputs['gamma_M'][i] = gamma_M

	def compute_partials(self, inputs, partials):
		
		for i in xrange(len(inputs['lambda_s'])):
			if inputs['lambda_s'][i] >= 0.5 and inputs['lambda_s'][i] <= 1.0:
				partials['gamma_M', 'lambda_s'][i] = 0.6
			else:
				partials['gamma_M', 'lambda_s'][i] = 0.