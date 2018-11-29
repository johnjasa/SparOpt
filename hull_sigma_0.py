import numpy as np

from openmdao.api import ExplicitComponent

class HullSigma0(ExplicitComponent):

	def setup(self):
		self.add_input('sigma_a', val=np.zeros(10), units='MPa')
		self.add_input('sigma_m', val=np.zeros(10), units='MPa')
		self.add_input('sigma_h', val=np.zeros(10), units='MPa')

		self.add_output('sigma_a0', val=np.zeros(10), units='MPa')
		self.add_output('sigma_m0', val=np.zeros(10), units='MPa')
		self.add_output('sigma_h0', val=np.zeros(10), units='MPa')

		self.declare_partials('sigma_a0', 'sigma_a', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_m0', 'sigma_m', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_h0', 'sigma_h', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		if inputs['sigma_a'] >= 0.:
			sigma_a0 = 0.
		else:
			sigma_a0 = -inputs['sigma_a']

		if inputs['sigma_m'] >= 0.:
			sigma_m0 = 0.
		else:
			sigma_m0 = -inputs['sigma_m']

		if inputs['sigma_h'] >= 0.: #internal net pressure
			sigma_h0 = 0.
		else:
			sigma_h0 = -inputs['sigma_h']
		
		outputs['sigma_a0'] = sigma_a0
		outputs['sigma_m0'] = sigma_m0
		outputs['sigma_h0'] = sigma_h0

	def compute_partials(self, inputs, partials):
		if inputs['sigma_a'] >= 0.:
			dsigma_a0 = 0.
		else:
			dsigma_a0 = -1.

		if inputs['sigma_m'] >= 0.:
			dsigma_m0 = 0.
		else:
			dsigma_m0 = -1.

		if inputs['sigma_h'] >= 0.:
			dsigma_h0 = 0.
		else:
			dsigma_h0 = -1.

		partials['sigma_a0', 'sigma_a'] = dsigma_a0
		partials['sigma_m0', 'sigma_m'] = dsigma_m0
		partials['sigma_h0', 'sigma_h'] = dsigma_h0