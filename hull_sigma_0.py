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
		sigma_a0 = np.zeros(len(inputs['sigma_a']))
		sigma_m0 = np.zeros(len(inputs['sigma_a']))
		sigma_h0 = np.zeros(len(inputs['sigma_a']))

		for i in xrange(len(inputs['sigma_a'])):
			if inputs['sigma_a'][i] >= 0.:
				sigma_a0[i] = 0.
			else:
				sigma_a0[i] = -inputs['sigma_a'][i]

			if inputs['sigma_m'][i] >= 0.:
				sigma_m0[i] = 0.
			else:
				sigma_m0[i] = -inputs['sigma_m'][i]

			if inputs['sigma_h'][i] >= 0.:
				sigma_h0[i] = 0.
			else:
				sigma_h0[i] = -inputs['sigma_h'][i]
		
		outputs['sigma_a0'] = sigma_a0
		outputs['sigma_m0'] = sigma_m0
		outputs['sigma_h0'] = sigma_h0

	def compute_partials(self, inputs, partials):
		dsigma_a0 = np.zeros(len(inputs['sigma_a']))
		dsigma_m0 = np.zeros(len(inputs['sigma_a']))
		dsigma_h0 = np.zeros(len(inputs['sigma_a']))

		for i in xrange(len(inputs['sigma_a'])):
			if inputs['sigma_a'][i] >= 0.:
				dsigma_a0[i] = 0.
			else:
				dsigma_a0[i] = -1.

			if inputs['sigma_m'][i] >= 0.:
				dsigma_m0[i] = 0.
			else:
				dsigma_m0[i] = -1.

			if inputs['sigma_h'][i] >= 0.:
				dsigma_h0[i] = 0.
			else:
				dsigma_h0[i] = -1.

		partials['sigma_a0', 'sigma_a'] = dsigma_a0
		partials['sigma_m0', 'sigma_m'] = dsigma_m0
		partials['sigma_h0', 'sigma_h'] = dsigma_h0