import numpy as np

from openmdao.api import ExplicitComponent

class HullIX(ExplicitComponent):

	def setup(self):
		self.add_input('wt_spar', val=np.zeros(10), units='m')
		self.add_input('sigma_a', val=np.zeros(10), units='MPa')
		self.add_input('sigma_m', val=np.zeros(10), units='MPa')
		self.add_input('r_0', val=np.zeros(10), units='m')
		self.add_input('l_stiff', val=np.zeros(10), units='m')

		self.add_output('I_x', val=np.zeros(10), units='m**4')

		self.declare_partials('I_x', 'wt_spar', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_x', 'sigma_a', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_x', 'sigma_m', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_x', 'r_0', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_x', 'l_stiff', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		E = 2.1e5 #MPa

		outputs['I_x'] = abs(inputs['sigma_a'] + inputs['sigma_m']) * inputs['wt_spar'] * inputs['r_0']**4. / (500. * E * inputs['l_stiff'])

	def compute_partials(self, inputs, partials):
		E = 2.1e5

		partials['I_x', 'wt_spar'] = abs(inputs['sigma_a'] + inputs['sigma_m']) * inputs['r_0']**4. / (500. * E * inputs['l_stiff'])
		partials['I_x', 'sigma_a'] = (inputs['sigma_a'] + inputs['sigma_m']) / abs(inputs['sigma_a'] + inputs['sigma_m']) * inputs['wt_spar'] * inputs['r_0']**4. / (500. * E * inputs['l_stiff'])
		partials['I_x', 'sigma_m'] = (inputs['sigma_a'] + inputs['sigma_m']) / abs(inputs['sigma_a'] + inputs['sigma_m']) * inputs['wt_spar'] * inputs['r_0']**4. / (500. * E * inputs['l_stiff'])
		partials['I_x', 'r_0'] = 4. * abs(inputs['sigma_a'] + inputs['sigma_m']) * inputs['wt_spar'] * inputs['r_0']**3. / (500. * E * inputs['l_stiff'])
		partials['I_x', 'l_stiff'] = -abs(inputs['sigma_a'] + inputs['sigma_m']) * inputs['wt_spar'] * inputs['r_0']**4. / (500. * E * inputs['l_stiff']**2.)