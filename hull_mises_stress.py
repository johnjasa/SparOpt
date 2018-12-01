import numpy as np

from openmdao.api import ExplicitComponent

class HullMisesStress(ExplicitComponent):

	def setup(self):
		self.add_input('sigma_a', val=np.zeros(10), units='MPa')
		self.add_input('sigma_m', val=np.zeros(10), units='MPa')
		self.add_input('sigma_h', val=np.zeros(10), units='MPa')
		self.add_input('tau', val=np.zeros(10), units='MPa')

		self.add_output('sigma_j', val=np.zeros(10), units='MPa')

		self.declare_partials('sigma_j', 'sigma_a', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_j', 'sigma_m', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_j', 'sigma_h', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_j', 'tau', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['sigma_j'] = np.sqrt((inputs['sigma_a'] + inputs['sigma_m'])**2. - (inputs['sigma_a'] + inputs['sigma_m']) * inputs['sigma_h'] + inputs['sigma_h']**2. + 3. * inputs['tau']**2.)

	def compute_partials(self, inputs, partials):
		partials['sigma_j', 'sigma_a'] = 0.5 / np.sqrt((inputs['sigma_a'] + inputs['sigma_m'])**2. - (inputs['sigma_a'] + inputs['sigma_m']) * inputs['sigma_h'] + inputs['sigma_h']**2. + 3. * inputs['tau']**2.) * (2. * (inputs['sigma_a'] + inputs['sigma_m']) - inputs['sigma_h'])
		partials['sigma_j', 'sigma_m'] = 0.5 / np.sqrt((inputs['sigma_a'] + inputs['sigma_m'])**2. - (inputs['sigma_a'] + inputs['sigma_m']) * inputs['sigma_h'] + inputs['sigma_h']**2. + 3. * inputs['tau']**2.) * (2. * (inputs['sigma_a'] + inputs['sigma_m']) - inputs['sigma_h'])
		partials['sigma_j', 'sigma_h'] = 0.5 / np.sqrt((inputs['sigma_a'] + inputs['sigma_m'])**2. - (inputs['sigma_a'] + inputs['sigma_m']) * inputs['sigma_h'] + inputs['sigma_h']**2. + 3. * inputs['tau']**2.) * (-inputs['sigma_a'] - inputs['sigma_m'] + 2. * inputs['sigma_h'])
		partials['sigma_j', 'tau'] = 0.5 / np.sqrt((inputs['sigma_a'] + inputs['sigma_m'])**2. - (inputs['sigma_a'] + inputs['sigma_m']) * inputs['sigma_h'] + inputs['sigma_h']**2. + 3. * inputs['tau']**2.) * 6. * inputs['tau']
		