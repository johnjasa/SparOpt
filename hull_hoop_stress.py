import numpy as np

from openmdao.api import ExplicitComponent

class HullHoopStressRingstiff(ExplicitComponent):

	def setup(self):
		self.add_input('net_pressure', val=np.zeros(10), units='MPa')
		self.add_input('r_hull', val=np.zeros(10), units='m')
		self.add_input('wt_spar', val=np.zeros(10), units='m')
		self.add_input('alpha', val=np.zeros(10))
		self.add_input('zeta', val=np.zeros(10))
		self.add_input('sigma_a', val=np.zeros(10), units='MPa')
		self.add_input('sigma_m', val=np.zeros(10), units='MPa')

		self.add_output('sigma_h', val=np.zeros(10))

		self.declare_partials('sigma_h', 'net_pressure', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_h', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_h', 'wt_spar', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_h', 'alpha', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_h', 'zeta', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_h', 'sigma_a', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_h', 'sigma_m', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		nu = 0.3

		outputs['sigma_h'] = inputs['net_pressure'] * inputs['r_hull'] / inputs['wt_spar'] - inputs['alpha'] * inputs['zeta'] / (inputs['alpha'] + 1.) * (inputs['net_pressure'] * inputs['r_hull'] / inputs['wt_spar'] - nu * (inputs['sigma_a'] + inputs['sigma_m']))

	def compute_partials(self, inputs, partials):
		nu = 0.3

		partials['sigma_h', 'net_pressure'] = inputs['r_hull'] / inputs['wt_spar'] - inputs['alpha'] * inputs['zeta'] / (inputs['alpha'] + 1.) * (inputs['r_hull'] / inputs['wt_spar'])
		partials['sigma_h', 'r_hull'] = inputs['net_pressure'] / inputs['wt_spar'] - inputs['alpha'] * inputs['zeta'] / (inputs['alpha'] + 1.) * (inputs['net_pressure'] / inputs['wt_spar'])
		partials['sigma_h', 'wt_spar'] = -inputs['net_pressure'] * inputs['r_hull'] / inputs['wt_spar']**2. + inputs['alpha'] * inputs['zeta'] / (inputs['alpha'] + 1.) * (inputs['net_pressure'] * inputs['r_hull'] / inputs['wt_spar']**2.)
		partials['sigma_h', 'alpha'] = -inputs['zeta'] / (inputs['alpha'] + 1.)**2. * (inputs['net_pressure'] * inputs['r_hull'] / inputs['wt_spar'] - nu * (inputs['sigma_a'] + inputs['sigma_m']))
		partials['sigma_h', 'zeta'] = -inputs['alpha'] / (inputs['alpha'] + 1.) * (inputs['net_pressure'] * inputs['r_hull'] / inputs['wt_spar'] - nu * (inputs['sigma_a'] + inputs['sigma_m']))
		partials['sigma_h', 'sigma_a'] = inputs['alpha'] * inputs['zeta'] / (inputs['alpha'] + 1.) * nu
		partials['sigma_h', 'sigma_m'] = inputs['alpha'] * inputs['zeta'] / (inputs['alpha'] + 1.) * nu