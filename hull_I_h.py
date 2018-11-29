import numpy as np

from openmdao.api import ExplicitComponent

class HullIH(ExplicitComponent):

	def setup(self):
		self.add_input('net_pressure', val=np.zeros(10), units='MPa')
		self.add_input('r_hull', val=np.zeros(10), units='m')
		self.add_input('r_0', val=np.zeros(10), units='m')
		self.add_input('l_stiff', val=np.zeros(10), units='m')
		self.add_input('z_t', val=np.zeros(10), units='m')
		self.add_input('delta_0', val=np.zeros(10), units='m')
		self.add_input('f_r', val=0., units='MPa')
		self.add_input('sigma_hR', val=np.zeros(10), units='MPa')

		self.add_output('I_h', val=np.zeros(10), units='m**4')

		self.declare_partials('I_h', 'net_pressure', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_h', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_h', 'r_0', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_h', 'l_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_h', 'z_t', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_h', 'delta_0', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_h', 'f_r')
		self.declare_partials('I_h', 'sigma_hR', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['I_h'] = abs(inputs['net_pressure']) * inputs['r_hull'] * inputs['r_0']**2. * inputs['l_stiff'] / (3. * E) * (1.5 + 3. * E * inputs['z_t'] * inputs['delta_0'] / (inputs['r_0']**2. * (inputs['f_r'] / 2. - abs(inputs['sigma_hR']))))

	def compute_partials(self, inputs, partials):
		partials['I_h', 'net_pressure'] = inputs['net_pressure'] / abs(inputs['net_pressure']) * inputs['r_hull'] * inputs['r_0']**2. * inputs['l_stiff'] / (3. * E) * (1.5 + 3. * E * inputs['z_t'] * inputs['delta_0'] / (inputs['r_0']**2. * (inputs['f_r'] / 2. - abs(inputs['sigma_hR']))))
		partials['I_h', 'r_hull'] = abs(inputs['net_pressure']) * inputs['r_0']**2. * inputs['l_stiff'] / (3. * E) * (1.5 + 3. * E * inputs['z_t'] * inputs['delta_0'] / (inputs['r_0']**2. * (inputs['f_r'] / 2. - abs(inputs['sigma_hR']))))
		partials['I_h', 'r_0'] =  abs(inputs['net_pressure']) * inputs['r_hull'] * inputs['l_stiff'] / (3. * E) * (2. * inputs['r_0'] * (1.5 + 3. * E * inputs['z_t'] * inputs['delta_0'] / (inputs['r_0']**2. * (inputs['f_r'] / 2. - abs(inputs['sigma_hR'])))) - inputs['r_0']**2. * 2. * (3. * E * inputs['z_t'] * inputs['delta_0'] / (inputs['r_0']**3. * (inputs['f_r'] / 2. - abs(inputs['sigma_hR']))))) 
		partials['I_h', 'l_stiff'] = abs(inputs['net_pressure']) * inputs['r_hull'] * inputs['r_0']**2. * 1. / (3. * E) * (1.5 + 3. * E * inputs['z_t'] * inputs['delta_0'] / (inputs['r_0']**2. * (inputs['f_r'] / 2. - abs(inputs['sigma_hR']))))
		partials['I_h', 'z_t'] = abs(inputs['net_pressure']) * inputs['r_hull'] * inputs['r_0']**2. * inputs['l_stiff'] / (3. * E) * (3. * E * inputs['delta_0'] / (inputs['r_0']**2. * (inputs['f_r'] / 2. - abs(inputs['sigma_hR']))))
		partials['I_h', 'delta_0'] = abs(inputs['net_pressure']) * inputs['r_hull'] * inputs['r_0']**2. * inputs['l_stiff'] / (3. * E) * (3. * E * inputs['z_t'] * 1. / (inputs['r_0']**2. * (inputs['f_r'] / 2. - abs(inputs['sigma_hR']))))
		partials['I_h', 'f_r'] = -abs(inputs['net_pressure']) * inputs['r_hull'] * inputs['r_0']**2. * inputs['l_stiff'] / (3. * E) * 3. * E * inputs['z_t'] * inputs['delta_0'] / (inputs['r_0']**2. * (inputs['f_r'] / 2. - abs(inputs['sigma_hR'])))**2. * inputs['r_0']**2. / 2.
		partials['I_h', 'sigma_hR'] = abs(inputs['net_pressure']) * inputs['r_hull'] * inputs['r_0']**2. * inputs['l_stiff'] / (3. * E) * 3. * E * inputs['z_t'] * inputs['delta_0'] / (inputs['r_0']**2. * (inputs['f_r'] / 2. - abs(inputs['sigma_hR'])))**2. * inputs['r_0']**2. * inputs['sigma_hR'] / abs(inputs['sigma_hR'])