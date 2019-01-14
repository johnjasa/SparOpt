import numpy as np

from openmdao.api import ExplicitComponent

class HullBendingStress(ExplicitComponent):

	def setup(self):
		self.add_input('My_hull', val=np.zeros(10), units='N*m')
		self.add_input('Mz_hull', val=np.zeros(10), units='N*m')
		self.add_input('r_hull', val=np.zeros(10), units='m')
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')
		self.add_input('angle_hull', val=0., units='rad')

		self.add_output('sigma_m', val=np.ones(10), units='MPa')

		self.declare_partials('sigma_m', 'My_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_m', 'Mz_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_m', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_m', 'wt_spar_p', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_m', 'angle_hull')

	def compute(self, inputs, outputs):
		outputs['sigma_m'] = inputs['My_hull'] / (np.pi * inputs['r_hull']**2. * inputs['wt_spar_p'][:-1]) * np.cos(inputs['angle_hull']) * 1e-6 - inputs['Mz_hull'] / (np.pi * inputs['r_hull']**2. * inputs['wt_spar_p'][:-1]) * np.sin(inputs['angle_hull']) * 1e-6

	def compute_partials(self, inputs, partials):
		partials['sigma_m', 'My_hull'] = 1. / (np.pi * inputs['r_hull']**2. * inputs['wt_spar_p'][:-1]) * np.cos(inputs['angle_hull']) * 1e-6
		partials['sigma_m', 'Mz_hull'] = -1. / (np.pi * inputs['r_hull']**2. * inputs['wt_spar_p'][:-1]) * np.sin(inputs['angle_hull']) * 1e-6
		partials['sigma_m', 'r_hull'] = -2. * inputs['My_hull'] / (np.pi * inputs['r_hull']**3. * inputs['wt_spar_p'][:-1]) * np.cos(inputs['angle_hull']) * 1e-6 + 2. * inputs['Mz_hull'] / (np.pi * inputs['r_hull']**3. * inputs['wt_spar_p'][:-1]) * np.sin(inputs['angle_hull']) * 1e-6
		partials['sigma_m', 'wt_spar_p'] = -inputs['My_hull'] / (np.pi * inputs['r_hull']**2. * inputs['wt_spar_p'][:-1]**2.) * np.cos(inputs['angle_hull']) * 1e-6 + inputs['Mz_hull'] / (np.pi * inputs['r_hull']**2. * inputs['wt_spar_p'][:-1]**2.) * np.sin(inputs['angle_hull']) * 1e-6
		partials['sigma_m', 'angle_hull'] = -inputs['My_hull'] / (np.pi * inputs['r_hull']**2. * inputs['wt_spar_p'][:-1]) * np.sin(inputs['angle_hull']) * 1e-6 - inputs['Mz_hull'] / (np.pi * inputs['r_hull']**2. * inputs['wt_spar_p'][:-1]) * np.cos(inputs['angle_hull']) * 1e-6
		