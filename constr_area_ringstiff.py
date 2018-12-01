import numpy as np

from openmdao.api import ExplicitComponent

class ConstrAreaRingstiff(ExplicitComponent):

	def setup(self):
		self.add_input('A_R', val=np.zeros(10), units='m**2')
		self.add_input('Z_l', val=np.zeros(10))
		self.add_input('l_stiff', val=np.zeros(10), units='m')
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')

		self.add_output('constr_area_ringstiff', val=np.zeros(10), units='m**2')

		self.declare_partials('constr_area_ringstiff', 'A_R', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('constr_area_ringstiff', 'Z_l', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('constr_area_ringstiff', 'l_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('constr_area_ringstiff', 'wt_spar_p', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['constr_area_ringstiff'] = ((2. / (inputs['Z_l']**2.) + 0.06) * inputs['l_stiff'] * inputs['wt_spar_p'][:-1]) / inputs['A_R'] - 1. #less than 0 to satisfy constraint

	def compute_partials(self, inputs, partials):
		partials['constr_area_ringstiff', 'A_R'] = -((2. / (inputs['Z_l']**2.) + 0.06) * inputs['l_stiff'] * inputs['wt_spar_p'][:-1]) / inputs['A_R']**2.
		partials['constr_area_ringstiff', 'Z_l'] = -4. / inputs['Z_l']**3. * inputs['l_stiff'] * inputs['wt_spar_p'][:-1] / inputs['A_R']
		partials['constr_area_ringstiff', 'l_stiff'] = ((2. / (inputs['Z_l']**2.) + 0.06) * inputs['wt_spar_p'][:-1]) / inputs['A_R']
		partials['constr_area_ringstiff', 'wt_spar_p'] = ((2. / (inputs['Z_l']**2.) + 0.06) * inputs['l_stiff']) / inputs['A_R']
		