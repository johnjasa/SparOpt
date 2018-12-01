import numpy as np

from openmdao.api import ExplicitComponent

class ColBuckling(ExplicitComponent):

	def setup(self):
		self.add_input('buck_len', val=0.)
		#self.add_input('L_C', val=0., units='m')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('i_C', val=np.zeros(10), units='m')
		self.add_input('f_y', val=0., units='MPa')

		self.add_output('col_buckling', val=np.zeros(10))

		self.declare_partials('col_buckling', 'buck_len')
		#self.declare_partials('col_buckling', 'L_C')
		self.declare_partials('col_buckling', 'spar_draft')
		self.declare_partials('col_buckling', 'i_C', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('col_buckling', 'f_y')

	def compute(self, inputs, outputs):
		E = 2.1e5 #MPa

		#outputs['col_buckling'] = ((inputs['buck_len'] * inputs['L_C'] / inputs['i_C'])**2.) / (2.5 * E / inputs['f_y']) - 1. #less than 0 to satisfy constraint
		outputs['col_buckling'] = ((inputs['buck_len'] * (inputs['spar_draft'] + 10.) / inputs['i_C'])**2.) / (2.5 * E / inputs['f_y']) - 1.

	def compute_partials(self, inputs, partials):
		E = 2.1e5

		#partials['col_buckling', 'buck_len'] = (2. * inputs['buck_len'] * (inputs['L_C'] / inputs['i_C'])**2.) / (2.5 * E / inputs['f_y'])
		#partials['col_buckling', 'L_C'] = (inputs['buck_len']**2. * 2. * inputs['L_C'] / inputs['i_C']**2.) / (2.5 * E / inputs['f_y'])
		#partials['col_buckling', 'i_C'] = (-2. * (inputs['buck_len'] * inputs['L_C'])**2. / inputs['i_C']**3.) / (2.5 * E / inputs['f_y'])
		#partials['col_buckling', 'f_y'] = ((inputs['buck_len'] * inputs['L_C'] / inputs['i_C'])**2.) / (2.5 * E)
		partials['col_buckling', 'buck_len'] = (2. * inputs['buck_len'] * ((inputs['spar_draft'] + 10.) / inputs['i_C'])**2.) / (2.5 * E / inputs['f_y'])
		partials['col_buckling', 'spar_draft'] = (inputs['buck_len']**2. * 2. * (inputs['spar_draft'] + 10.) / inputs['i_C']**2.) / (2.5 * E / inputs['f_y'])
		partials['col_buckling', 'i_C'] = (-2. * (inputs['buck_len'] * (inputs['spar_draft'] + 10.))**2. / inputs['i_C']**3.) / (2.5 * E / inputs['f_y'])
		partials['col_buckling', 'f_y'] = ((inputs['buck_len'] * (inputs['spar_draft'] + 10.) / inputs['i_C'])**2.) / (2.5 * E)
		