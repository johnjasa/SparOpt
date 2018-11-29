import numpy as np

from openmdao.api import ExplicitComponent

class HullIC(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('wt_spar', val=np.zeros(10), units='m')

		self.add_output('I_C', val=np.zeros(10), units='m**4')

		self.declare_partials('I_C', 'D_spar', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_C', 'wt_spar', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['I_C'] = np.pi / 64. * (inputs['D_spar']**4. - (inputs['D_spar'] - 2. * inputs['t_spar'])**4.)

	def compute_partials(self, inputs, partials):
		partials['I_C', 'D_spar'] = np.pi / 16. * (inputs['D_spar']**3. - (inputs['D_spar'] - 2. * inputs['t_spar'])**3.)
		partials['I_C', 'wt_spar'] = np.pi / 8. * (inputs['D_spar'] - 2. * inputs['t_spar'])**3.