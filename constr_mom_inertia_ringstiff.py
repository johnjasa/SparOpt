import numpy as np

from openmdao.api import ExplicitComponent

class ConstrMomInertiaRingstiff(ExplicitComponent):

	def setup(self):
		self.add_input('I_R', val=np.zeros(10), units='m**4')
		self.add_input('I_stiff', val=np.zeros(10), units='m**4')

		self.add_output('mom_inertia_ringstiff', val=np.zeros(10), units='m**4')

		self.declare_partials('mom_inertia_ringstiff', 'I_R', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('mom_inertia_ringstiff', 'I_stiff', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['mom_inertia_ringstiff'] = inputs['I_R'] / inputs['I_stiff'] - 1. #less than 0 to satisfy constraint

	def compute_partials(self, inputs, partials):
		partials['mom_inertia_ringstiff', 'I_R'] = 1. / inputs['I_stiff']
		partials['mom_inertia_ringstiff', 'I_stiff'] = -inputs['I_R'] / inputs['I_stiff']**2.
		