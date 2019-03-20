import numpy as np

from openmdao.api import ExplicitComponent

class HullIR(ExplicitComponent):

	def setup(self):
		self.add_input('I_x', val=np.zeros(10), units='m**4')
		#self.add_input('I_xh', val=np.zeros(10), units='m**4')
		self.add_input('I_h', val=np.zeros(10), units='m**4')

		self.add_output('I_R', val=np.zeros(10), units='m**4')

		self.declare_partials('I_R', 'I_x', rows=np.arange(10), cols=np.arange(10))
		#self.declare_partials('I_R', 'I_xh', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_R', 'I_h', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		#outputs['I_R'] = inputs['I_x'] + inputs['I_xh'] + inputs['I_h']
		outputs['I_R'] = inputs['I_x'] + inputs['I_h']

	def compute_partials(self, inputs, partials):
		partials['I_R', 'I_x'] = np.ones(10)
		#partials['I_R', 'I_xh'] = np.ones(10)
		partials['I_R', 'I_h'] = np.ones(10)