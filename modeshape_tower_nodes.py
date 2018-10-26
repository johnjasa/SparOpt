import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeTowerNodes(ExplicitComponent):

	def setup(self):
		self.add_input('Z_tower', val=np.zeros(11), units='m')

		self.add_output('z_towernode', val=np.zeros(11), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['z_towernode'] = inputs['Z_tower']

	def compute_partials(self, inputs, partials):
		partials['z_towernode', 'Z_tower'] = np.identity(11)