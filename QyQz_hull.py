import numpy as np

from openmdao.api import ExplicitComponent

class QyQzHull(ExplicitComponent):

	def setup(self):
		self.add_input('dthrust_dv', val=0., units='N*s/m')

		self.add_output('Qy_hull', val=np.zeros(10), units='N')
		self.add_output('Qz_hull', val=np.zeros(10), units='N')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		# = inputs['']

		outputs['Qy_hull'] = np.ones(10)
		outputs['Qz_hull'] = np.ones(10)

	#def compute_partials(self, inputs, partials):
	#	partials['', ''] = 