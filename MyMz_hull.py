import numpy as np

from openmdao.api import ExplicitComponent

class MyMzHull(ExplicitComponent):

	def setup(self):
		self.add_input('dthrust_dv', val=0., units='N*s/m')

		self.add_output('My_hull', val=np.zeros(10), units='N*m')
		self.add_output('Mz_hull', val=np.zeros(10), units='N*m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		# = inputs['']

		outputs['My_hull'] = np.zeros(10)
		outputs['Mz_hull'] = np.zeros(10)

	#def compute_partials(self, inputs, partials):
	#	partials['', ''] = 