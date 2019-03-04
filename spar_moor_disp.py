import numpy as np

from openmdao.api import ExplicitComponent

class SparMoorDisp(ExplicitComponent):

	def setup(self):
		self.add_input('z_sparnode', val=np.zeros(13), units='m')
		self.add_input('x_sparnode', val=np.zeros(13), units='m')
		self.add_input('z_moor', val=0., units='m')

		self.add_output('x_moor', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		z_sparnode = inputs['z_sparnode']
		x_sparnode = inputs['x_sparnode']
		z_moor = inputs['z_moor']

		moorel = 0

		for i in xrange(len(z_sparnode)-1):
			if (z_moor > z_sparnode[i]) and (z_moor <= z_sparnode[i+1]):
				x_moor = x_sparnode[i] + (z_moor - z_sparnode[i]) * (x_sparnode[i+1] - x_sparnode[i]) / (z_sparnode[i+1] - z_sparnode[i])
				break

		outputs['x_moor'] = x_moor

	def compute_partials(self, inputs, partials):
		z_sparnode = inputs['z_sparnode']
		x_sparnode = inputs['x_sparnode']
		z_moor = inputs['z_moor']

		partials['x_moor', 'z_sparnode'] = np.zeros((1,13))
		partials['x_moor', 'x_sparnode'] = np.zeros((1,13))
		partials['x_moor', 'z_moor'] = 0.

		moorel = 0

		for i in xrange(len(z_sparnode)-1):
			if (z_moor > z_sparnode[i]) and (z_moor <= z_sparnode[i+1]):
				x_moor = x_sparnode[i] + (z_moor - z_sparnode[i]) * (x_sparnode[i+1] - x_sparnode[i]) / (z_sparnode[i+1] - z_sparnode[i])
				partials['x_moor', 'z_sparnode'][0,i] += -(x_sparnode[i+1] - x_sparnode[i]) / (z_sparnode[i+1] - z_sparnode[i]) + (z_moor - z_sparnode[i]) * (x_sparnode[i+1] - x_sparnode[i]) / (z_sparnode[i+1] - z_sparnode[i])**2.
				partials['x_moor', 'z_sparnode'][0,i+1] += -(z_moor - z_sparnode[i]) * (x_sparnode[i+1] - x_sparnode[i]) / (z_sparnode[i+1] - z_sparnode[i])**2.
				partials['x_moor', 'x_sparnode'][0,i] += 1. - (z_moor - z_sparnode[i]) / (z_sparnode[i+1] - z_sparnode[i])
				partials['x_moor', 'x_sparnode'][0,i+1] += (z_moor - z_sparnode[i]) / (z_sparnode[i+1] - z_sparnode[i])
				partials['x_moor', 'z_moor'] += (x_sparnode[i+1] - x_sparnode[i]) / (z_sparnode[i+1] - z_sparnode[i])
				break