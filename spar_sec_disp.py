import numpy as np

from openmdao.api import ExplicitComponent

class SparSecDisp(ExplicitComponent):

	def setup(self):
		self.add_input('z_sparnode', val=np.zeros(13), units='m')
		self.add_input('x_sparnode', val=np.zeros(13), units='m')
		self.add_input('x_sparelem', val=np.zeros(12), units='m')
		self.add_input('stddev_vel_distr', val=np.zeros(12), units='m/s')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('L_ball', val=0., units='m')
		self.add_input('z_moor', val=0., units='m')

		self.add_output('X_sparnode', val=np.zeros(11), units='m')
		self.add_output('X_sparelem', val=np.zeros(10), units='m')
		self.add_output('stddev_vel_X_sparelem', val=np.zeros(10), units='m/s')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		z_sparnode = inputs['z_sparnode']
		x_sparnode = inputs['x_sparnode']
		x_sparelem = inputs['x_sparelem']
		stddev_vel_distr = inputs['stddev_vel_distr']
		L_ball = inputs['L_ball'][0]
		z_ball = -inputs['spar_draft'][0] + L_ball #top of ballast
		z_moor = inputs['z_moor']
		z_SWL = 0.
		
		count = 0

		for i in xrange(len(z_sparnode)):
			if (z_sparnode[i] != z_moor) and (z_sparnode[i] != z_ball) and (z_sparnode[i] != z_SWL):
				outputs['X_sparnode'][count] = x_sparnode[i]
				count += 1

		count = 0

		for i in xrange(len(z_sparnode)-1):
			if (z_sparnode[i] != z_moor) and (z_sparnode[i] != z_ball) and (z_sparnode[i] != z_SWL):
				outputs['X_sparelem'][count] = x_sparelem[i]
				outputs['stddev_vel_X_sparelem'][count] = stddev_vel_distr[i]
				count += 1

	def compute_partials(self, inputs, partials):
		z_sparnode = inputs['z_sparnode']
		x_sparnode = inputs['x_sparnode']
		x_sparelem = inputs['x_sparelem']
		stddev_vel_distr = inputs['stddev_vel_distr']
		L_ball = inputs['L_ball'][0]
		z_ball = -inputs['spar_draft'][0] + L_ball #top of ballast
		z_moor = inputs['z_moor']
		z_SWL = 0.

		partials['X_sparnode', 'x_sparnode'] = np.zeros((11,13))
		partials['X_sparelem', 'x_sparelem'] = np.zeros((10,12))
		partials['stddev_vel_X_sparelem', 'stddev_vel_distr'] = np.zeros((10,12))

		count = 0
		
		for i in xrange(len(z_sparnode)):
			if (z_sparnode[i] != z_moor) and (z_sparnode[i] != z_ball) and (z_sparnode[i] != z_SWL):
				partials['X_sparnode', 'x_sparnode'][count,i] += 1.
				count += 1

		count = 0

		for i in xrange(len(z_sparnode)-1):
			if (z_sparnode[i] != z_moor) and (z_sparnode[i] != z_ball) and (z_sparnode[i] != z_SWL):
				partials['X_sparelem', 'x_sparelem'][count,i] += 1.
				partials['stddev_vel_X_sparelem', 'stddev_vel_distr'][count,i] += 1.
				count += 1