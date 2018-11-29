from __future__ import division
import numpy as np
import scipy.interpolate as si

from openmdao.api import ExplicitComponent

class BendingDamping(ExplicitComponent):

	def setup(self):
		self.add_input('x_dd_sparelem', val=np.zeros(13), units='1/m')
		self.add_input('x_dd_towerelem', val=np.zeros(10), units='1/m')
		self.add_input('EI_mode_elem', val=np.zeros(23), units='N*m**2')
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('z_towernode', val=np.zeros(11), units='m')
		self.add_input('alpha_damp', val=0., units='s')

		self.add_output('B_struct_77', val=0., units='N*s/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		z_sparnode = inputs['z_sparnode']
		z_towernode = inputs['z_towernode']
		x_dd_sparelem = inputs['x_dd_sparelem']
		x_dd_towerelem = inputs['x_dd_towerelem']
		EI = inputs['EI_mode_elem']
		alpha = inputs['alpha_damp']

		N_sparelem = len(x_dd_sparelem)
		N_towerelem = len(x_dd_towerelem)

		outputs['B_struct_77'] = 0.

		for i in xrange(N_sparelem):
			z = (z_sparnode[i] + z_sparnode[i+1]) / 2
			dz = z_sparnode[i+1] - z_sparnode[i]

			outputs['B_struct_77'] += alpha * dz * EI[i] * x_dd_sparelem[i]**2.

		for i in xrange(N_towerelem):
			z = (z_towernode[i] + z_towernode[i+1]) / 2
			dz = z_towernode[i+1] - z_towernode[i]

			outputs['B_struct_77'] += alpha * dz * EI[N_sparelem+i] * x_dd_towerelem[i]**2.

	def compute_partials(self, inputs, partials):
		z_sparnode = inputs['z_sparnode']
		z_towernode = inputs['z_towernode']
		x_dd_sparelem = inputs['x_dd_sparelem']
		x_dd_towerelem = inputs['x_dd_towerelem']
		EI = inputs['EI_mode_elem']
		alpha = inputs['alpha_damp']

		partials['B_struct_77', 'z_sparnode'] = np.zeros((1,14))
		partials['B_struct_77', 'z_towernode'] = np.zeros((1,11))
		partials['B_struct_77', 'x_dd_sparelem'] = np.zeros((1,13))
		partials['B_struct_77', 'x_dd_towerelem'] = np.zeros((1,10))
		partials['B_struct_77', 'EI_mode_elem'] = np.zeros((1,23))
		partials['B_struct_77', 'alpha_damp'] = 0.

		N_sparelem = len(x_dd_sparelem)
		N_towerelem = len(x_dd_towerelem)

		for i in xrange(N_sparelem):
			dz = z_sparnode[i+1] - z_sparnode[i]

			partials['B_struct_77', 'z_sparnode'][0,i] += -alpha * EI[i] * x_dd_sparelem[i]**2.
			partials['B_struct_77', 'z_sparnode'][0,i+1] += alpha * EI[i] * x_dd_sparelem[i]**2.
			partials['B_struct_77', 'x_dd_sparelem'][0,i] += 2. * alpha * dz * EI[i] * x_dd_sparelem[i]
			partials['B_struct_77', 'EI_mode_elem'][0,i] += alpha * dz * x_dd_sparelem[i]**2.
			partials['B_struct_77', 'alpha_damp'] += dz * EI[i] * x_dd_sparelem[i]**2.

		for i in xrange(N_towerelem):
			dz = z_towernode[i+1] - z_towernode[i]

			partials['B_struct_77', 'z_towernode'][0,i] += -alpha * EI[N_sparelem+i] * x_dd_towerelem[i]**2.
			partials['B_struct_77', 'z_towernode'][0,i+1] += alpha * EI[N_sparelem+i] * x_dd_towerelem[i]**2.
			partials['B_struct_77', 'x_dd_towerelem'][0,i] += 2. * alpha * dz * EI[N_sparelem+i] * x_dd_towerelem[i]
			partials['B_struct_77', 'EI_mode_elem'][0,N_sparelem+i] += alpha * dz * x_dd_towerelem[i]**2.
			partials['B_struct_77', 'alpha_damp'] += dz * EI[N_sparelem+i] * x_dd_towerelem[i]**2.