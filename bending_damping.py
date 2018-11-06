from __future__ import division
import numpy as np
import scipy.interpolate as si

from openmdao.api import ExplicitComponent

class BendingDamping(ExplicitComponent):

	def setup(self):
		self.add_input('D_tower', val=np.zeros(10), units='m')
		self.add_input('wt_tower', val=np.zeros(10), units='m')
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('x_dd_towerelem', val=np.zeros(10), units='1/m')
		self.add_input('z_towernode', val=np.zeros(11), units='m')

		self.add_output('B_struct_77', val=0., units='N*s/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_tower = inputs['D_tower']
		wt_tower = inputs['wt_tower']
		Z_tower = inputs['Z_tower']
		z_towernode = inputs['z_towernode']
		x_dd_towerelem = inputs['x_dd_towerelem']

		EI_tower = np.pi / 64. * (D_tower**4. - (D_tower - 2. * wt_tower)**4.) * 2.1e11

		EI = 0.

		N_elem = len(x_dd_towerelem)

		outputs['B_struct_77'] = 0.

		#TODO: Bending stiffness and damping in spar currently neglected as it is assumed to be rigid
		#TODO: Stiffness-proportional Rayleigh damping coefficient should not be hard-coded

		for i in xrange(N_elem):
			z = (z_towernode[i] + z_towernode[i+1]) / 2
			dz = z_towernode[i+1] - z_towernode[i]
			for j in xrange(len(Z_tower)-1):
				if (z < Z_tower[j+1]) and (z >= Z_tower[j]):
					EI = EI_tower[j]
					break

			outputs['B_struct_77'] += 0.007 * dz * EI * x_dd_towerelem[i]**2.

	def compute_partials(self, inputs, partials):
		D_tower = inputs['D_tower']
		wt_tower = inputs['wt_tower']
		Z_tower = inputs['Z_tower']
		z_towernode = inputs['z_towernode']
		x_dd_towerelem = inputs['x_dd_towerelem']

		EI_tower = np.pi / 64. * (D_tower**4. - (D_tower - 2. * wt_tower)**4.) * 2.1e11

		EI = 0.

		N_elem = len(x_dd_towerelem)

		partials['B_struct_77', 'D_tower'] = np.zeros((1,10))
		partials['B_struct_77', 'wt_tower'] = np.zeros((1,10))
		partials['B_struct_77', 'z_towernode'] = np.zeros((1,11))
		partials['B_struct_77', 'x_dd_towerelem'] = np.zeros((1,10))

		for i in xrange(N_elem):
			z = (z_towernode[i] + z_towernode[i+1]) / 2
			dz = z_towernode[i+1] - z_towernode[i]
			for j in xrange(len(Z_tower)-1):
				if (z < Z_tower[j+1]) and (z >= Z_tower[j]):
					EI = EI_tower[j]
					partials['B_struct_77', 'D_tower'][0,j] += 0.007 * dz * x_dd_towerelem[i]**2. * np.pi / 16. * (D_tower[j]**3. - (D_tower[j] - 2. * wt_tower[j])**3.) * 2.1e11
					partials['B_struct_77', 'wt_tower'][0,j] += 0.007 * dz * x_dd_towerelem[i]**2. * np.pi / 8. * (D_tower[j] - 2. * wt_tower[j])**3. * 2.1e11
					break

			partials['B_struct_77', 'z_towernode'][0,i] += -0.007 * EI * x_dd_towerelem[i]**2.
			partials['B_struct_77', 'z_towernode'][0,i+1] += 0.007 * EI * x_dd_towerelem[i]**2.
			partials['B_struct_77', 'x_dd_towerelem'][0,i] += 0.007 * dz * EI * 2. * x_dd_towerelem[i]