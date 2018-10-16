from __future__ import division
import numpy as np
import scipy.interpolate as si

from openmdao.api import ExplicitComponent

class BendingDamping(ExplicitComponent):

	def setup(self):
		self.add_input('D_tower', val=np.zeros(10), units='m')
		self.add_input('wt_tower', val=np.zeros(10), units='m')
		self.add_input('L_tower', val=np.zeros(10), units='m')
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('x_towermode', val=np.zeros(11), units='m')
		self.add_input('z_towermode', val=np.zeros(11), units='m')

		self.add_output('B_struct_77', val=0., units='N*s/m')

	def compute(self, inputs, outputs):
		D_tower = inputs['D_tower']
		wt_tower = inputs['wt_tower']
		L_tower = inputs['L_tower']
		Z_tower = inputs['Z_tower']

		f_psi = si.UnivariateSpline(inputs['z_towermode'], inputs['x_towermode'], s=0)
		f_psi_d = f_psi.derivative(n=1)
		f_psi_dd = f_psi.derivative(n=2)

		EI_tower = np.pi / 64. * (D_tower**4. - (D_tower - 2. * wt_tower)**4.) * 2.1e11

		EI = 0.

		N_elem = 200

		outputs['B_struct_77'] = 0.

		#TODO: Bending stiffness and damping in spar currently neglected as it is assumed to be rigid
		#TODO: Stiffness-proportional Rayleigh damping coefficient should not be hard-coded

		for i in xrange(N_elem):
			z = 10. + (i + 0.5) / N_elem * np.sum(L_tower)
			dz = np.sum(L_tower) / N_elem
			for j in xrange(len(Z_tower)-1):
				if (z < Z_tower[j+1]) and (z >= Z_tower[j]):
					EI = EI_tower[j]
					break

			outputs['B_struct_77'] += 0.007 * dz * EI * f_psi_dd(z)**2.