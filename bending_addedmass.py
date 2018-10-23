from __future__ import division
import numpy as np
import scipy.interpolate as si

from openmdao.api import ExplicitComponent

class BendingAddedMass(ExplicitComponent):

	def setup(self):
		self.add_input('x_sparelem', val=np.zeros(13), units='m')
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('D_spar', np.zeros(10), units='m')

		self.add_output('A17', val=0., units='kg')
		self.add_output('A57', val=0., units='kg*m')
		self.add_output('A77', val=0., units='kg')

	def compute(self, inputs, outputs):
		Z_spar = inputs['Z_spar']
		spar_draft = inputs['spar_draft']
		D_spar = inputs['D_spar']
		z_sparnode = inputs['z_sparnode']
		x_sparelem = inputs['x_sparelem']

		N_spar = len(x_sparelem)

		D = 0.

		outputs['A17'] = 0.
		outputs['A57'] = 0.
		outputs['A77'] = 0.

		for i in xrange(N_spar):
			z = (z_sparnode[i] + z_sparnode[i+1]) / 2
			dz = z_sparnode[i+1] - z_sparnode[i]
			for j in xrange(len(Z_spar) - 1):
				if (z < Z_spar[j+1]) and (z >= Z_spar[j]):
					D = D_spar[j]
					break
			if z <= 0.:
				a11 = 1025. * np.pi / 4. * D**2.
			else:
				a11 = 0.

			outputs['A17'] += dz * a11 * x_sparelem[i]
			outputs['A57'] += dz * a11 * z * x_sparelem[i]
			outputs['A77'] += dz * a11 * x_sparelem[i]**2.