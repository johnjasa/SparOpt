from __future__ import division
import numpy as np
import scipy.interpolate as si

from openmdao.api import ExplicitComponent

class BendingAddedMass(ExplicitComponent):

	def setup(self):
		self.add_input('x_sparmode', val=np.zeros(7), units='m')
		self.add_input('z_sparmode', val=np.zeros(7), units='m')
		self.add_input('Z_spar', val=np.zeros(4), units='m')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('D_secs', np.zeros(3), units='m')

		self.add_output('A17', val=0., units='kg')
		self.add_output('A57', val=0., units='kg*m')
		self.add_output('A77', val=0., units='kg')

	def compute(self, inputs, outputs):
		Z_spar = inputs['Z_spar']
		spar_draft = inputs['spar_draft']
		D_secs = inputs['D_secs']

		f_psi_spar = si.UnivariateSpline(inputs['z_sparmode'], inputs['x_sparmode'], s=0)

		N_elem = 200

		D = 0.

		outputs['A17'] = 0.
		outputs['A57'] = 0.
		outputs['A77'] = 0.

		for i in xrange(N_elem):
			z = -spar_draft + (i + 0.5) / N_elem * spar_draft
			dz = spar_draft / N_elem
			for j in xrange(len(Z_spar) - 1):
				if (z < Z_spar[j+1]) and (z >= Z_spar[j]):
					D = D_secs[j]
					break

			a11 = 1025. * np.pi / 4. * D**2.

			outputs['A17'] += dz * a11 * f_psi_spar(z)
			outputs['A57'] += dz * a11 * z * f_psi_spar(z)
			outputs['A77'] += dz * a11 * f_psi_spar(z)**2.