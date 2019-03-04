from __future__ import division
import numpy as np
import scipy.interpolate as si

from openmdao.api import ExplicitComponent

class BendingAddedMass(ExplicitComponent):

	def setup(self):
		self.add_input('x_sparelem', val=np.zeros(12), units='m')
		self.add_input('z_sparnode', val=np.zeros(13), units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('D_spar', np.zeros(10), units='m')

		self.add_output('A17', val=0., units='kg')
		self.add_output('A57', val=0., units='kg*m')
		self.add_output('A77', val=0., units='kg')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		Z_spar = inputs['Z_spar']
		D_spar = inputs['D_spar']
		z_sparnode = inputs['z_sparnode']
		x_sparelem = inputs['x_sparelem']

		N_elem = len(x_sparelem)

		D = 0.

		outputs['A17'] = 0.
		outputs['A57'] = 0.
		outputs['A77'] = 0.

		for i in xrange(N_elem):
			z = (z_sparnode[i] + z_sparnode[i+1]) / 2
			dz = z_sparnode[i+1] - z_sparnode[i]

			if z <= 0.:
				for j in xrange(len(Z_spar) - 1):
					if (z < Z_spar[j+1]) and (z >= Z_spar[j]):
						D = D_spar[j]
						break

				a11 = 1025. * np.pi / 4. * D**2.

				outputs['A17'] += dz * a11 * x_sparelem[i]
				outputs['A57'] += dz * a11 * z * x_sparelem[i]
				outputs['A77'] += dz * a11 * x_sparelem[i]**2.

	def compute_partials(self, inputs, partials):
		Z_spar = inputs['Z_spar']
		D_spar = inputs['D_spar']
		z_sparnode = inputs['z_sparnode']
		x_sparelem = inputs['x_sparelem']

		N_elem = len(x_sparelem)

		D = 0.

		partials['A17', 'x_sparelem'] = np.zeros((1,12))
		partials['A17', 'z_sparnode'] = np.zeros((1,13))
		partials['A17', 'D_spar'] = np.zeros((1,10))

		partials['A57', 'x_sparelem'] = np.zeros((1,12))
		partials['A57', 'z_sparnode'] = np.zeros((1,13))
		partials['A57', 'D_spar'] = np.zeros((1,10))

		partials['A77', 'x_sparelem'] = np.zeros((1,12))
		partials['A77', 'z_sparnode'] = np.zeros((1,13))
		partials['A77', 'D_spar'] = np.zeros((1,10))

		for i in xrange(N_elem):
			z = (z_sparnode[i] + z_sparnode[i+1]) / 2
			dz = z_sparnode[i+1] - z_sparnode[i]
			if z <= 0.:
				for j in xrange(len(Z_spar) - 1):
					if (z < Z_spar[j+1]) and (z >= Z_spar[j]):
						D = D_spar[j]
						partials['A17', 'D_spar'][0,j] += dz * x_sparelem[i] * 1025. * np.pi / 2. * D
						partials['A57', 'D_spar'][0,j] += dz * z * x_sparelem[i] * 1025. * np.pi / 2. * D
						partials['A77', 'D_spar'][0,j] += dz * x_sparelem[i]**2. * 1025. * np.pi / 2. * D
						break

				a11 = 1025. * np.pi / 4. * D**2.

				partials['A17', 'z_sparnode'][0,i] += -a11 * x_sparelem[i]
				partials['A57', 'z_sparnode'][0,i] += -a11 * z * x_sparelem[i] + dz * a11 * x_sparelem[i] * 0.5
				partials['A77', 'z_sparnode'][0,i] += -a11 * x_sparelem[i]**2.
				partials['A17', 'z_sparnode'][0,i+1] += a11 * x_sparelem[i]
				partials['A57', 'z_sparnode'][0,i+1] += a11 * z * x_sparelem[i] + dz * a11 * x_sparelem[i] * 0.5
				partials['A77', 'z_sparnode'][0,i+1] += a11 * x_sparelem[i]**2.

				partials['A17', 'x_sparelem'][0,i] += dz * a11
				partials['A57', 'x_sparelem'][0,i] += dz * a11 * z
				partials['A77', 'x_sparelem'][0,i] += dz * a11 * 2. * x_sparelem[i]