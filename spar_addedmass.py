import numpy as np

from openmdao.api import ExplicitComponent

class SparAddedMass(ExplicitComponent):

	def setup(self):
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('D_spar', np.zeros(10), units='m')

		self.add_output('A11', val=0., units='kg')
		self.add_output('A55', val=0., units='kg*m**2')
		self.add_output('A15', val=0., units='kg*m')

	def compute(self, inputs, outputs):
		Z_spar = inputs['Z_spar']
		D_spar = inputs['D_spar']
		z_sparnode = inputs['z_sparnode']

		N_elem = len(z_sparnode) - 1

		D = 0.

		outputs['A11'] = 0.
		outputs['A15'] = 0.
		outputs['A55'] = 0.

		for i in xrange(N_elem):
			z = (z_sparnode[i] + z_sparnode[i+1]) / 2
			dz = z_sparnode[i+1] - z_sparnode[i]

			if z <= 0.:
				for j in xrange(len(Z_spar) - 1):
					if (z < Z_spar[j+1]) and (z >= Z_spar[j]):
						D = D_spar[j]
						break

				a11 = 1025. * np.pi / 4. * D**2.

				outputs['A11'] += dz * a11
				outputs['A15'] += dz * a11 * z
				outputs['A55'] += dz * a11 * z**2.