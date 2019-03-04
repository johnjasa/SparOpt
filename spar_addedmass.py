import numpy as np

from openmdao.api import ExplicitComponent

class SparAddedMass(ExplicitComponent):

	def setup(self):
		self.add_input('z_sparnode', val=np.zeros(13), units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('D_spar', np.zeros(10), units='m')

		self.add_output('A11', val=0., units='kg')
		self.add_output('A55', val=0., units='kg*m**2')
		self.add_output('A15', val=0., units='kg*m')

		self.declare_partials('*', '*')

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

	def compute_partials(self, inputs, partials):
		Z_spar = inputs['Z_spar']
		D_spar = inputs['D_spar']
		z_sparnode = inputs['z_sparnode']

		N_elem = len(z_sparnode) - 1

		D = 0.

		partials['A11', 'D_spar'] = np.zeros((1,len(D_spar)))
		partials['A15', 'D_spar'] = np.zeros((1,len(D_spar)))
		partials['A55', 'D_spar'] = np.zeros((1,len(D_spar)))
		partials['A11', 'z_sparnode'] = np.zeros((1,len(z_sparnode)))
		partials['A15', 'z_sparnode'] = np.zeros((1,len(z_sparnode)))
		partials['A55', 'z_sparnode'] = np.zeros((1,len(z_sparnode)))

		for i in xrange(N_elem):
			z = (z_sparnode[i] + z_sparnode[i+1]) / 2
			dz = z_sparnode[i+1] - z_sparnode[i]

			if z <= 0.:
				for j in xrange(len(Z_spar) - 1):
					if (z < Z_spar[j+1]) and (z >= Z_spar[j]):
						D = D_spar[j]
						break

				a11 = 1025. * np.pi / 4. * D**2.

				partials['A11', 'D_spar'][0,j] += dz * 1025. * np.pi / 2. * D
				partials['A15', 'D_spar'][0,j] += dz * 1025. * np.pi / 2. * D * z
				partials['A55', 'D_spar'][0,j] += dz * 1025. * np.pi / 2. * D * z**2.
				partials['A11', 'z_sparnode'][0,i] += -a11
				partials['A15', 'z_sparnode'][0,i] += a11 * (0.5 * dz - z)
				partials['A55', 'z_sparnode'][0,i] += a11 * (z * dz - z**2.)
				partials['A11', 'z_sparnode'][0,i+1] += a11
				partials['A15', 'z_sparnode'][0,i+1] += a11 * (0.5 * dz + z)
				partials['A55', 'z_sparnode'][0,i+1] += a11 * (z * dz + z**2.)