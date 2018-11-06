import numpy as np

from openmdao.api import ExplicitComponent

class ViscousDamping(ExplicitComponent):

	def setup(self):
		self.add_input('Cd', val=0.)
		self.add_input('x_sparelem', val=np.zeros(13), units='m')
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('D_spar', np.zeros(10), units='m')
		self.add_input('stddev_vel_surge', val=0., units='m/s')
		self.add_input('stddev_vel_pitch', val=0., units='rad/s')
		self.add_input('stddev_vel_bend', val=0., units='m/s')

		self.add_output('B_visc_11', val=0., units='N*s/m')
		self.add_output('B_visc_15', val=0., units='N*s/m')
		self.add_output('B_visc_17', val=0., units='N*s/m')
		self.add_output('B_visc_55', val=0., units='N*s/m')
		self.add_output('B_visc_57', val=0., units='N*s/m')
		self.add_output('B_visc_77', val=0., units='N*s/m')

	def compute(self, inputs, outputs):
		Cd = inputs['Cd']
		Z_spar = inputs['Z_spar']
		D_spar = inputs['D_spar']
		z_sparnode = inputs['z_sparnode']
		x_sparelem = inputs['x_sparelem']
		vel_stddev = 0.27509931#inputs['stddev_vel_surge']


		N_elem = len(x_sparelem)

		D = 0.

		outputs['B_visc_11'] = 0.
		outputs['B_visc_15'] = 0.
		outputs['B_visc_17'] = 0.
		outputs['B_visc_55'] = 0.
		outputs['B_visc_57'] = 0.
		outputs['B_visc_77'] = 0.

		for i in xrange(N_elem):
			z = (z_sparnode[i] + z_sparnode[i+1]) / 2
			dz = z_sparnode[i+1] - z_sparnode[i]

			if z <= 0.:
				for j in xrange(len(Z_spar) - 1):
					if (z < Z_spar[j+1]) and (z >= Z_spar[j]):
						D = D_spar[j]
						break

			psi = x_sparelem[i]

			outputs['B_visc_11'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * vel_stddev * D * dz
			outputs['B_visc_15'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * vel_stddev * D * z * dz
			outputs['B_visc_17'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * vel_stddev * D * psi * dz
			outputs['B_visc_55'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * vel_stddev * D * z**2. * dz
			outputs['B_visc_57'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * vel_stddev * D * z * psi * dz
			outputs['B_visc_77'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * vel_stddev * D * psi**2. * dz
		