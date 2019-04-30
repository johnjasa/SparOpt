import numpy as np

from openmdao.api import ExplicitComponent

class ViscousDamping(ExplicitComponent):

	def setup(self):
		self.add_input('Cd', val=0.)
		self.add_input('x_sparelem', val=np.zeros(12), units='m')
		self.add_input('z_sparnode', val=np.zeros(13), units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('D_spar', np.zeros(10), units='m')
		self.add_input('stddev_vel_distr', val=np.zeros(12), units='m/s')

		self.add_output('B_visc_11', val=0., units='N*s/m')
		self.add_output('B_visc_15', val=0., units='N*s')
		self.add_output('B_visc_17', val=0., units='N*s/m')
		self.add_output('B_visc_55', val=0., units='N*s*m')
		self.add_output('B_visc_57', val=0., units='N*s')
		self.add_output('B_visc_77', val=0., units='N*s/m')

		self.declare_partials('B_visc_11', 'Cd', val = 0.)
		self.declare_partials('B_visc_11', 'x_sparelem', val = np.zeros((1,12)))
		self.declare_partials('B_visc_11', 'z_sparnode', val = np.zeros((1,13)))
		self.declare_partials('B_visc_11', 'Z_spar', val = np.zeros((1,11)))
		self.declare_partials('B_visc_11', 'D_spar', val = np.zeros((1,10)))
		self.declare_partials('B_visc_11', 'stddev_vel_distr', val = np.zeros((1,12)))
		self.declare_partials('B_visc_15', 'Cd', val = 0.)
		self.declare_partials('B_visc_15', 'x_sparelem', val = np.zeros((1,12)))
		self.declare_partials('B_visc_15', 'z_sparnode', val = np.zeros((1,13)))
		self.declare_partials('B_visc_15', 'Z_spar', val = np.zeros((1,11)))
		self.declare_partials('B_visc_15', 'D_spar', val = np.zeros((1,10)))
		self.declare_partials('B_visc_15', 'stddev_vel_distr', val = np.zeros((1,12)))
		self.declare_partials('B_visc_17', 'Cd', val = 0.)
		self.declare_partials('B_visc_17', 'x_sparelem', val = np.zeros((1,12)))
		self.declare_partials('B_visc_17', 'z_sparnode', val = np.zeros((1,13)))
		self.declare_partials('B_visc_17', 'Z_spar', val = np.zeros((1,11)))
		self.declare_partials('B_visc_17', 'D_spar', val = np.zeros((1,10)))
		self.declare_partials('B_visc_17', 'stddev_vel_distr', val = np.zeros((1,12)))
		self.declare_partials('B_visc_55', 'Cd', val = 0.)
		self.declare_partials('B_visc_55', 'x_sparelem', val = np.zeros((1,12)))
		self.declare_partials('B_visc_55', 'z_sparnode', val = np.zeros((1,13)))
		self.declare_partials('B_visc_55', 'Z_spar', val = np.zeros((1,11)))
		self.declare_partials('B_visc_55', 'D_spar', val = np.zeros((1,10)))
		self.declare_partials('B_visc_55', 'stddev_vel_distr', val = np.zeros((1,12)))
		self.declare_partials('B_visc_57', 'Cd', val = 0.)
		self.declare_partials('B_visc_57', 'x_sparelem', val = np.zeros((1,12)))
		self.declare_partials('B_visc_57', 'z_sparnode', val = np.zeros((1,13)))
		self.declare_partials('B_visc_57', 'Z_spar', val = np.zeros((1,11)))
		self.declare_partials('B_visc_57', 'D_spar', val = np.zeros((1,10)))
		self.declare_partials('B_visc_57', 'stddev_vel_distr', val = np.zeros((1,12)))
		self.declare_partials('B_visc_77', 'Cd', val = 0.)
		self.declare_partials('B_visc_77', 'x_sparelem', val = np.zeros((1,12)))
		self.declare_partials('B_visc_77', 'z_sparnode', val = np.zeros((1,13)))
		self.declare_partials('B_visc_77', 'Z_spar', val = np.zeros((1,11)))
		self.declare_partials('B_visc_77', 'D_spar', val = np.zeros((1,10)))
		self.declare_partials('B_visc_77', 'stddev_vel_distr', val = np.zeros((1,12)))

	def compute(self, inputs, outputs):
		Cd = inputs['Cd']
		Z_spar = inputs['Z_spar']
		D_spar = inputs['D_spar']
		z_sparnode = inputs['z_sparnode']
		x_sparelem = inputs['x_sparelem']
		stddev_vel_distr = inputs['stddev_vel_distr'] #np.array([0.17856366, 0.14913746, 0.13331558, 0.12660471, 0.12420269, 0.12422336, 0.13459537, 0.15813834, 0.18982264, 0.22625248, 0.25732357, 0.257976, 0.2964546])

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

				outputs['B_visc_11'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * dz
				outputs['B_visc_15'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * z * dz
				outputs['B_visc_17'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * x_sparelem[i] * dz
				outputs['B_visc_55'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * z**2. * dz
				outputs['B_visc_57'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * z * x_sparelem[i] * dz
				outputs['B_visc_77'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * x_sparelem[i]**2. * dz

	def compute_partials(self, inputs, partials):
		Cd = inputs['Cd']
		Z_spar = inputs['Z_spar']
		D_spar = inputs['D_spar']
		z_sparnode = inputs['z_sparnode']
		x_sparelem = inputs['x_sparelem']
		stddev_vel_distr = inputs['stddev_vel_distr'] #np.array([0.17856366, 0.14913746, 0.13331558, 0.12660471, 0.12420269, 0.12422336, 0.13459537, 0.15813834, 0.18982264, 0.22625248, 0.25732357, 0.257976, 0.2964546])


		N_elem = len(x_sparelem)

		D = 0.

		partials['B_visc_11', 'Cd'][:] = 0.
		partials['B_visc_11', 'x_sparelem'][:] = 0.
		partials['B_visc_11', 'z_sparnode'][:] = 0.
		partials['B_visc_11', 'Z_spar'][:] = 0.
		partials['B_visc_11', 'D_spar'][:] = 0.
		partials['B_visc_11', 'stddev_vel_distr'][:] = 0.
		partials['B_visc_15', 'Cd'][:] = 0.
		partials['B_visc_15', 'x_sparelem'][:] = 0.
		partials['B_visc_15', 'z_sparnode'][:] = 0.
		partials['B_visc_15', 'Z_spar'][:] = 0.
		partials['B_visc_15', 'D_spar'][:] = 0.
		partials['B_visc_15', 'stddev_vel_distr'][:] = 0.
		partials['B_visc_17', 'Cd'][:] = 0.
		partials['B_visc_17', 'x_sparelem'][:] = 0.
		partials['B_visc_17', 'z_sparnode'][:] = 0.
		partials['B_visc_17', 'Z_spar'][:] = 0.
		partials['B_visc_17', 'D_spar'][:] = 0.
		partials['B_visc_17', 'stddev_vel_distr'][:] = 0.
		partials['B_visc_55', 'Cd'][:] = 0.
		partials['B_visc_55', 'x_sparelem'][:] = 0.
		partials['B_visc_55', 'z_sparnode'][:] = 0.
		partials['B_visc_55', 'Z_spar'][:] = 0.
		partials['B_visc_55', 'D_spar'][:] = 0.
		partials['B_visc_55', 'stddev_vel_distr'][:] = 0.
		partials['B_visc_57', 'Cd'][:] = 0.
		partials['B_visc_57', 'x_sparelem'][:] = 0.
		partials['B_visc_57', 'z_sparnode'][:] = 0.
		partials['B_visc_57', 'Z_spar'][:] = 0.
		partials['B_visc_57', 'D_spar'][:] = 0.
		partials['B_visc_57', 'stddev_vel_distr'][:] = 0.
		partials['B_visc_77', 'Cd'][:] = 0.
		partials['B_visc_77', 'x_sparelem'][:] = 0.
		partials['B_visc_77', 'z_sparnode'][:] = 0.
		partials['B_visc_77', 'Z_spar'][:] = 0.
		partials['B_visc_77', 'D_spar'][:] = 0.
		partials['B_visc_77', 'stddev_vel_distr'][:] = 0.

		for i in xrange(N_elem):
			z = (z_sparnode[i] + z_sparnode[i+1]) / 2
			dz = z_sparnode[i+1] - z_sparnode[i]

			if z <= 0.:
				for j in xrange(len(Z_spar) - 1):
					if (z < Z_spar[j+1]) and (z >= Z_spar[j]):
						D = D_spar[j]
						break

				partials['B_visc_11', 'Cd'] += 0.5 * 1025. * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * dz
				partials['B_visc_11', 'z_sparnode'][0,i] += -0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D
				partials['B_visc_11', 'z_sparnode'][0,i+1] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D
				partials['B_visc_11', 'D_spar'][0,j] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * dz
				partials['B_visc_11', 'stddev_vel_distr'][0,i] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * D * dz

				partials['B_visc_15', 'Cd'] += 0.5 * 1025. * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * z * dz
				partials['B_visc_15', 'z_sparnode'][0,i] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * (-z + dz * 0.5)
				partials['B_visc_15', 'z_sparnode'][0,i+1] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * (z + dz * 0.5 )
				partials['B_visc_15', 'D_spar'][0,j] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * z * dz
				partials['B_visc_15', 'stddev_vel_distr'][0,i] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * D * z * dz

				partials['B_visc_17', 'Cd'] += 0.5 * 1025. * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * x_sparelem[i] * dz
				partials['B_visc_17', 'z_sparnode'][0,i] += -0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * x_sparelem[i]
				partials['B_visc_17', 'z_sparnode'][0,i+1] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * x_sparelem[i]
				partials['B_visc_17', 'D_spar'][0,j] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * x_sparelem[i] * dz
				partials['B_visc_17', 'stddev_vel_distr'][0,i] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * D * x_sparelem[i] * dz
				partials['B_visc_17', 'x_sparelem'][0,i] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * dz

				partials['B_visc_55', 'Cd'] += 0.5 * 1025. * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * z**2. * dz
				partials['B_visc_55', 'z_sparnode'][0,i] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * (-z**2. + z * dz)
				partials['B_visc_55', 'z_sparnode'][0,i+1] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * (z**2. + z * dz)
				partials['B_visc_55', 'D_spar'][0,j] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * z**2. * dz
				partials['B_visc_55', 'stddev_vel_distr'][0,i] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * D * z**2. * dz

				partials['B_visc_57', 'Cd'] += 0.5 * 1025. * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * z * x_sparelem[i] * dz
				partials['B_visc_57', 'z_sparnode'][0,i] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * x_sparelem[i] * (-z + 0.5 * dz)
				partials['B_visc_57', 'z_sparnode'][0,i+1] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * x_sparelem[i] * (z + 0.5 * dz)
				partials['B_visc_57', 'D_spar'][0,j] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * z * x_sparelem[i] * dz
				partials['B_visc_57', 'stddev_vel_distr'][0,i] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * D * z * x_sparelem[i] * dz
				partials['B_visc_57', 'x_sparelem'][0,i] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * z * dz

				partials['B_visc_77', 'Cd'] += 0.5 * 1025. * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * x_sparelem[i]**2. * dz
				partials['B_visc_77', 'z_sparnode'][0,i] += -0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * x_sparelem[i]**2.
				partials['B_visc_77', 'z_sparnode'][0,i+1] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * x_sparelem[i]**2.
				partials['B_visc_77', 'D_spar'][0,j] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * x_sparelem[i]**2. * dz
				partials['B_visc_77', 'stddev_vel_distr'][0,i] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * D * x_sparelem[i]**2. * dz
				partials['B_visc_77', 'x_sparelem'][0,i] += 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_distr[i] * D * x_sparelem[i] * dz
