import numpy as np

from openmdao.api import ExplicitComponent

class MeanTowerStress(ExplicitComponent):

	def setup(self):
		self.add_input('thrust_0', val=0., units='N')
		self.add_input('D_tower_p', val=np.zeros(11), units='m')
		self.add_input('wt_tower_p', val=np.zeros(11), units='m')
		self.add_input('M_turb', val=0., units='kg')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('CoG_nacelle', val=0., units='m')
		self.add_input('mean_pitch', val=0., units='rad')
		self.add_input('Z_tower', val=np.zeros(11), units='m')

		self.add_output('mean_tower_stress', val=np.zeros(10), units='MPa')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		thrust_0 = inputs['thrust_0']
		D_tower_p = inputs['D_tower_p']
		wt_tower_p = inputs['wt_tower_p']
		M_turb = inputs['M_turb']
		M_rotor = inputs['M_rotor']
		M_nacelle = inputs['M_nacelle']
		M_tower = inputs['M_tower']
		CoG_rotor = inputs['CoG_rotor']
		CoG_nacelle = inputs['CoG_nacelle']
		mean_pitch = inputs['mean_pitch']
		Z_tower = inputs['Z_tower']

		z_tower = np.zeros(10)

		for i in xrange(10):
			z_tower[i] = (Z_tower[i] + Z_tower[i+1]) / 2.

		for i in xrange(10):
			A = np.pi / 4. * (D_tower_p[i]**2. - (D_tower_p[i] - 2. * wt_tower_p[i])**2.)
			I = np.pi / 64. * (D_tower_p[i]**4. - (D_tower_p[i] - 2. * wt_tower_p[i])**4.)

			outputs['mean_tower_stress'][i] = -((CoG_rotor - Z_tower[i]) * thrust_0 + (M_rotor * (CoG_rotor - Z_tower[i]) + M_nacelle * (CoG_nacelle - Z_tower[i]) + np.sum(M_tower[i:] * (z_tower[i:] - Z_tower[i]))) * 9.80665 * np.sin(mean_pitch)) / I * (D_tower_p[i] / 2.) * 10.**(-6.) - (M_turb - np.sum(M_tower[:i])) * 9.80665 / A * 10.**(-6.) #downwind (compression) side, assumed to be worst 

	def compute_partials(self, inputs, partials):
		thrust_0 = inputs['thrust_0']
		D_tower_p = inputs['D_tower_p']
		wt_tower_p = inputs['wt_tower_p']
		M_turb = inputs['M_turb']
		M_rotor = inputs['M_rotor']
		M_nacelle = inputs['M_nacelle']
		M_tower = inputs['M_tower']
		CoG_rotor = inputs['CoG_rotor']
		CoG_nacelle = inputs['CoG_nacelle']
		mean_pitch = inputs['mean_pitch']
		Z_tower = inputs['Z_tower']

		partials['mean_tower_stress', 'thrust_0'] = np.zeros((10,1))
		partials['mean_tower_stress', 'D_tower_p'] = np.zeros((10,11))
		partials['mean_tower_stress', 'wt_tower_p'] = np.zeros((10,11))
		partials['mean_tower_stress', 'M_turb'] = np.zeros((10,1))
		partials['mean_tower_stress', 'M_rotor'] = np.zeros((10,1))
		partials['mean_tower_stress', 'M_nacelle'] = np.zeros((10,1))
		partials['mean_tower_stress', 'M_tower'] = np.zeros((10,10))
		partials['mean_tower_stress', 'CoG_rotor'] = np.zeros((10,1))
		partials['mean_tower_stress', 'CoG_nacelle'] = np.zeros((10,1))
		partials['mean_tower_stress', 'mean_pitch'] = np.zeros((10,1))
		partials['mean_tower_stress', 'Z_tower'] = np.zeros((10,11))

		z_tower = np.zeros(10)

		for i in xrange(10):
			z_tower[i] = (Z_tower[i] + Z_tower[i+1]) / 2.

		for i in xrange(10):
			A = np.pi / 4. * (D_tower_p[i]**2. - (D_tower_p[i] - 2. * wt_tower_p[i])**2.)
			I = np.pi / 64. * (D_tower_p[i]**4. - (D_tower_p[i] - 2. * wt_tower_p[i])**4.)

			partials['mean_tower_stress', 'thrust_0'][i,0] += -(CoG_rotor - Z_tower[i]) / I * (D_tower_p[i] / 2.) * 10.**(-6.)
			partials['mean_tower_stress', 'D_tower_p'][i,i] += ((CoG_rotor - Z_tower[i]) * thrust_0 + (M_rotor * (CoG_rotor - Z_tower[i]) + M_nacelle * (CoG_nacelle - Z_tower[i]) + np.sum(M_tower[i:] * (z_tower[i:] - Z_tower[i]))) * 9.80665 * np.sin(mean_pitch)) / I**2. * (D_tower_p[i] / 2.) * 10.**(-6.) * np.pi / 16. * (D_tower_p[i]**3. - (D_tower_p[i] - 2. * wt_tower_p[i])**3.) - ((CoG_rotor - Z_tower[i]) * thrust_0 + (M_rotor * (CoG_rotor - Z_tower[i]) + M_nacelle * (CoG_nacelle - Z_tower[i]) + np.sum(M_tower[i:] * (z_tower[i:] - Z_tower[i]))) * 9.80665 * np.sin(mean_pitch)) / I * (1. / 2.) * 10.**(-6.) + (M_turb - np.sum(M_tower[:i])) * 9.80665 / A**2. * 10.**(-6.) * np.pi / 2. * (D_tower_p[i] - (D_tower_p[i] - 2. * wt_tower_p[i]))
			partials['mean_tower_stress', 'wt_tower_p'][i,i] += ((CoG_rotor - Z_tower[i]) * thrust_0 + (M_rotor * (CoG_rotor - Z_tower[i]) + M_nacelle * (CoG_nacelle - Z_tower[i]) + np.sum(M_tower[i:] * (z_tower[i:] - Z_tower[i]))) * 9.80665 * np.sin(mean_pitch)) / I**2. * (D_tower_p[i] / 2.) * 10.**(-6.) * np.pi / 8. * (D_tower_p[i] - 2. * wt_tower_p[i])**3. + (M_turb - np.sum(M_tower[:i])) * 9.80665 / A**2. * 10.**(-6.) * np.pi * (D_tower_p[i] - 2. * wt_tower_p[i])
			partials['mean_tower_stress', 'M_turb'][i,0] += -9.80665 / A * 10.**(-6.)
			partials['mean_tower_stress', 'M_rotor'][i,0] += -(CoG_rotor - Z_tower[i]) * 9.80665 * np.sin(mean_pitch) / I * (D_tower_p[i] / 2.) * 10.**(-6.)
			partials['mean_tower_stress', 'M_nacelle'][i,0] += -(CoG_nacelle - Z_tower[i]) * 9.80665 * np.sin(mean_pitch) / I * (D_tower_p[i] / 2.) * 10.**(-6.)
			partials['mean_tower_stress', 'CoG_rotor'][i,0] += -(thrust_0 + M_rotor * 9.80665 * np.sin(mean_pitch)) / I * (D_tower_p[i] / 2.) * 10.**(-6.)
			partials['mean_tower_stress', 'CoG_nacelle'][i,0] += -(M_nacelle * 9.80665 * np.sin(mean_pitch)) / I * (D_tower_p[i] / 2.) * 10.**(-6.)
			partials['mean_tower_stress', 'mean_pitch'][i,0] += -(M_rotor * (CoG_rotor - Z_tower[i]) + M_nacelle * (CoG_nacelle - Z_tower[i]) + np.sum(M_tower[i:] * (z_tower[i:] - Z_tower[i]))) * 9.80665 * np.cos(mean_pitch) / I * (D_tower_p[i] / 2.) * 10.**(-6.)
			partials['mean_tower_stress', 'Z_tower'][i,i] += (thrust_0 + (M_rotor + M_nacelle + np.sum(M_tower[i:])) * 9.80665 * np.sin(mean_pitch)) / I * (D_tower_p[i] / 2.) * 10.**(-6.)

			for j in xrange(i,10):
				partials['mean_tower_stress', 'Z_tower'][i,j] += -M_tower[j] * 9.80665 * np.sin(mean_pitch) / 2. / I * (D_tower_p[i] / 2.) * 10.**(-6.)
				partials['mean_tower_stress', 'Z_tower'][i,j+1] += -M_tower[j] * 9.80665 * np.sin(mean_pitch) / 2. / I * (D_tower_p[i] / 2.) * 10.**(-6.)
				partials['mean_tower_stress', 'M_tower'][i,j] += -(z_tower[j] - Z_tower[i]) * 9.80665 * np.sin(mean_pitch) / I * (D_tower_p[i] / 2.) * 10.**(-6.)

			for j in xrange(i):
				partials['mean_tower_stress', 'M_tower'][i,j] += 9.80665 / A * 10.**(-6.)