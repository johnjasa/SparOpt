import numpy as np

from openmdao.api import ExplicitComponent

class MeanTowerStress(ExplicitComponent):

	def setup(self):
		self.add_input('thrust_0', val=0., units='N')
		self.add_input('D_tower_p', val=np.zeros(11), units='m')
		self.add_input('wt_tower_p', val=np.zeros(11), units='m')
		self.add_input('M_turb', val=0., units='kg')
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('Z_tower', val=np.zeros(11), units='m')

		self.add_output('mean_tower_stress', val=np.zeros(11), units='MPa')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		thrust_0 = inputs['thrust_0']
		D_tower_p = inputs['D_tower_p']
		wt_tower_p = inputs['wt_tower_p']
		M_turb = inputs['M_turb']
		M_tower = inputs['M_tower']
		CoG_rotor = inputs['CoG_rotor']
		Z_tower = inputs['Z_tower']

		for i in xrange(11):
			A = np.pi / 4. * (D_tower_p[i]**2. - (D_tower_p[i] - 2. * wt_tower_p[i])**2.)
			I = np.pi / 64. * (D_tower_p[i]**4. - (D_tower_p[i] - 2. * wt_tower_p[i])**4.)

			outputs['mean_tower_stress'][i] = (CoG_rotor - Z_tower[i]) * thrust_0 / I * (D_tower_p[i] / 2.) * 10.**(-6.) + (M_turb - np.sum(M_tower[:i])) * 9.80665 / A * 10.**(-6.) #downwind (compression) side, assumed to be worst 

	def compute_partials(self, inputs, partials):
		thrust_0 = inputs['thrust_0']
		D_tower_p = inputs['D_tower_p']
		wt_tower_p = inputs['wt_tower_p']
		M_turb = inputs['M_turb']
		M_tower = inputs['M_tower']
		CoG_rotor = inputs['CoG_rotor']
		Z_tower = inputs['Z_tower']

		partials['mean_tower_stress', 'thrust_0'] = np.zeros((11,1))
		partials['mean_tower_stress', 'D_tower_p'] = np.zeros((11,11))
		partials['mean_tower_stress', 'wt_tower_p'] = np.zeros((11,11))
		partials['mean_tower_stress', 'M_turb'] = np.zeros((11,1))
		partials['mean_tower_stress', 'M_tower'] = np.zeros((11,10))
		partials['mean_tower_stress', 'CoG_rotor'] = np.zeros((11,1))
		partials['mean_tower_stress', 'Z_tower'] = np.zeros((11,11))

		for i in xrange(11):
			A = np.pi / 4. * (D_tower_p[i]**2. - (D_tower_p[i] - 2. * wt_tower_p[i])**2.)
			I = np.pi / 64. * (D_tower_p[i]**4. - (D_tower_p[i] - 2. * wt_tower_p[i])**4.)

			partials['mean_tower_stress', 'thrust_0'][i,0] += (CoG_rotor - Z_tower[i]) / I * (D_tower_p[i] / 2.) * 10.**(-6.)
			partials['mean_tower_stress', 'D_tower_p'][i,i] += -(CoG_rotor - Z_tower[i]) * thrust_0 / I**2. * (D_tower_p[i] / 2.) * 10.**(-6.) * np.pi / 16. * (D_tower_p[i]**3. - (D_tower_p[i] - 2. * wt_tower_p[i])**3.) + (CoG_rotor - Z_tower[i]) * thrust_0 / I * (1. / 2.) * 10.**(-6.) - (M_turb - np.sum(M_tower[:i])) * 9.80665 / A**2. * 10.**(-6.) * np.pi / 2. * (D_tower_p[i] - (D_tower_p[i] - 2. * wt_tower_p[i]))
			partials['mean_tower_stress', 'wt_tower_p'][i,i] += -(CoG_rotor - Z_tower[i]) * thrust_0 / I**2. * (D_tower_p[i] / 2.) * 10.**(-6.) * np.pi / 8. * (D_tower_p[i] - 2. * wt_tower_p[i])**3. - (M_turb - np.sum(M_tower[:i])) * 9.80665 / A**2. * 10.**(-6.) * np.pi * (D_tower_p[i] - 2. * wt_tower_p[i])
			partials['mean_tower_stress', 'M_turb'][i,0] += 9.80665 / A * 10.**(-6.)
			partials['mean_tower_stress', 'CoG_rotor'][i,0] += thrust_0 / I * (D_tower_p[i] / 2.) * 10.**(-6.)
			partials['mean_tower_stress', 'Z_tower'][i,i] += -thrust_0 / I * (D_tower_p[i] / 2.) * 10.**(-6.)

			for j in xrange(i):
				partials['mean_tower_stress', 'M_tower'][i,j] += -9.80665 / A * 10.**(-6.)