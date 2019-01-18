import numpy as np

from openmdao.api import ExplicitComponent

class MeanHullMoment(ExplicitComponent):

	def setup(self):
		self.add_input('thrust_0', val=0., units='N')
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('CoG_nacelle', val=0., units='m')
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('M_spar', val=np.zeros(10), units='kg')
		self.add_input('M_ball_elem', val=np.zeros(10), units='kg')
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('L_spar', val=np.zeros(10), units='m')
		self.add_input('mean_pitch', val=0., units='rad')
		self.add_input('K_moor', val=0., units='N/m')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('moor_offset', val=0., units='m')

		self.add_output('mean_hull_moment', val=np.zeros(10), units='N*m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		thrust_0 = inputs['thrust_0']
		CoG_rotor = inputs['CoG_rotor']
		Z_spar = inputs['Z_spar']
		M_rotor = inputs['M_rotor']
		M_nacelle = inputs['M_nacelle']
		CoG_nacelle = inputs['CoG_nacelle']
		M_tower = inputs['M_tower']
		Z_tower = inputs['Z_tower']
		M_spar = inputs['M_spar']
		M_ball_elem = inputs['M_ball_elem']
		D_spar = inputs['D_spar']
		L_spar = inputs['L_spar']
		mean_pitch = inputs['mean_pitch']
		K_moor = inputs['K_moor']
		z_moor = inputs['z_moor']
		moor_offset = inputs['moor_offset']

		z_tower = np.zeros(10)
		z_spar = np.zeros(10)

		for i in xrange(10):
			z_tower[i] = (Z_tower[i] + Z_tower[i+1]) / 2.
			z_spar[i] = (Z_spar[i] + Z_spar[i+1]) / 2.

		for i in xrange(10):
			outputs['mean_hull_moment'][i] = -((CoG_rotor - Z_spar[i]) * thrust_0 + (M_rotor * (CoG_rotor - Z_spar[i]) + M_nacelle * (CoG_nacelle - Z_spar[i]) + np.sum(M_tower * (z_tower - Z_spar[i])) + np.sum((M_spar[i:] + M_ball_elem[i:]) * (z_spar[i:] - Z_spar[i])) - np.sum(1025. * np.pi / 4. * D_spar[i:-1]**2. * L_spar[i:-1] * (z_spar[i:-1] - Z_spar[i])) - (1025. * np.pi / 4. * D_spar[-1]**2. * (L_spar[-1] - 10.) * ((0. - Z_spar[-2]) / 2. - Z_spar[i]))) * 9.80665 * np.sin(mean_pitch)) #downwind (compression) side, assumed to be worst 
			if Z_spar[i] < z_moor:
				outputs['mean_hull_moment'][i] += K_moor * moor_offset * (z_moor - Z_spar[i])

	def compute_partials(self, inputs, partials):
		thrust_0 = inputs['thrust_0']
		CoG_rotor = inputs['CoG_rotor']
		Z_spar = inputs['Z_spar']
		M_rotor = inputs['M_rotor']
		M_nacelle = inputs['M_nacelle']
		CoG_nacelle = inputs['CoG_nacelle']
		M_tower = inputs['M_tower']
		Z_tower = inputs['Z_tower']
		M_spar = inputs['M_spar']
		M_ball_elem = inputs['M_ball_elem']
		D_spar = inputs['D_spar']
		L_spar = inputs['L_spar']
		mean_pitch = inputs['mean_pitch']
		K_moor = inputs['K_moor']
		z_moor = inputs['z_moor']
		moor_offset = inputs['moor_offset']

		partials['mean_hull_moment', 'thrust_0'] = np.zeros((10,1))
		partials['mean_hull_moment', 'CoG_rotor'] = np.zeros((10,1))
		partials['mean_hull_moment', 'Z_spar'] = np.zeros((10,11))
		partials['mean_hull_moment', 'M_rotor'] = np.zeros((10,1))
		partials['mean_hull_moment', 'M_nacelle'] = np.zeros((10,1))
		partials['mean_hull_moment', 'CoG_nacelle'] = np.zeros((10,1))
		partials['mean_hull_moment', 'M_tower'] = np.zeros((10,10))
		partials['mean_hull_moment', 'Z_tower'] = np.zeros((10,11))
		partials['mean_hull_moment', 'M_spar'] = np.zeros((10,10))
		partials['mean_hull_moment', 'M_ball_elem'] = np.zeros((10,10))
		partials['mean_hull_moment', 'D_spar'] = np.zeros((10,10))
		partials['mean_hull_moment', 'L_spar'] = np.zeros((10,10))
		partials['mean_hull_moment', 'mean_pitch'] = np.zeros((10,1))
		partials['mean_hull_moment', 'K_moor'] = np.zeros((10,1))
		partials['mean_hull_moment', 'z_moor'] = np.zeros((10,1))
		partials['mean_hull_moment', 'moor_offset'] = np.zeros((10,1))

		z_tower = np.zeros(10)
		z_spar = np.zeros(10)

		for i in xrange(10):
			z_tower[i] = (Z_tower[i] + Z_tower[i+1]) / 2.
			z_spar[i] = (Z_spar[i] + Z_spar[i+1]) / 2.

		for i in xrange(10):
			partials['mean_hull_moment', 'thrust_0'][i,0] += -(CoG_rotor - Z_spar[i])
			partials['mean_hull_moment', 'CoG_rotor'][i,0] += -thrust_0 - M_rotor * 9.80665 * np.sin(mean_pitch)
			partials['mean_hull_moment', 'Z_spar'][i,i] += thrust_0 + 9.80665 * np.sin(mean_pitch) * (M_rotor + M_nacelle + np.sum(M_tower) + np.sum(M_spar[i:] + M_ball_elem[i:]) - np.sum(1025. * np.pi / 4. * D_spar[i:-1]**2. * L_spar[i:-1]) - 1025. * np.pi / 4. * D_spar[-1]**2. * (L_spar[-1] - 10.)) - K_moor * moor_offset
			partials['mean_hull_moment', 'M_rotor'][i,0] += -(CoG_rotor - Z_spar[i]) * 9.80665 * np.sin(mean_pitch)
			partials['mean_hull_moment', 'M_nacelle'][i,0] += -(CoG_nacelle - Z_spar[i]) * 9.80665 * np.sin(mean_pitch)
			partials['mean_hull_moment', 'CoG_nacelle'][i,0] += -M_nacelle * 9.80665 * np.sin(mean_pitch)
			partials['mean_hull_moment', 'M_tower'][i,:] += -(z_tower - Z_spar[i]) * 9.80665 * np.sin(mean_pitch)
			partials['mean_hull_moment', 'Z_tower'][i,:-1] += -M_tower * 9.80665 * np.sin(mean_pitch) / 2.
			partials['mean_hull_moment', 'Z_tower'][i,1:] += -M_tower * 9.80665 * np.sin(mean_pitch) / 2.
			
			for j in xrange(i,10):
				if j == 9:
					partials['mean_hull_moment', 'Z_spar'][i,j] += -9.80665 * np.sin(mean_pitch) * (M_spar[j] + M_ball_elem[j] - 1025. * np.pi / 4. * D_spar[j]**2. * (L_spar[j] - 10.)) / 2.
					partials['mean_hull_moment', 'Z_spar'][i,j+1] += -9.80665 * np.sin(mean_pitch) * (M_spar[j] + M_ball_elem[j]) / 2.
					partials['mean_hull_moment', 'D_spar'][i,j] += 1025. * np.pi / 2. * D_spar[j] * (L_spar[j] - 10.) * ((0. - Z_spar[j]) / 2. - Z_spar[i]) * 9.80665 * np.sin(mean_pitch)
					partials['mean_hull_moment', 'L_spar'][i,j] += 1025. * np.pi / 4. * D_spar[j]**2. * ((0. - Z_spar[j]) / 2. - Z_spar[i]) * 9.80665 * np.sin(mean_pitch)
				else:
					partials['mean_hull_moment', 'Z_spar'][i,j] += -9.80665 * np.sin(mean_pitch) * (M_spar[j] + M_ball_elem[j] - 1025. * np.pi / 4. * D_spar[j]**2. * L_spar[j]) / 2.
					partials['mean_hull_moment', 'Z_spar'][i,j+1] += -9.80665 * np.sin(mean_pitch) * (M_spar[j] + M_ball_elem[j] - 1025. * np.pi / 4. * D_spar[j]**2. * L_spar[j]) / 2.
					partials['mean_hull_moment', 'D_spar'][i,j] += 1025. * np.pi / 2. * D_spar[j] * L_spar[j] * (z_spar[j] - Z_spar[i]) * 9.80665 * np.sin(mean_pitch)
					partials['mean_hull_moment', 'L_spar'][i,j] += 1025. * np.pi / 4. * D_spar[j]**2. * (z_spar[j] - Z_spar[i]) * 9.80665 * np.sin(mean_pitch)

				partials['mean_hull_moment', 'M_spar'][i,j] += -(z_spar[j] - Z_spar[i]) * 9.80665 * np.sin(mean_pitch)
				partials['mean_hull_moment', 'M_ball_elem'][i,j] += -(z_spar[j] - Z_spar[i]) * 9.80665 * np.sin(mean_pitch)

			partials['mean_hull_moment', 'mean_pitch'][i,0] += -(M_rotor * (CoG_rotor - Z_spar[i]) + M_nacelle * (CoG_nacelle - Z_spar[i]) + np.sum(M_tower * (z_tower - Z_spar[i])) + np.sum((M_spar[i:] + M_ball_elem[i:]) * (z_spar[i:] - Z_spar[i])) - np.sum(1025. * np.pi / 4. * D_spar[i:-1]**2. * L_spar[i:-1] * (z_spar[i:-1] - Z_spar[i])) - (1025. * np.pi / 4. * D_spar[-1]**2. * (L_spar[-1] - 10.) * ((0. - Z_spar[-2]) / 2. - Z_spar[i])) * 9.80665 * np.cos(mean_pitch))

			if Z_spar[i] < z_moor:
				partials['mean_hull_moment', 'K_moor'][i,0] += moor_offset * (z_moor - Z_spar[i])
				partials['mean_hull_moment', 'z_moor'][i,0] += K_moor * moor_offset
				partials['mean_hull_moment', 'moor_offset'][i,0] += K_moor * (z_moor - Z_spar[i])
