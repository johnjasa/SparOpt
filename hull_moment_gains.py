import numpy as np

from openmdao.api import ExplicitComponent

class HullMomentGains(ExplicitComponent):

	def setup(self):
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('I_rotor', val=0., units='kg*m**2')
		self.add_input('CoG_nacelle', val=0., units='m')
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('z_towernode', val=np.zeros(11), units='m')
		self.add_input('x_towerelem', val=np.zeros(10), units='m')
		self.add_input('x_towernode', val=np.zeros(11), units='m')
		self.add_input('x_d_towertop', val=0., units='m/m')
		self.add_input('dthrust_dv', val=0., units='N*s/m')
		self.add_input('dmoment_dv', val=0., units='N*s')
		self.add_input('dthrust_drotspeed', val=0., units='N*s/rad')
		self.add_input('dthrust_dbldpitch', val=0., units='N/rad')
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('M_spar', val=np.zeros(10), units='kg')
		self.add_input('X_sparnode', val=np.zeros(11), units='m')
		self.add_input('X_sparelem', val=np.zeros(10), units='m')
		self.add_input('Cd', val=0.)
		self.add_input('stddev_vel_X_sparelem', val=np.zeros(10), units='m/s')
		self.add_input('M_ball_elem', val=np.zeros(10), units='kg')
		self.add_input('K_moor', val=0., units='N/m')
		self.add_input('z_moor', val=0., units='m')
		
		self.add_output('hull_mom_acc_surge', val=np.zeros(10), units='kg*m')
		self.add_output('hull_mom_acc_pitch', val=np.zeros(10), units='kg*m**2/rad')
		self.add_output('hull_mom_acc_bend', val=np.zeros(10), units='kg*m')
		self.add_output('hull_mom_damp_surge', val=np.zeros(10), units='N*s')
		self.add_output('hull_mom_damp_pitch', val=np.zeros(10), units='N*m*s/rad')
		self.add_output('hull_mom_damp_bend', val=np.zeros(10), units='N*s')
		self.add_output('hull_mom_grav_pitch', val=np.zeros(10), units='N*m/rad')
		self.add_output('hull_mom_grav_bend', val=np.zeros(10), units='N')
		self.add_output('hull_mom_rotspeed', val=np.zeros(10), units='N*m*s/rad')
		self.add_output('hull_mom_bldpitch', val=np.zeros(10), units='N*m/rad')
		self.add_output('hull_mom_fairlead', val=np.zeros(10), units='N*m/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		M_tower = inputs['M_tower']
		M_nacelle = inputs['M_nacelle']
		M_rotor = inputs['M_rotor']
		I_rotor = inputs['I_rotor']
		CoG_nacelle = inputs['CoG_nacelle']
		CoG_rotor = inputs['CoG_rotor']
		z_towernode = inputs['z_towernode']
		x_towerelem = inputs['x_towerelem']
		x_towernode = inputs['x_towernode']
		x_d_towertop = inputs['x_d_towertop']
		dthrust_dv = inputs['dthrust_dv']
		dmoment_dv = inputs['dmoment_dv']
		dthrust_drotspeed = inputs['dthrust_drotspeed']
		dthrust_dbldpitch = inputs['dthrust_dbldpitch']
		D_spar = inputs['D_spar']
		Z_spar = inputs['Z_spar']
		M_spar = inputs['M_spar']
		M_ball_elem = inputs['M_ball_elem']
		Cd = inputs['Cd']
		stddev_vel_X_sparelem = inputs['stddev_vel_X_sparelem']
		K_moor = inputs['K_moor']
		z_moor = inputs['z_moor']
		X_sparnode = inputs['X_sparnode']
		X_sparelem = inputs['X_sparelem']
		
		A_spar = np.zeros(10)
		V_spar = np.zeros(10)
		drag_spar = np.zeros(10)

		N_spar = len(Z_spar) - 1
		N_tower = len(z_towernode)

		#for i in xrange(N_spar):

			#if i == (N_spar - 1): #surface-piercing element
			#	z = (Z_spar[i] + 0.) / 2
			#	dz = 0. - Z_spar[i]
			#else:
			#	z = (Z_spar[i] + Z_spar[i+1]) / 2.
			#	dz = Z_spar[i+1] - Z_spar[i]

			#A_spar[i] = 1025. * np.pi / 4. * D_spar[i]**2. * dz
			#V_spar[i] = np.pi / 4. * D_spar[i]**2. * dz
			#drag_spar[i] = 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[i] * D_spar[i] * dz

		outputs['hull_mom_acc_surge'] = np.zeros(N_spar)
		outputs['hull_mom_acc_pitch'] = np.zeros(N_spar)
		outputs['hull_mom_acc_bend'] = np.zeros(N_spar)
		outputs['hull_mom_damp_surge'] = np.zeros(N_spar)
		outputs['hull_mom_damp_pitch'] = np.zeros(N_spar)
		outputs['hull_mom_damp_bend'] = np.zeros(N_spar)
		outputs['hull_mom_grav_pitch'] = np.zeros(N_spar)
		outputs['hull_mom_grav_bend'] = np.zeros(N_spar)
		outputs['hull_mom_rotspeed'] = np.zeros(N_spar)
		outputs['hull_mom_bldpitch'] = np.zeros(N_spar)
		outputs['hull_mom_fairlead'] = np.zeros(N_spar)

		for i in xrange(N_spar):
			for j in xrange(N_tower-1):
				z = (z_towernode[j] + z_towernode[j+1]) / 2.
				outputs['hull_mom_acc_surge'][i] += M_tower[j] * (z - Z_spar[i])
				outputs['hull_mom_acc_pitch'][i] += M_tower[j] * z * (z - Z_spar[i])
				outputs['hull_mom_acc_bend'][i] += M_tower[j] * x_towerelem[j] * (z - Z_spar[i])
				outputs['hull_mom_grav_pitch'][i] += M_tower[j] * 9.80665 * (z - Z_spar[i])
				outputs['hull_mom_grav_bend'][i] += M_tower[j] * 9.80665 * (x_towerelem[j] - X_sparnode[i])
			outputs['hull_mom_acc_surge'][i] += M_nacelle * (CoG_nacelle - Z_spar[i]) + M_rotor * (CoG_rotor - Z_spar[i])
			outputs['hull_mom_acc_pitch'][i] += M_nacelle * CoG_nacelle * (CoG_nacelle - Z_spar[i]) + M_rotor * CoG_rotor * (CoG_rotor - Z_spar[i]) + I_rotor
			outputs['hull_mom_acc_bend'][i] += M_nacelle * (CoG_nacelle - Z_spar[i]) + M_rotor * (CoG_rotor - Z_spar[i]) + I_rotor * x_d_towertop
			outputs['hull_mom_damp_surge'][i] += dthrust_dv * (CoG_rotor - Z_spar[i])
			outputs['hull_mom_damp_pitch'][i] += dthrust_dv * CoG_rotor * (CoG_rotor - Z_spar[i]) + dmoment_dv
			outputs['hull_mom_damp_bend'][i] += dthrust_dv * (CoG_rotor - Z_spar[i]) + x_d_towertop * dmoment_dv
			outputs['hull_mom_grav_pitch'][i] += M_nacelle * 9.80665 * (CoG_nacelle - Z_spar[i]) + M_rotor * 9.80665 * (CoG_rotor - Z_spar[i])
			outputs['hull_mom_grav_bend'][i] += M_nacelle * 9.80665 * (1. - X_sparnode[i]) + M_rotor * 9.80665 * (1. - X_sparnode[i])
			outputs['hull_mom_rotspeed'][i] += dthrust_drotspeed * (CoG_rotor - Z_spar[i])
			outputs['hull_mom_bldpitch'][i] += dthrust_dbldpitch * (CoG_rotor - Z_spar[i])

			for j in xrange(i,N_spar):
				if j == (N_spar - 1): #surface-piercing element
					z = (Z_spar[j] + 0.) / 2.
					dz = 0. - Z_spar[j]
				else:
					z = (Z_spar[j] + Z_spar[j+1]) / 2.
					dz = Z_spar[j+1] - Z_spar[j]

				outputs['hull_mom_acc_surge'][i] += (M_spar[j] + 1025. * np.pi / 4. * D_spar[j]**2. * dz + M_ball_elem[j]) * (z - Z_spar[i])
				outputs['hull_mom_acc_pitch'][i] += (M_spar[j] + 1025. * np.pi / 4. * D_spar[j]**2. * dz + M_ball_elem[j]) * z * (z - Z_spar[i])
				outputs['hull_mom_acc_bend'][i] += (M_spar[j] + 1025. * np.pi / 4. * D_spar[j]**2. * dz + M_ball_elem[j]) * X_sparelem[j] * (z - Z_spar[i])
				outputs['hull_mom_damp_surge'][i] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * D_spar[j] * dz * (z - Z_spar[i])
				outputs['hull_mom_damp_pitch'][i] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * D_spar[j] * dz * z * (z - Z_spar[i])
				outputs['hull_mom_damp_bend'][i] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * D_spar[j] * dz * X_sparelem[j] * (z - Z_spar[i])
				outputs['hull_mom_grav_pitch'][i] += (M_spar[j] + M_ball_elem[j] - 1025. * np.pi / 4. * D_spar[j]**2. * dz) * 9.80665 * (z - Z_spar[i])
				outputs['hull_mom_grav_bend'][i] += (M_spar[j] + M_ball_elem[j] - 1025. * np.pi / 4. * D_spar[j]**2. * dz) * 9.80665 * (X_sparelem[j] - X_sparnode[i])
			if Z_spar[i] < z_moor:
				outputs['hull_mom_fairlead'][i] += -K_moor * (z_moor - Z_spar[i])

	def compute_partials(self, inputs, partials):
		M_tower = inputs['M_tower']
		M_nacelle = inputs['M_nacelle']
		M_rotor = inputs['M_rotor']
		I_rotor = inputs['I_rotor']
		CoG_nacelle = inputs['CoG_nacelle']
		CoG_rotor = inputs['CoG_rotor']
		z_towernode = inputs['z_towernode']
		x_towerelem = inputs['x_towerelem']
		x_towernode = inputs['x_towernode']
		x_d_towertop = inputs['x_d_towertop']
		dthrust_dv = inputs['dthrust_dv']
		dmoment_dv = inputs['dmoment_dv']
		dthrust_drotspeed = inputs['dthrust_drotspeed']
		dthrust_dbldpitch = inputs['dthrust_dbldpitch']
		D_spar = inputs['D_spar']
		Z_spar = inputs['Z_spar']
		M_spar = inputs['M_spar']
		M_ball_elem = inputs['M_ball_elem']
		Cd = inputs['Cd']
		stddev_vel_X_sparelem = inputs['stddev_vel_X_sparelem']
		K_moor = inputs['K_moor']
		z_moor = inputs['z_moor']
		X_sparnode = inputs['X_sparnode']
		X_sparelem = inputs['X_sparelem']
		
		A_spar = np.zeros(10)
		V_spar = np.zeros(10)
		drag_spar = np.zeros(10)

		N_spar = len(Z_spar) - 1
		N_tower = len(z_towernode)

		for i in xrange(N_spar):

			if i == (N_spar - 1): #surface-piercing element
				z = (Z_spar[i] + 0.) / 2.
				dz = 0. - Z_spar[i]
			else:
				z = (Z_spar[i] + Z_spar[i+1]) / 2.
				dz = Z_spar[i+1] - Z_spar[i]

			A_spar[i] = 1025. * np.pi / 4. * D_spar[i]**2. * dz
			V_spar[i] = np.pi / 4. * D_spar[i]**2. * dz
			drag_spar[i] = 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[i] * D_spar[i] * dz

		partials['hull_mom_acc_surge', 'M_tower'] = np.zeros((10,10))
		partials['hull_mom_acc_pitch', 'M_tower'] = np.zeros((10,10))
		partials['hull_mom_acc_bend', 'M_tower'] = np.zeros((10,10))
		partials['hull_mom_damp_surge', 'M_tower'] = np.zeros((10,10))
		partials['hull_mom_damp_pitch', 'M_tower'] = np.zeros((10,10))
		partials['hull_mom_damp_bend', 'M_tower'] = np.zeros((10,10))
		partials['hull_mom_grav_pitch', 'M_tower'] = np.zeros((10,10))
		partials['hull_mom_grav_bend', 'M_tower'] = np.zeros((10,10))
		partials['hull_mom_rotspeed', 'M_tower'] = np.zeros((10,10))
		partials['hull_mom_bldpitch', 'M_tower'] = np.zeros((10,10))
		partials['hull_mom_fairlead', 'M_tower'] = np.zeros((10,10))

		partials['hull_mom_acc_surge', 'M_nacelle'] = np.zeros((10,1))
		partials['hull_mom_acc_pitch', 'M_nacelle'] = np.zeros((10,1))
		partials['hull_mom_acc_bend', 'M_nacelle'] = np.zeros((10,1))
		partials['hull_mom_damp_surge', 'M_nacelle'] = np.zeros((10,1))
		partials['hull_mom_damp_pitch', 'M_nacelle'] = np.zeros((10,1))
		partials['hull_mom_damp_bend', 'M_nacelle'] = np.zeros((10,1))
		partials['hull_mom_grav_pitch', 'M_nacelle'] = np.zeros((10,1))
		partials['hull_mom_grav_bend', 'M_nacelle'] = np.zeros((10,1))
		partials['hull_mom_rotspeed', 'M_nacelle'] = np.zeros((10,1))
		partials['hull_mom_bldpitch', 'M_nacelle'] = np.zeros((10,1))
		partials['hull_mom_fairlead', 'M_nacelle'] = np.zeros((10,1))

		partials['hull_mom_acc_surge', 'M_rotor'] = np.zeros((10,1))
		partials['hull_mom_acc_pitch', 'M_rotor'] = np.zeros((10,1))
		partials['hull_mom_acc_bend', 'M_rotor'] = np.zeros((10,1))
		partials['hull_mom_damp_surge', 'M_rotor'] = np.zeros((10,1))
		partials['hull_mom_damp_pitch', 'M_rotor'] = np.zeros((10,1))
		partials['hull_mom_damp_bend', 'M_rotor'] = np.zeros((10,1))
		partials['hull_mom_grav_pitch', 'M_rotor'] = np.zeros((10,1))
		partials['hull_mom_grav_bend', 'M_rotor'] = np.zeros((10,1))
		partials['hull_mom_rotspeed', 'M_rotor'] = np.zeros((10,1))
		partials['hull_mom_bldpitch', 'M_rotor'] = np.zeros((10,1))
		partials['hull_mom_fairlead', 'M_rotor'] = np.zeros((10,1))

		partials['hull_mom_acc_surge', 'I_rotor'] = np.zeros((10,1))
		partials['hull_mom_acc_pitch', 'I_rotor'] = np.zeros((10,1))
		partials['hull_mom_acc_bend', 'I_rotor'] = np.zeros((10,1))
		partials['hull_mom_damp_surge', 'I_rotor'] = np.zeros((10,1))
		partials['hull_mom_damp_pitch', 'I_rotor'] = np.zeros((10,1))
		partials['hull_mom_damp_bend', 'I_rotor'] = np.zeros((10,1))
		partials['hull_mom_grav_pitch', 'I_rotor'] = np.zeros((10,1))
		partials['hull_mom_grav_bend', 'I_rotor'] = np.zeros((10,1))
		partials['hull_mom_rotspeed', 'I_rotor'] = np.zeros((10,1))
		partials['hull_mom_bldpitch', 'I_rotor'] = np.zeros((10,1))
		partials['hull_mom_fairlead', 'I_rotor'] = np.zeros((10,1))

		partials['hull_mom_acc_surge', 'CoG_nacelle'] = np.zeros((10,1))
		partials['hull_mom_acc_pitch', 'CoG_nacelle'] = np.zeros((10,1))
		partials['hull_mom_acc_bend', 'CoG_nacelle'] = np.zeros((10,1))
		partials['hull_mom_damp_surge', 'CoG_nacelle'] = np.zeros((10,1))
		partials['hull_mom_damp_pitch', 'CoG_nacelle'] = np.zeros((10,1))
		partials['hull_mom_damp_bend', 'CoG_nacelle'] = np.zeros((10,1))
		partials['hull_mom_grav_pitch', 'CoG_nacelle'] = np.zeros((10,1))
		partials['hull_mom_grav_bend', 'CoG_nacelle'] = np.zeros((10,1))
		partials['hull_mom_rotspeed', 'CoG_nacelle'] = np.zeros((10,1))
		partials['hull_mom_bldpitch', 'CoG_nacelle'] = np.zeros((10,1))
		partials['hull_mom_fairlead', 'CoG_nacelle'] = np.zeros((10,1))

		partials['hull_mom_acc_surge', 'CoG_rotor'] = np.zeros((10,1))
		partials['hull_mom_acc_pitch', 'CoG_rotor'] = np.zeros((10,1))
		partials['hull_mom_acc_bend', 'CoG_rotor'] = np.zeros((10,1))
		partials['hull_mom_damp_surge', 'CoG_rotor'] = np.zeros((10,1))
		partials['hull_mom_damp_pitch', 'CoG_rotor'] = np.zeros((10,1))
		partials['hull_mom_damp_bend', 'CoG_rotor'] = np.zeros((10,1))
		partials['hull_mom_grav_pitch', 'CoG_rotor'] = np.zeros((10,1))
		partials['hull_mom_grav_bend', 'CoG_rotor'] = np.zeros((10,1))
		partials['hull_mom_rotspeed', 'CoG_rotor'] = np.zeros((10,1))
		partials['hull_mom_bldpitch', 'CoG_rotor'] = np.zeros((10,1))
		partials['hull_mom_fairlead', 'CoG_rotor'] = np.zeros((10,1))

		partials['hull_mom_acc_surge', 'z_towernode'] = np.zeros((10,11))
		partials['hull_mom_acc_pitch', 'z_towernode'] = np.zeros((10,11))
		partials['hull_mom_acc_bend', 'z_towernode'] = np.zeros((10,11))
		partials['hull_mom_damp_surge', 'z_towernode'] = np.zeros((10,11))
		partials['hull_mom_damp_pitch', 'z_towernode'] = np.zeros((10,11))
		partials['hull_mom_damp_bend', 'z_towernode'] = np.zeros((10,11))
		partials['hull_mom_grav_pitch', 'z_towernode'] = np.zeros((10,11))
		partials['hull_mom_grav_bend', 'z_towernode'] = np.zeros((10,11))
		partials['hull_mom_rotspeed', 'z_towernode'] = np.zeros((10,11))
		partials['hull_mom_bldpitch', 'z_towernode'] = np.zeros((10,11))
		partials['hull_mom_fairlead', 'z_towernode'] = np.zeros((10,11))

		partials['hull_mom_acc_surge', 'x_towerelem'] = np.zeros((10,10))
		partials['hull_mom_acc_pitch', 'x_towerelem'] = np.zeros((10,10))
		partials['hull_mom_acc_bend', 'x_towerelem'] = np.zeros((10,10))
		partials['hull_mom_damp_surge', 'x_towerelem'] = np.zeros((10,10))
		partials['hull_mom_damp_pitch', 'x_towerelem'] = np.zeros((10,10))
		partials['hull_mom_damp_bend', 'x_towerelem'] = np.zeros((10,10))
		partials['hull_mom_grav_pitch', 'x_towerelem'] = np.zeros((10,10))
		partials['hull_mom_grav_bend', 'x_towerelem'] = np.zeros((10,10))
		partials['hull_mom_rotspeed', 'x_towerelem'] = np.zeros((10,10))
		partials['hull_mom_bldpitch', 'x_towerelem'] = np.zeros((10,10))
		partials['hull_mom_fairlead', 'x_towerelem'] = np.zeros((10,10))

		partials['hull_mom_acc_surge', 'x_towernode'] = np.zeros((10,11))
		partials['hull_mom_acc_pitch', 'x_towernode'] = np.zeros((10,11))
		partials['hull_mom_acc_bend', 'x_towernode'] = np.zeros((10,11))
		partials['hull_mom_damp_surge', 'x_towernode'] = np.zeros((10,11))
		partials['hull_mom_damp_pitch', 'x_towernode'] = np.zeros((10,11))
		partials['hull_mom_damp_bend', 'x_towernode'] = np.zeros((10,11))
		partials['hull_mom_grav_pitch', 'x_towernode'] = np.zeros((10,11))
		partials['hull_mom_grav_bend', 'x_towernode'] = np.zeros((10,11))
		partials['hull_mom_rotspeed', 'x_towernode'] = np.zeros((10,11))
		partials['hull_mom_bldpitch', 'x_towernode'] = np.zeros((10,11))
		partials['hull_mom_fairlead', 'x_towernode'] = np.zeros((10,11))

		partials['hull_mom_acc_surge', 'x_d_towertop'] = np.zeros((10,1))
		partials['hull_mom_acc_pitch', 'x_d_towertop'] = np.zeros((10,1))
		partials['hull_mom_acc_bend', 'x_d_towertop'] = np.zeros((10,1))
		partials['hull_mom_damp_surge', 'x_d_towertop'] = np.zeros((10,1))
		partials['hull_mom_damp_pitch', 'x_d_towertop'] = np.zeros((10,1))
		partials['hull_mom_damp_bend', 'x_d_towertop'] = np.zeros((10,1))
		partials['hull_mom_grav_pitch', 'x_d_towertop'] = np.zeros((10,1))
		partials['hull_mom_grav_bend', 'x_d_towertop'] = np.zeros((10,1))
		partials['hull_mom_rotspeed', 'x_d_towertop'] = np.zeros((10,1))
		partials['hull_mom_bldpitch', 'x_d_towertop'] = np.zeros((10,1))
		partials['hull_mom_fairlead', 'x_d_towertop'] = np.zeros((10,1))

		partials['hull_mom_acc_surge', 'dthrust_dv'] = np.zeros((10,1))
		partials['hull_mom_acc_pitch', 'dthrust_dv'] = np.zeros((10,1))
		partials['hull_mom_acc_bend', 'dthrust_dv'] = np.zeros((10,1))
		partials['hull_mom_damp_surge', 'dthrust_dv'] = np.zeros((10,1))
		partials['hull_mom_damp_pitch', 'dthrust_dv'] = np.zeros((10,1))
		partials['hull_mom_damp_bend', 'dthrust_dv'] = np.zeros((10,1))
		partials['hull_mom_grav_pitch', 'dthrust_dv'] = np.zeros((10,1))
		partials['hull_mom_grav_bend', 'dthrust_dv'] = np.zeros((10,1))
		partials['hull_mom_rotspeed', 'dthrust_dv'] = np.zeros((10,1))
		partials['hull_mom_bldpitch', 'dthrust_dv'] = np.zeros((10,1))
		partials['hull_mom_fairlead', 'dthrust_dv'] = np.zeros((10,1))

		partials['hull_mom_acc_surge', 'dmoment_dv'] = np.zeros((10,1))
		partials['hull_mom_acc_pitch', 'dmoment_dv'] = np.zeros((10,1))
		partials['hull_mom_acc_bend', 'dmoment_dv'] = np.zeros((10,1))
		partials['hull_mom_damp_surge', 'dmoment_dv'] = np.zeros((10,1))
		partials['hull_mom_damp_pitch', 'dmoment_dv'] = np.zeros((10,1))
		partials['hull_mom_damp_bend', 'dmoment_dv'] = np.zeros((10,1))
		partials['hull_mom_grav_pitch', 'dmoment_dv'] = np.zeros((10,1))
		partials['hull_mom_grav_bend', 'dmoment_dv'] = np.zeros((10,1))
		partials['hull_mom_rotspeed', 'dmoment_dv'] = np.zeros((10,1))
		partials['hull_mom_bldpitch', 'dmoment_dv'] = np.zeros((10,1))
		partials['hull_mom_fairlead', 'dmoment_dv'] = np.zeros((10,1))

		partials['hull_mom_acc_surge', 'dthrust_drotspeed'] = np.zeros((10,1))
		partials['hull_mom_acc_pitch', 'dthrust_drotspeed'] = np.zeros((10,1))
		partials['hull_mom_acc_bend', 'dthrust_drotspeed'] = np.zeros((10,1))
		partials['hull_mom_damp_surge', 'dthrust_drotspeed'] = np.zeros((10,1))
		partials['hull_mom_damp_pitch', 'dthrust_drotspeed'] = np.zeros((10,1))
		partials['hull_mom_damp_bend', 'dthrust_drotspeed'] = np.zeros((10,1))
		partials['hull_mom_grav_pitch', 'dthrust_drotspeed'] = np.zeros((10,1))
		partials['hull_mom_grav_bend', 'dthrust_drotspeed'] = np.zeros((10,1))
		partials['hull_mom_rotspeed', 'dthrust_drotspeed'] = np.zeros((10,1))
		partials['hull_mom_bldpitch', 'dthrust_drotspeed'] = np.zeros((10,1))
		partials['hull_mom_fairlead', 'dthrust_drotspeed'] = np.zeros((10,1))

		partials['hull_mom_acc_surge', 'dthrust_dbldpitch'] = np.zeros((10,1))
		partials['hull_mom_acc_pitch', 'dthrust_dbldpitch'] = np.zeros((10,1))
		partials['hull_mom_acc_bend', 'dthrust_dbldpitch'] = np.zeros((10,1))
		partials['hull_mom_damp_surge', 'dthrust_dbldpitch'] = np.zeros((10,1))
		partials['hull_mom_damp_pitch', 'dthrust_dbldpitch'] = np.zeros((10,1))
		partials['hull_mom_damp_bend', 'dthrust_dbldpitch'] = np.zeros((10,1))
		partials['hull_mom_grav_pitch', 'dthrust_dbldpitch'] = np.zeros((10,1))
		partials['hull_mom_grav_bend', 'dthrust_dbldpitch'] = np.zeros((10,1))
		partials['hull_mom_rotspeed', 'dthrust_dbldpitch'] = np.zeros((10,1))
		partials['hull_mom_bldpitch', 'dthrust_dbldpitch'] = np.zeros((10,1))
		partials['hull_mom_fairlead', 'dthrust_dbldpitch'] = np.zeros((10,1))

		partials['hull_mom_acc_surge', 'D_spar'] = np.zeros((10,10))
		partials['hull_mom_acc_pitch', 'D_spar'] = np.zeros((10,10))
		partials['hull_mom_acc_bend', 'D_spar'] = np.zeros((10,10))
		partials['hull_mom_damp_surge', 'D_spar'] = np.zeros((10,10))
		partials['hull_mom_damp_pitch', 'D_spar'] = np.zeros((10,10))
		partials['hull_mom_damp_bend', 'D_spar'] = np.zeros((10,10))
		partials['hull_mom_grav_pitch', 'D_spar'] = np.zeros((10,10))
		partials['hull_mom_grav_bend', 'D_spar'] = np.zeros((10,10))
		partials['hull_mom_rotspeed', 'D_spar'] = np.zeros((10,10))
		partials['hull_mom_bldpitch', 'D_spar'] = np.zeros((10,10))
		partials['hull_mom_fairlead', 'D_spar'] = np.zeros((10,10))

		partials['hull_mom_acc_surge', 'Z_spar'] = np.zeros((10,11))
		partials['hull_mom_acc_pitch', 'Z_spar'] = np.zeros((10,11))
		partials['hull_mom_acc_bend', 'Z_spar'] = np.zeros((10,11))
		partials['hull_mom_damp_surge', 'Z_spar'] = np.zeros((10,11))
		partials['hull_mom_damp_pitch', 'Z_spar'] = np.zeros((10,11))
		partials['hull_mom_damp_bend', 'Z_spar'] = np.zeros((10,11))
		partials['hull_mom_grav_pitch', 'Z_spar'] = np.zeros((10,11))
		partials['hull_mom_grav_bend', 'Z_spar'] = np.zeros((10,11))
		partials['hull_mom_rotspeed', 'Z_spar'] = np.zeros((10,11))
		partials['hull_mom_bldpitch', 'Z_spar'] = np.zeros((10,11))
		partials['hull_mom_fairlead', 'Z_spar'] = np.zeros((10,11))

		partials['hull_mom_acc_surge', 'M_spar'] = np.zeros((10,10))
		partials['hull_mom_acc_pitch', 'M_spar'] = np.zeros((10,10))
		partials['hull_mom_acc_bend', 'M_spar'] = np.zeros((10,10))
		partials['hull_mom_damp_surge', 'M_spar'] = np.zeros((10,10))
		partials['hull_mom_damp_pitch', 'M_spar'] = np.zeros((10,10))
		partials['hull_mom_damp_bend', 'M_spar'] = np.zeros((10,10))
		partials['hull_mom_grav_pitch', 'M_spar'] = np.zeros((10,10))
		partials['hull_mom_grav_bend', 'M_spar'] = np.zeros((10,10))
		partials['hull_mom_rotspeed', 'M_spar'] = np.zeros((10,10))
		partials['hull_mom_bldpitch', 'M_spar'] = np.zeros((10,10))
		partials['hull_mom_fairlead', 'M_spar'] = np.zeros((10,10))

		partials['hull_mom_acc_surge', 'X_sparnode'] = np.zeros((10,11))
		partials['hull_mom_acc_pitch', 'X_sparnode'] = np.zeros((10,11))
		partials['hull_mom_acc_bend', 'X_sparnode'] = np.zeros((10,11))
		partials['hull_mom_damp_surge', 'X_sparnode'] = np.zeros((10,11))
		partials['hull_mom_damp_pitch', 'X_sparnode'] = np.zeros((10,11))
		partials['hull_mom_damp_bend', 'X_sparnode'] = np.zeros((10,11))
		partials['hull_mom_grav_pitch', 'X_sparnode'] = np.zeros((10,11))
		partials['hull_mom_grav_bend', 'X_sparnode'] = np.zeros((10,11))
		partials['hull_mom_rotspeed', 'X_sparnode'] = np.zeros((10,11))
		partials['hull_mom_bldpitch', 'X_sparnode'] = np.zeros((10,11))
		partials['hull_mom_fairlead', 'X_sparnode'] = np.zeros((10,11))

		partials['hull_mom_acc_surge', 'X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_acc_pitch', 'X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_acc_bend', 'X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_damp_surge', 'X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_damp_pitch', 'X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_damp_bend', 'X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_grav_pitch', 'X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_grav_bend', 'X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_rotspeed', 'X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_bldpitch', 'X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_fairlead', 'X_sparelem'] = np.zeros((10,10))

		partials['hull_mom_acc_surge', 'Cd'] = np.zeros((10,1))
		partials['hull_mom_acc_pitch', 'Cd'] = np.zeros((10,1))
		partials['hull_mom_acc_bend', 'Cd'] = np.zeros((10,1))
		partials['hull_mom_damp_surge', 'Cd'] = np.zeros((10,1))
		partials['hull_mom_damp_pitch', 'Cd'] = np.zeros((10,1))
		partials['hull_mom_damp_bend', 'Cd'] = np.zeros((10,1))
		partials['hull_mom_grav_pitch', 'Cd'] = np.zeros((10,1))
		partials['hull_mom_grav_bend', 'Cd'] = np.zeros((10,1))
		partials['hull_mom_rotspeed', 'Cd'] = np.zeros((10,1))
		partials['hull_mom_bldpitch', 'Cd'] = np.zeros((10,1))
		partials['hull_mom_fairlead', 'Cd'] = np.zeros((10,1))

		partials['hull_mom_acc_surge', 'stddev_vel_X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_acc_pitch', 'stddev_vel_X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_acc_bend', 'stddev_vel_X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_damp_surge', 'stddev_vel_X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_damp_pitch', 'stddev_vel_X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_damp_bend', 'stddev_vel_X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_grav_pitch', 'stddev_vel_X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_grav_bend', 'stddev_vel_X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_rotspeed', 'stddev_vel_X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_bldpitch', 'stddev_vel_X_sparelem'] = np.zeros((10,10))
		partials['hull_mom_fairlead', 'stddev_vel_X_sparelem'] = np.zeros((10,10))

		partials['hull_mom_acc_surge', 'M_ball_elem'] = np.zeros((10,10))
		partials['hull_mom_acc_pitch', 'M_ball_elem'] = np.zeros((10,10))
		partials['hull_mom_acc_bend', 'M_ball_elem'] = np.zeros((10,10))
		partials['hull_mom_damp_surge', 'M_ball_elem'] = np.zeros((10,10))
		partials['hull_mom_damp_pitch', 'M_ball_elem'] = np.zeros((10,10))
		partials['hull_mom_damp_bend', 'M_ball_elem'] = np.zeros((10,10))
		partials['hull_mom_grav_pitch', 'M_ball_elem'] = np.zeros((10,10))
		partials['hull_mom_grav_bend', 'M_ball_elem'] = np.zeros((10,10))
		partials['hull_mom_rotspeed', 'M_ball_elem'] = np.zeros((10,10))
		partials['hull_mom_bldpitch', 'M_ball_elem'] = np.zeros((10,10))
		partials['hull_mom_fairlead', 'M_ball_elem'] = np.zeros((10,10))

		partials['hull_mom_acc_surge', 'K_moor'] = np.zeros((10,1))
		partials['hull_mom_acc_pitch', 'K_moor'] = np.zeros((10,1))
		partials['hull_mom_acc_bend', 'K_moor'] = np.zeros((10,1))
		partials['hull_mom_damp_surge', 'K_moor'] = np.zeros((10,1))
		partials['hull_mom_damp_pitch', 'K_moor'] = np.zeros((10,1))
		partials['hull_mom_damp_bend', 'K_moor'] = np.zeros((10,1))
		partials['hull_mom_grav_pitch', 'K_moor'] = np.zeros((10,1))
		partials['hull_mom_grav_bend', 'K_moor'] = np.zeros((10,1))
		partials['hull_mom_rotspeed', 'K_moor'] = np.zeros((10,1))
		partials['hull_mom_bldpitch', 'K_moor'] = np.zeros((10,1))
		partials['hull_mom_fairlead', 'K_moor'] = np.zeros((10,1))

		partials['hull_mom_acc_surge', 'z_moor'] = np.zeros((10,1))
		partials['hull_mom_acc_pitch', 'z_moor'] = np.zeros((10,1))
		partials['hull_mom_acc_bend', 'z_moor'] = np.zeros((10,1))
		partials['hull_mom_damp_surge', 'z_moor'] = np.zeros((10,1))
		partials['hull_mom_damp_pitch', 'z_moor'] = np.zeros((10,1))
		partials['hull_mom_damp_bend', 'z_moor'] = np.zeros((10,1))
		partials['hull_mom_grav_pitch', 'z_moor'] = np.zeros((10,1))
		partials['hull_mom_grav_bend', 'z_moor'] = np.zeros((10,1))
		partials['hull_mom_rotspeed', 'z_moor'] = np.zeros((10,1))
		partials['hull_mom_bldpitch', 'z_moor'] = np.zeros((10,1))
		partials['hull_mom_fairlead', 'z_moor'] = np.zeros((10,1))

		for i in xrange(N_spar):
			for j in xrange(N_tower-1):
				z = (z_towernode[j] + z_towernode[j+1]) / 2.
				partials['hull_mom_acc_surge', 'M_tower'][i,j] += (z - Z_spar[i])
				partials['hull_mom_acc_surge', 'z_towernode'][i,j] += M_tower[j] / 2.
				partials['hull_mom_acc_surge', 'z_towernode'][i,j+1] += M_tower[j] / 2.
				partials['hull_mom_acc_surge', 'Z_spar'][i,i] += -M_tower[j]
				
				partials['hull_mom_acc_pitch', 'M_tower'][i,j] += z * (z - Z_spar[i])
				partials['hull_mom_acc_pitch', 'z_towernode'][i,j] += M_tower[j] * z / 2. + M_tower[j] * (z - Z_spar[i]) / 2.
				partials['hull_mom_acc_pitch', 'z_towernode'][i,j+1] += M_tower[j] * z / 2. + M_tower[j] * (z - Z_spar[i]) / 2.
				partials['hull_mom_acc_pitch', 'Z_spar'][i,i] += -M_tower[j] * z
				
				partials['hull_mom_acc_bend', 'M_tower'][i,j] += x_towerelem[j] * (z - Z_spar[i])
				partials['hull_mom_acc_bend', 'z_towernode'][i,j] += M_tower[j] * x_towerelem[j] / 2.
				partials['hull_mom_acc_bend', 'z_towernode'][i,j+1] += M_tower[j] * x_towerelem[j] / 2.
				partials['hull_mom_acc_bend', 'Z_spar'][i,i] += -M_tower[j] * x_towerelem[j]
				partials['hull_mom_acc_bend', 'x_towerelem'][i,j] += M_tower[j] * (z - Z_spar[i])
				
				partials['hull_mom_grav_pitch', 'M_tower'][i,j] += 9.80665 * (z - Z_spar[i])
				partials['hull_mom_grav_pitch', 'z_towernode'][i,j] += M_tower[j] * 9.80665 / 2.
				partials['hull_mom_grav_pitch', 'z_towernode'][i,j+1] += M_tower[j] * 9.80665 / 2.
				partials['hull_mom_grav_pitch', 'Z_spar'][i,i] += -M_tower[j] * 9.80665
				
				partials['hull_mom_grav_bend', 'M_tower'][i,j] += 9.80665 * (x_towerelem[j] - X_sparnode[i])
				partials['hull_mom_grav_bend', 'X_sparnode'][i,i] += -M_tower[j] * 9.80665
				partials['hull_mom_grav_bend', 'x_towerelem'][i,j] += M_tower[j] * 9.80665

			partials['hull_mom_acc_surge', 'M_nacelle'][i,0] += (CoG_nacelle - Z_spar[i])
			partials['hull_mom_acc_surge', 'M_rotor'][i,0] += (CoG_rotor - Z_spar[i])
			partials['hull_mom_acc_surge', 'CoG_nacelle'][i,0] += M_nacelle
			partials['hull_mom_acc_surge', 'CoG_rotor'][i,0] += M_rotor
			partials['hull_mom_acc_surge', 'Z_spar'][i,i] += -M_nacelle - M_rotor

			partials['hull_mom_acc_pitch', 'M_nacelle'][i,0] += CoG_nacelle * (CoG_nacelle - Z_spar[i])
			partials['hull_mom_acc_pitch', 'M_rotor'][i,0] += CoG_rotor * (CoG_rotor - Z_spar[i])
			partials['hull_mom_acc_pitch', 'CoG_nacelle'][i,0] += M_nacelle * (CoG_nacelle - Z_spar[i]) + M_nacelle * CoG_nacelle
			partials['hull_mom_acc_pitch', 'CoG_rotor'][i,0] += M_rotor * (CoG_rotor - Z_spar[i]) + M_rotor * CoG_rotor
			partials['hull_mom_acc_pitch', 'I_rotor'][i,0] += 1.
			partials['hull_mom_acc_pitch', 'Z_spar'][i,i] += -M_nacelle * CoG_nacelle - M_rotor * CoG_rotor

			partials['hull_mom_acc_bend', 'M_nacelle'][i,0] += (CoG_nacelle - Z_spar[i])
			partials['hull_mom_acc_bend', 'M_rotor'][i,0] += (CoG_rotor - Z_spar[i])
			partials['hull_mom_acc_bend', 'CoG_nacelle'][i,0] += M_nacelle
			partials['hull_mom_acc_bend', 'CoG_rotor'][i,0] += M_rotor
			partials['hull_mom_acc_bend', 'Z_spar'][i,i] += -M_nacelle - M_rotor
			partials['hull_mom_acc_bend', 'I_rotor'][i,0] += x_d_towertop
			partials['hull_mom_acc_bend', 'x_d_towertop'][i,0] += I_rotor

			partials['hull_mom_damp_surge', 'dthrust_dv'][i,0] += (CoG_rotor - Z_spar[i])
			partials['hull_mom_damp_surge', 'CoG_rotor'][i,0] += dthrust_dv
			partials['hull_mom_damp_surge', 'Z_spar'][i,i] += -dthrust_dv

			partials['hull_mom_damp_pitch', 'dthrust_dv'][i,0] += CoG_rotor * (CoG_rotor - Z_spar[i])
			partials['hull_mom_damp_pitch', 'CoG_rotor'][i,0] += dthrust_dv * (CoG_rotor - Z_spar[i]) + dthrust_dv * CoG_rotor
			partials['hull_mom_damp_pitch', 'Z_spar'][i,i] += -dthrust_dv * CoG_rotor
			partials['hull_mom_damp_pitch', 'dmoment_dv'][i,0] += 1.

			partials['hull_mom_damp_bend', 'dthrust_dv'][i,0] += (CoG_rotor - Z_spar[i])
			partials['hull_mom_damp_bend', 'CoG_rotor'][i,0] += dthrust_dv
			partials['hull_mom_damp_bend', 'Z_spar'][i,i] += -dthrust_dv
			partials['hull_mom_damp_bend', 'dmoment_dv'][i,0] += x_d_towertop
			partials['hull_mom_damp_bend', 'x_d_towertop'][i,0] += dmoment_dv

			partials['hull_mom_grav_pitch', 'M_nacelle'][i,0] += 9.80665 * (CoG_nacelle - Z_spar[i])
			partials['hull_mom_grav_pitch', 'M_rotor'][i,0] += 9.80665 * (CoG_rotor - Z_spar[i])
			partials['hull_mom_grav_pitch', 'CoG_nacelle'][i,0] += M_nacelle * 9.80665
			partials['hull_mom_grav_pitch', 'CoG_rotor'][i,0] += M_rotor * 9.80665
			partials['hull_mom_grav_pitch', 'Z_spar'][i,i] += -M_nacelle * 9.80665 - M_rotor * 9.80665

			partials['hull_mom_grav_bend', 'M_nacelle'][i,0] += 9.80665 * (1. - X_sparnode[i])
			partials['hull_mom_grav_bend', 'M_rotor'][i,0] += 9.80665 * (1. - X_sparnode[i])
			partials['hull_mom_grav_bend', 'X_sparnode'][i,i] += -M_nacelle * 9.80665 - M_rotor * 9.80665

			partials['hull_mom_rotspeed', 'dthrust_drotspeed'][i,0] += (CoG_rotor - Z_spar[i])
			partials['hull_mom_rotspeed', 'CoG_rotor'][i,0] += dthrust_drotspeed
			partials['hull_mom_rotspeed', 'Z_spar'][i,i] += -dthrust_drotspeed

			partials['hull_mom_bldpitch', 'dthrust_dbldpitch'][i,0] += (CoG_rotor - Z_spar[i])
			partials['hull_mom_bldpitch', 'CoG_rotor'][i,0] += dthrust_dbldpitch
			partials['hull_mom_bldpitch', 'Z_spar'][i,i] += -dthrust_dbldpitch

			for j in xrange(i,N_spar):
				
				if j == (N_spar - 1): #surface-piercing element
					z = (Z_spar[j] + 0.) / 2.
					dz = 0. - Z_spar[j]
				else:
					z = (Z_spar[j] + Z_spar[j+1]) / 2.
					dz = Z_spar[j+1] - Z_spar[j]

				partials['hull_mom_acc_surge', 'M_spar'][i,j] +=  (z - Z_spar[i])
				partials['hull_mom_acc_surge', 'M_ball_elem'][i,j] += (z - Z_spar[i])
				partials['hull_mom_acc_surge', 'Z_spar'][i,i] += -(M_spar[j] + M_ball_elem[j] + A_spar[j])
				partials['hull_mom_acc_surge', 'Z_spar'][i,j] += (M_spar[j] + M_ball_elem[j] + A_spar[j]) / 2. - 1025. * np.pi / 4. * D_spar[j]**2. * (z - Z_spar[i])
				if j != (N_spar - 1):
					partials['hull_mom_acc_surge', 'Z_spar'][i,j+1] += (M_spar[j] + M_ball_elem[j] + A_spar[j]) / 2. + 1025. * np.pi / 4. * D_spar[j]**2. * (z - Z_spar[i])
				partials['hull_mom_acc_surge', 'D_spar'][i,j] += 1025. * np.pi / 2. * D_spar[j] * dz * (z - Z_spar[i])
				
				partials['hull_mom_acc_pitch', 'M_spar'][i,j] +=  z * (z - Z_spar[i])
				partials['hull_mom_acc_pitch', 'M_ball_elem'][i,j] += z * (z - Z_spar[i])
				partials['hull_mom_acc_pitch', 'Z_spar'][i,i] += -(M_spar[j] + M_ball_elem[j] + A_spar[j]) * z
				partials['hull_mom_acc_pitch', 'Z_spar'][i,j] += (M_spar[j] + M_ball_elem[j] + A_spar[j]) * z / 2. + (M_spar[j] + M_ball_elem[j] + A_spar[j]) * (z - Z_spar[i]) / 2. - 1025. * np.pi / 4. * D_spar[j]**2. * z * (z - Z_spar[i])
				if j != (N_spar - 1):
					partials['hull_mom_acc_pitch', 'Z_spar'][i,j+1] += (M_spar[j] + M_ball_elem[j] + A_spar[j]) * z / 2. + (M_spar[j] + M_ball_elem[j] + A_spar[j]) * (z - Z_spar[i]) / 2. + 1025. * np.pi / 4. * D_spar[j]**2. * z * (z - Z_spar[i])
				partials['hull_mom_acc_pitch', 'D_spar'][i,j] += 1025. * np.pi / 2. * D_spar[j] * dz * z * (z - Z_spar[i])
				
				partials['hull_mom_acc_bend', 'M_spar'][i,j] += X_sparelem[j] * (z - Z_spar[i])
				partials['hull_mom_acc_bend', 'M_ball_elem'][i,j] += X_sparelem[j] * (z - Z_spar[i])
				partials['hull_mom_acc_bend', 'Z_spar'][i,i] += -(M_spar[j] + M_ball_elem[j] + A_spar[j]) * X_sparelem[j]
				partials['hull_mom_acc_bend', 'Z_spar'][i,j] += (M_spar[j] + M_ball_elem[j] + A_spar[j]) * X_sparelem[j] / 2. - 1025. * np.pi / 4. * D_spar[j]**2. * X_sparelem[j] * (z - Z_spar[i])
				if j != (N_spar - 1):
					partials['hull_mom_acc_bend', 'Z_spar'][i,j+1] += (M_spar[j] + M_ball_elem[j] + A_spar[j]) * X_sparelem[j] / 2. + 1025. * np.pi / 4. * D_spar[j]**2. * X_sparelem[j] * (z - Z_spar[i])
				partials['hull_mom_acc_bend', 'D_spar'][i,j] += 1025. * np.pi / 2. * D_spar[j] * dz * X_sparelem[j] * (z - Z_spar[i])
				partials['hull_mom_acc_bend', 'X_sparelem'][i,j] += (M_spar[j] + M_ball_elem[j] + A_spar[j]) * (z - Z_spar[i])

				partials['hull_mom_damp_surge', 'Cd'][i,0] += 0.5 * 1025. * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * D_spar[j] * dz * (z - Z_spar[i])
				partials['hull_mom_damp_surge', 'stddev_vel_X_sparelem'][i,j] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * D_spar[j] * dz * (z - Z_spar[i])
				partials['hull_mom_damp_surge', 'D_spar'][i,j] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * dz * (z - Z_spar[i])
				partials['hull_mom_damp_surge', 'Z_spar'][i,i] += -drag_spar[j]
				partials['hull_mom_damp_surge', 'Z_spar'][i,j] += drag_spar[j] / 2. - 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * D_spar[j] * (z - Z_spar[i])
				if j != (N_spar - 1):
					partials['hull_mom_damp_surge', 'Z_spar'][i,j+1] += drag_spar[j] / 2. + 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * D_spar[j] * (z - Z_spar[i])

				partials['hull_mom_damp_pitch', 'Cd'][i,0] += 0.5 * 1025. * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * D_spar[j] * dz * z * (z - Z_spar[i])
				partials['hull_mom_damp_pitch', 'stddev_vel_X_sparelem'][i,j] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * D_spar[j] * dz * z * (z - Z_spar[i])
				partials['hull_mom_damp_pitch', 'D_spar'][i,j] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * dz * z * (z - Z_spar[i])
				partials['hull_mom_damp_pitch', 'Z_spar'][i,i] += -drag_spar[j] * z
				partials['hull_mom_damp_pitch', 'Z_spar'][i,j] += drag_spar[j] * (z - Z_spar[i]) / 2. + drag_spar[j] * z / 2. - 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * D_spar[j] * z * (z - Z_spar[i])
				if j != (N_spar - 1):
					partials['hull_mom_damp_pitch', 'Z_spar'][i,j+1] += drag_spar[j] * (z - Z_spar[i]) / 2. + drag_spar[j] * z / 2. + 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * D_spar[j] * z * (z - Z_spar[i])

				partials['hull_mom_damp_bend', 'Cd'][i,0] += 0.5 * 1025. * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * D_spar[j] * dz * X_sparelem[j] * (z - Z_spar[i])
				partials['hull_mom_damp_bend', 'stddev_vel_X_sparelem'][i,j] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * D_spar[j] * dz * X_sparelem[j] * (z - Z_spar[i])
				partials['hull_mom_damp_bend', 'D_spar'][i,j] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * dz * X_sparelem[j] * (z - Z_spar[i])
				partials['hull_mom_damp_bend', 'X_sparelem'][i,j] += drag_spar[j] * (z - Z_spar[i])
				partials['hull_mom_damp_bend', 'Z_spar'][i,i] += -drag_spar[j] * X_sparelem[j]
				partials['hull_mom_damp_bend', 'Z_spar'][i,j] += drag_spar[j] * X_sparelem[j] / 2. - 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * D_spar[j] * X_sparelem[j] * (z - Z_spar[i])
				if j != (N_spar - 1):
					partials['hull_mom_damp_bend', 'Z_spar'][i,j+1] += drag_spar[j] * X_sparelem[j] / 2. + 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * stddev_vel_X_sparelem[j] * D_spar[j] * X_sparelem[j] * (z - Z_spar[i])
				
				partials['hull_mom_grav_pitch', 'M_spar'][i,j] += 9.80665 * (z - Z_spar[i])
				partials['hull_mom_grav_pitch', 'M_ball_elem'][i,j] += 9.80665 * (z - Z_spar[i])
				partials['hull_mom_grav_pitch', 'D_spar'][i,j] += -np.pi / 2. * D_spar[j] * dz * 1025. * 9.80665 * (z - Z_spar[i])
				partials['hull_mom_grav_pitch', 'Z_spar'][i,i] += -(M_spar[j] + M_ball_elem[j] - 1025. * V_spar[j]) * 9.80665
				partials['hull_mom_grav_pitch', 'Z_spar'][i,j] += (M_spar[j] + M_ball_elem[j] - 1025. * V_spar[j]) * 9.80665 / 2. + 1025. * np.pi / 4. * D_spar[j]**2. * 9.80665 * (z - Z_spar[i])
				if j != (N_spar - 1):
					partials['hull_mom_grav_pitch', 'Z_spar'][i,j+1] += (M_spar[j] + M_ball_elem[j] - 1025. * V_spar[j]) * 9.80665 / 2. - 1025. * np.pi / 4. * D_spar[j]**2. * 9.80665 * (z - Z_spar[i])
				
				partials['hull_mom_grav_bend', 'M_spar'][i,j] += 9.80665 * (X_sparelem[j] - X_sparnode[i])
				partials['hull_mom_grav_bend', 'M_ball_elem'][i,j] += 9.80665 * (X_sparelem[j] - X_sparnode[i])
				partials['hull_mom_grav_bend', 'D_spar'][i,j] += -np.pi / 2. * D_spar[j] * dz * 1025. * 9.80665 * (X_sparelem[j] - X_sparnode[i])
				partials['hull_mom_grav_bend', 'X_sparnode'][i,i] += -(M_spar[j] + M_ball_elem[j] - 1025. * V_spar[j]) * 9.80665
				partials['hull_mom_grav_bend', 'X_sparelem'][i,j] += (M_spar[j] + M_ball_elem[j] - 1025. * V_spar[j]) * 9.80665
				partials['hull_mom_grav_bend', 'Z_spar'][i,j] += 1025. * np.pi / 4. * D_spar[j]**2. * 9.80665 * (X_sparelem[j] - X_sparnode[i])
				if j != (N_spar - 1):
					partials['hull_mom_grav_bend', 'Z_spar'][i,j+1] += -1025. * np.pi / 4. * D_spar[j]**2. * 9.80665 * (X_sparelem[j] - X_sparnode[i])

			if Z_spar[i] < z_moor:
				partials['hull_mom_fairlead', 'K_moor'][i,0] += -(z_moor - Z_spar[i])
				partials['hull_mom_fairlead', 'z_moor'][i,0] += -K_moor
				partials['hull_mom_fairlead', 'Z_spar'][i,i] += K_moor