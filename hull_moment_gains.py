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

		#self.declare_partials('*', '*')

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

		N_spar = len(Z_spar) - 1
		N_tower = len(z_towernode)

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

		for i in xrange(N_spar):
			for j in xrange(N_tower-1):
				z = (z_towernode[j] + z_towernode[j+1]) / 2.
				outputs['hull_mom_acc_surge'][i] += M_tower[j] * (z - Z_spar[i])
				outputs['hull_mom_acc_pitch'][i] += M_tower[j] * z * (z - Z_spar[i])
				outputs['hull_mom_acc_bend'][i] += M_tower[j] * x_towerelem[j] * (z - Z_spar[i])
				outputs['hull_mom_grav_pitch'][i] += M_tower[j] * 9.80665 * (z - Z_spar[i])
				outputs['hull_mom_grav_bend'][i] += M_tower[j] * 9.80665 * (x_towerelem[j] - X_spar[i])
			outputs['hull_mom_acc_surge'][i] += M_nacelle * (CoG_nacelle - Z_spar[i]) + M_rotor * (CoG_rotor - Z_spar[i])
			outputs['hull_mom_acc_pitch'][i] += M_nacelle * CoG_nacelle * (CoG_nacelle - Z_spar[i]) + M_rotor * CoG_rotor * (CoG_rotor - Z_spar[i]) + I_rotor
			outputs['hull_mom_acc_bend'][i] += M_nacelle * (CoG_nacelle - Z_spar[i]) + M_rotor * (CoG_rotor - Z_spar[i]) + I_rotor * x_d_towertop
			outputs['hull_mom_damp_surge'][i] += dthrust_dv * (CoG_rotor - Z_spar[i])
			outputs['hull_mom_damp_pitch'][i] += dthrust_dv * CoG_rotor * (CoG_rotor - Z_spar[i]) + dmoment_dv
			outputs['hull_mom_damp_bend'][i] += dthrust_dv * (CoG_rotor - Z_spar[i]) + x_d_towertop * dmoment_dv
			outputs['hull_mom_grav_pitch'][i] += M_nacelle * 9.80665 * (CoG_nacelle - Z_spar[i]) + M_rotor * 9.80665 * (CoG_rotor - Z_spar[i])
			outputs['hull_mom_grav_bend'][i] += M_nacelle * 9.80665 * (1. - X_spar[i]) + M_rotor * 9.80665 * (1. - X_spar[i])
			outputs['hull_mom_rotspeed'][i] += dthrust_drotspeed * (CoG_rotor - Z_spar[i])
			outputs['hull_mom_bldpitch'][i] += dthrust_dbldpitch * (CoG_rotor - Z_spar[i])

			for j in xrange(i,N_spar):
				z = (Z_spar[j] + Z_spar[j+1]) / 2.
				outputs['hull_mom_acc_surge'][i] += (M_spar[j] + A_spar[j]) * (z - Z_spar[i])
				outputs['hull_mom_acc_pitch'][i] += (M_spar[j] + A_spar[j]) * z * (z - Z_spar[i])
				outputs['hull_mom_acc_bend'][i] += (M_spar[j] + A_spar[j]) * X_spar[j] * (z - Z_spar[i])
				outputs['hull_mom_damp_surge'][i] += drag_spar[j] * (z - Z_spar[i])
				outputs['hull_mom_damp_pitch'][i] += drag_spar[j] * z * (z - Z_spar[i])
				outputs['hull_mom_damp_bend'][i] += drag_spar[j] * X_spar[j] * (z - Z_spar[i])
				outputs['hull_mom_grav_pitch'][i] += (M_spar[j] - 1025. * V_spar[j]) * 9.80665 * (z - Z_spar[i])
				outputs['hull_mom_grav_bend'][i] += (M_spar[j] - 1025. * V_spar[j]) * 9.80665 * (X_spar[j] - X_spar[i])