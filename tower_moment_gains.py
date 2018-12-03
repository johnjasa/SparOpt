import numpy as np

from openmdao.api import ExplicitComponent

class TowerMomentGains(ExplicitComponent):

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

		self.add_output('mom_acc_surge', val=np.zeros(11), units='kg*m')
		self.add_output('mom_acc_pitch', val=np.zeros(11), units='kg*m**2/rad')
		self.add_output('mom_acc_bend', val=np.zeros(11), units='kg*m')
		self.add_output('mom_damp_surge', val=np.zeros(11), units='N*s')
		self.add_output('mom_damp_pitch', val=np.zeros(11), units='N*m*s/rad')
		self.add_output('mom_damp_bend', val=np.zeros(11), units='N*s')
		self.add_output('mom_grav_pitch', val=np.zeros(11), units='N*m/rad')
		self.add_output('mom_grav_bend', val=np.zeros(11), units='N')
		self.add_output('mom_rotspeed', val=np.zeros(11), units='N*m*s/rad')
		self.add_output('mom_bldpitch', val=np.zeros(11), units='N*m/rad')

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

		N_tower = len(z_towernode)

		outputs['mom_acc_surge'] = np.zeros(N_tower)
		outputs['mom_acc_pitch'] = np.zeros(N_tower)
		outputs['mom_acc_bend'] = np.zeros(N_tower)
		outputs['mom_damp_surge'] = np.zeros(N_tower)
		outputs['mom_damp_pitch'] = np.zeros(N_tower)
		outputs['mom_damp_bend'] = np.zeros(N_tower)
		outputs['mom_grav_pitch'] = np.zeros(N_tower)
		outputs['mom_grav_bend'] = np.zeros(N_tower)
		outputs['mom_rotspeed'] = np.zeros(N_tower)
		outputs['mom_bldpitch'] = np.zeros(N_tower)

		for i in xrange(N_tower):
			for j in xrange(i,N_tower-1):
				z = (z_towernode[j] + z_towernode[j+1]) / 2.
				outputs['mom_acc_surge'][i] += M_tower[j] * (z - z_towernode[i])
				outputs['mom_acc_pitch'][i] += M_tower[j] * z * (z - z_towernode[i])
				outputs['mom_acc_bend'][i] += M_tower[j] * x_towerelem[j] * (z - z_towernode[i])
				outputs['mom_grav_pitch'][i] += M_tower[j] * 9.80665 * (z - z_towernode[i])
				outputs['mom_grav_bend'][i] += M_tower[j] * 9.80665 * (x_towerelem[j] - x_towernode[i])
			outputs['mom_acc_surge'][i] += M_nacelle * (CoG_nacelle - z_towernode[i]) + M_rotor * (CoG_rotor - z_towernode[i])
			outputs['mom_acc_pitch'][i] += M_nacelle * CoG_nacelle * (CoG_nacelle - z_towernode[i]) + M_rotor * CoG_rotor * (CoG_rotor - z_towernode[i]) + I_rotor
			outputs['mom_acc_bend'][i] += M_nacelle * (CoG_nacelle - z_towernode[i]) + M_rotor * (CoG_rotor - z_towernode[i]) + I_rotor * x_d_towertop
			outputs['mom_damp_surge'][i] += dthrust_dv * (CoG_rotor - z_towernode[i])
			outputs['mom_damp_pitch'][i] += dthrust_dv * CoG_rotor * (CoG_rotor - z_towernode[i]) + dmoment_dv
			outputs['mom_damp_bend'][i] += dthrust_dv * (CoG_rotor - z_towernode[i]) + x_d_towertop * dmoment_dv
			outputs['mom_grav_pitch'][i] += M_nacelle * 9.80665 * (CoG_nacelle - z_towernode[i]) + M_rotor * 9.80665 * (CoG_rotor - z_towernode[i])
			outputs['mom_grav_bend'][i] += M_nacelle * 9.80665 * (1. - x_towernode[i]) + M_rotor * 9.80665 * (1. - x_towernode[i])
			outputs['mom_rotspeed'][i] += dthrust_drotspeed * (CoG_rotor - z_towernode[i])
			outputs['mom_bldpitch'][i] += dthrust_dbldpitch * (CoG_rotor - z_towernode[i])

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

		N_tower = len(z_towernode)

		partials['mom_acc_surge', 'M_tower'] = np.zeros((N_tower,N_tower-1))
		partials['mom_acc_surge', 'M_nacelle'] = np.zeros((N_tower,1))
		partials['mom_acc_surge', 'M_rotor'] = np.zeros((N_tower,1))
		partials['mom_acc_surge', 'I_rotor'] = np.zeros((N_tower,1))
		partials['mom_acc_surge', 'CoG_nacelle'] = np.zeros((N_tower,1))
		partials['mom_acc_surge', 'CoG_rotor'] = np.zeros((N_tower,1))
		partials['mom_acc_surge', 'z_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_acc_surge', 'x_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_acc_surge', 'x_towerelem'] = np.zeros((N_tower,N_tower-1))
		partials['mom_acc_surge', 'x_d_towertop'] = np.zeros((N_tower,1))
		partials['mom_acc_surge', 'dthrust_dv'] = np.zeros((N_tower,1))
		partials['mom_acc_surge', 'dmoment_dv'] = np.zeros((N_tower,1))
		partials['mom_acc_surge', 'dthrust_drotspeed'] = np.zeros((N_tower,1))
		partials['mom_acc_surge', 'dthrust_dbldpitch'] = np.zeros((N_tower,1))

		partials['mom_acc_pitch', 'M_tower'] = np.zeros((N_tower,N_tower-1))
		partials['mom_acc_pitch', 'M_nacelle'] = np.zeros((N_tower,1))
		partials['mom_acc_pitch', 'M_rotor'] = np.zeros((N_tower,1))
		partials['mom_acc_pitch', 'I_rotor'] = np.zeros((N_tower,1))
		partials['mom_acc_pitch', 'CoG_nacelle'] = np.zeros((N_tower,1))
		partials['mom_acc_pitch', 'CoG_rotor'] = np.zeros((N_tower,1))
		partials['mom_acc_pitch', 'z_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_acc_pitch', 'x_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_acc_pitch', 'x_towerelem'] = np.zeros((N_tower,N_tower-1))
		partials['mom_acc_pitch', 'x_d_towertop'] = np.zeros((N_tower,1))
		partials['mom_acc_pitch', 'dthrust_dv'] = np.zeros((N_tower,1))
		partials['mom_acc_pitch', 'dmoment_dv'] = np.zeros((N_tower,1))
		partials['mom_acc_pitch', 'dthrust_drotspeed'] = np.zeros((N_tower,1))
		partials['mom_acc_pitch', 'dthrust_dbldpitch'] = np.zeros((N_tower,1))

		partials['mom_acc_bend', 'M_tower'] = np.zeros((N_tower,N_tower-1))
		partials['mom_acc_bend', 'M_nacelle'] = np.zeros((N_tower,1))
		partials['mom_acc_bend', 'M_rotor'] = np.zeros((N_tower,1))
		partials['mom_acc_bend', 'I_rotor'] = np.zeros((N_tower,1))
		partials['mom_acc_bend', 'CoG_nacelle'] = np.zeros((N_tower,1))
		partials['mom_acc_bend', 'CoG_rotor'] = np.zeros((N_tower,1))
		partials['mom_acc_bend', 'z_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_acc_bend', 'x_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_acc_bend', 'x_towerelem'] = np.zeros((N_tower,N_tower-1))
		partials['mom_acc_bend', 'x_d_towertop'] = np.zeros((N_tower,1))
		partials['mom_acc_bend', 'dthrust_dv'] = np.zeros((N_tower,1))
		partials['mom_acc_bend', 'dmoment_dv'] = np.zeros((N_tower,1))
		partials['mom_acc_bend', 'dthrust_drotspeed'] = np.zeros((N_tower,1))
		partials['mom_acc_bend', 'dthrust_dbldpitch'] = np.zeros((N_tower,1))

		partials['mom_damp_surge', 'M_tower'] = np.zeros((N_tower,N_tower-1))
		partials['mom_damp_surge', 'M_nacelle'] = np.zeros((N_tower,1))
		partials['mom_damp_surge', 'M_rotor'] = np.zeros((N_tower,1))
		partials['mom_damp_surge', 'I_rotor'] = np.zeros((N_tower,1))
		partials['mom_damp_surge', 'CoG_nacelle'] = np.zeros((N_tower,1))
		partials['mom_damp_surge', 'CoG_rotor'] = np.zeros((N_tower,1))
		partials['mom_damp_surge', 'z_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_damp_surge', 'x_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_damp_surge', 'x_towerelem'] = np.zeros((N_tower,N_tower-1))
		partials['mom_damp_surge', 'x_d_towertop'] = np.zeros((N_tower,1))
		partials['mom_damp_surge', 'dthrust_dv'] = np.zeros((N_tower,1))
		partials['mom_damp_surge', 'dmoment_dv'] = np.zeros((N_tower,1))
		partials['mom_damp_surge', 'dthrust_drotspeed'] = np.zeros((N_tower,1))
		partials['mom_damp_surge', 'dthrust_dbldpitch'] = np.zeros((N_tower,1))

		partials['mom_damp_pitch', 'M_tower'] = np.zeros((N_tower,N_tower-1))
		partials['mom_damp_pitch', 'M_nacelle'] = np.zeros((N_tower,1))
		partials['mom_damp_pitch', 'M_rotor'] = np.zeros((N_tower,1))
		partials['mom_damp_pitch', 'I_rotor'] = np.zeros((N_tower,1))
		partials['mom_damp_pitch', 'CoG_nacelle'] = np.zeros((N_tower,1))
		partials['mom_damp_pitch', 'CoG_rotor'] = np.zeros((N_tower,1))
		partials['mom_damp_pitch', 'z_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_damp_pitch', 'x_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_damp_pitch', 'x_towerelem'] = np.zeros((N_tower,N_tower-1))
		partials['mom_damp_pitch', 'x_d_towertop'] = np.zeros((N_tower,1))
		partials['mom_damp_pitch', 'dthrust_dv'] = np.zeros((N_tower,1))
		partials['mom_damp_pitch', 'dmoment_dv'] = np.zeros((N_tower,1))
		partials['mom_damp_pitch', 'dthrust_drotspeed'] = np.zeros((N_tower,1))
		partials['mom_damp_pitch', 'dthrust_dbldpitch'] = np.zeros((N_tower,1))

		partials['mom_damp_bend', 'M_tower'] = np.zeros((N_tower,N_tower-1))
		partials['mom_damp_bend', 'M_nacelle'] = np.zeros((N_tower,1))
		partials['mom_damp_bend', 'M_rotor'] = np.zeros((N_tower,1))
		partials['mom_damp_bend', 'I_rotor'] = np.zeros((N_tower,1))
		partials['mom_damp_bend', 'CoG_nacelle'] = np.zeros((N_tower,1))
		partials['mom_damp_bend', 'CoG_rotor'] = np.zeros((N_tower,1))
		partials['mom_damp_bend', 'z_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_damp_bend', 'x_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_damp_bend', 'x_towerelem'] = np.zeros((N_tower,N_tower-1))
		partials['mom_damp_bend', 'x_d_towertop'] = np.zeros((N_tower,1))
		partials['mom_damp_bend', 'dthrust_dv'] = np.zeros((N_tower,1))
		partials['mom_damp_bend', 'dmoment_dv'] = np.zeros((N_tower,1))
		partials['mom_damp_bend', 'dthrust_drotspeed'] = np.zeros((N_tower,1))
		partials['mom_damp_bend', 'dthrust_dbldpitch'] = np.zeros((N_tower,1))

		partials['mom_grav_pitch', 'M_tower'] = np.zeros((N_tower,N_tower-1))
		partials['mom_grav_pitch', 'M_nacelle'] = np.zeros((N_tower,1))
		partials['mom_grav_pitch', 'M_rotor'] = np.zeros((N_tower,1))
		partials['mom_grav_pitch', 'I_rotor'] = np.zeros((N_tower,1))
		partials['mom_grav_pitch', 'CoG_nacelle'] = np.zeros((N_tower,1))
		partials['mom_grav_pitch', 'CoG_rotor'] = np.zeros((N_tower,1))
		partials['mom_grav_pitch', 'z_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_grav_pitch', 'x_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_grav_pitch', 'x_towerelem'] = np.zeros((N_tower,N_tower-1))
		partials['mom_grav_pitch', 'x_d_towertop'] = np.zeros((N_tower,1))
		partials['mom_grav_pitch', 'dthrust_dv'] = np.zeros((N_tower,1))
		partials['mom_grav_pitch', 'dmoment_dv'] = np.zeros((N_tower,1))
		partials['mom_grav_pitch', 'dthrust_drotspeed'] = np.zeros((N_tower,1))
		partials['mom_grav_pitch', 'dthrust_dbldpitch'] = np.zeros((N_tower,1))

		partials['mom_grav_bend', 'M_tower'] = np.zeros((N_tower,N_tower-1))
		partials['mom_grav_bend', 'M_nacelle'] = np.zeros((N_tower,1))
		partials['mom_grav_bend', 'M_rotor'] = np.zeros((N_tower,1))
		partials['mom_grav_bend', 'I_rotor'] = np.zeros((N_tower,1))
		partials['mom_grav_bend', 'CoG_nacelle'] = np.zeros((N_tower,1))
		partials['mom_grav_bend', 'CoG_rotor'] = np.zeros((N_tower,1))
		partials['mom_grav_bend', 'z_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_grav_bend', 'x_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_grav_bend', 'x_towerelem'] = np.zeros((N_tower,N_tower-1))
		partials['mom_grav_bend', 'x_d_towertop'] = np.zeros((N_tower,1))
		partials['mom_grav_bend', 'dthrust_dv'] = np.zeros((N_tower,1))
		partials['mom_grav_bend', 'dmoment_dv'] = np.zeros((N_tower,1))
		partials['mom_grav_bend', 'dthrust_drotspeed'] = np.zeros((N_tower,1))
		partials['mom_grav_bend', 'dthrust_dbldpitch'] = np.zeros((N_tower,1))

		partials['mom_rotspeed', 'M_tower'] = np.zeros((N_tower,N_tower-1))
		partials['mom_rotspeed', 'M_nacelle'] = np.zeros((N_tower,1))
		partials['mom_rotspeed', 'M_rotor'] = np.zeros((N_tower,1))
		partials['mom_rotspeed', 'I_rotor'] = np.zeros((N_tower,1))
		partials['mom_rotspeed', 'CoG_nacelle'] = np.zeros((N_tower,1))
		partials['mom_rotspeed', 'CoG_rotor'] = np.zeros((N_tower,1))
		partials['mom_rotspeed', 'z_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_rotspeed', 'x_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_rotspeed', 'x_towerelem'] = np.zeros((N_tower,N_tower-1))
		partials['mom_rotspeed', 'x_d_towertop'] = np.zeros((N_tower,1))
		partials['mom_rotspeed', 'dthrust_dv'] = np.zeros((N_tower,1))
		partials['mom_rotspeed', 'dmoment_dv'] = np.zeros((N_tower,1))
		partials['mom_rotspeed', 'dthrust_drotspeed'] = np.zeros((N_tower,1))
		partials['mom_rotspeed', 'dthrust_dbldpitch'] = np.zeros((N_tower,1))

		partials['mom_bldpitch', 'M_tower'] = np.zeros((N_tower,N_tower-1))
		partials['mom_bldpitch', 'M_nacelle'] = np.zeros((N_tower,1))
		partials['mom_bldpitch', 'M_rotor'] = np.zeros((N_tower,1))
		partials['mom_bldpitch', 'I_rotor'] = np.zeros((N_tower,1))
		partials['mom_bldpitch', 'CoG_nacelle'] = np.zeros((N_tower,1))
		partials['mom_bldpitch', 'CoG_rotor'] = np.zeros((N_tower,1))
		partials['mom_bldpitch', 'z_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_bldpitch', 'x_towernode'] = np.zeros((N_tower,N_tower))
		partials['mom_bldpitch', 'x_towerelem'] = np.zeros((N_tower,N_tower-1))
		partials['mom_bldpitch', 'x_d_towertop'] = np.zeros((N_tower,1))
		partials['mom_bldpitch', 'dthrust_dv'] = np.zeros((N_tower,1))
		partials['mom_bldpitch', 'dmoment_dv'] = np.zeros((N_tower,1))
		partials['mom_bldpitch', 'dthrust_drotspeed'] = np.zeros((N_tower,1))
		partials['mom_bldpitch', 'dthrust_dbldpitch'] = np.zeros((N_tower,1))

		for i in xrange(N_tower):
			for j in xrange(i,N_tower-1):
				z = (z_towernode[j] + z_towernode[j+1]) / 2
				partials['mom_acc_surge', 'M_tower'][i,j] += (z - z_towernode[i])
				partials['mom_acc_surge', 'z_towernode'][i,i] += -M_tower[j]
				partials['mom_acc_surge', 'z_towernode'][i,j] += M_tower[j] / 2.
				partials['mom_acc_surge', 'z_towernode'][i,j+1] += M_tower[j] / 2.
				partials['mom_acc_pitch', 'M_tower'][i,j] += z * (z - z_towernode[i])
				partials['mom_acc_pitch', 'z_towernode'][i,i] += -M_tower[j] * z
				partials['mom_acc_pitch', 'z_towernode'][i,j] += M_tower[j] * z
				partials['mom_acc_pitch', 'z_towernode'][i,j+1] += M_tower[j] * z
				partials['mom_acc_bend', 'M_tower'][i,j] += x_towerelem[j] * (z - z_towernode[i])
				partials['mom_acc_bend', 'z_towernode'][i,i] += -M_tower[j] * x_towerelem[j]
				partials['mom_acc_bend', 'z_towernode'][i,j] += M_tower[j] * x_towerelem[j] / 2.
				partials['mom_acc_bend', 'z_towernode'][i,j+1] += M_tower[j] * x_towerelem[j] / 2.
				partials['mom_acc_bend', 'x_towerelem'][i,j] += M_tower[j] * (z - z_towernode[i])
				partials['mom_grav_pitch', 'M_tower'][i,j] += 9.80665 * (z - z_towernode[i])
				partials['mom_grav_pitch', 'z_towernode'][i,i] += -M_tower[j] * 9.80665
				partials['mom_grav_pitch', 'z_towernode'][i,j] += M_tower[j] * 9.80665 / 2.
				partials['mom_grav_pitch', 'z_towernode'][i,j+1] += M_tower[j] * 9.80665 / 2.
				partials['mom_grav_bend', 'M_tower'][i,j] += 9.80665 * (x_towerelem[j] - x_towerelem[i])
				partials['mom_grav_bend', 'x_towernode'][i,i] += -M_tower[j] * 9.80665
				partials['mom_grav_bend', 'x_towerelem'][i,j] += M_tower[j] * 9.80665
			partials['mom_acc_surge', 'M_nacelle'][i,0] += CoG_nacelle - z_towernode[i]
			partials['mom_acc_surge', 'CoG_nacelle'][i,0] += M_nacelle
			partials['mom_acc_surge', 'z_towernode'][i,i] += -M_nacelle - M_rotor
			partials['mom_acc_surge', 'M_rotor'][i,0] += CoG_rotor - z_towernode[i]
			partials['mom_acc_surge', 'CoG_rotor'][i,0] += M_rotor
			partials['mom_acc_pitch', 'M_nacelle'][i,0] += CoG_nacelle * (CoG_nacelle - z_towernode[i])
			partials['mom_acc_pitch', 'CoG_nacelle'][i,0] += M_nacelle * (CoG_nacelle - z_towernode[i]) +  M_nacelle * CoG_nacelle
			partials['mom_acc_pitch', 'z_towernode'][i,i] += -M_nacelle * CoG_nacelle - M_rotor * CoG_rotor
			partials['mom_acc_pitch', 'M_rotor'][i,0] += CoG_rotor * (CoG_rotor - z_towernode[i])
			partials['mom_acc_pitch', 'CoG_rotor'][i,0] += M_rotor * (CoG_rotor - z_towernode[i]) +  M_rotor * CoG_rotor
			partials['mom_acc_pitch', 'I_rotor'][i,0] += 1.
			partials['mom_acc_bend', 'M_nacelle'][i,0] += CoG_nacelle - z_towernode[i]
			partials['mom_acc_bend', 'CoG_nacelle'][i,0] += M_nacelle
			partials['mom_acc_bend', 'z_towernode'][i,i] += -M_nacelle - M_rotor
			partials['mom_acc_bend', 'M_rotor'][i,0] += CoG_rotor - z_towernode[i]
			partials['mom_acc_bend', 'CoG_rotor'][i,0] += M_rotor
			partials['mom_acc_bend', 'I_rotor'][i,0] += x_d_towertop
			partials['mom_acc_bend', 'x_d_towertop'][i,0] += I_rotor
			partials['mom_damp_surge', 'dthrust_dv'][i,0] += CoG_rotor - z_towernode[i]
			partials['mom_damp_surge', 'CoG_rotor'][i,0] += dthrust_dv
			partials['mom_damp_surge', 'z_towernode'][i,i] += -dthrust_dv
			partials['mom_damp_pitch', 'dthrust_dv'][i,0] += CoG_rotor * (CoG_rotor - z_towernode[i])
			partials['mom_damp_pitch', 'CoG_rotor'][i,0] += dthrust_dv * CoG_rotor + dthrust_dv * (CoG_rotor - z_towernode[i])
			partials['mom_damp_pitch', 'z_towernode'][i,i] += -dthrust_dv * CoG_rotor
			partials['mom_damp_pitch', 'dmoment_dv'][i,0] += 1.
			partials['mom_damp_bend', 'dthrust_dv'][i,0] += CoG_rotor - z_towernode[i]
			partials['mom_damp_bend', 'CoG_rotor'][i,0] += dthrust_dv
			partials['mom_damp_bend', 'z_towernode'][i,i] += -dthrust_dv
			partials['mom_damp_bend', 'x_d_towertop'][i,0] += dmoment_dv
			partials['mom_damp_bend', 'dmoment_dv'][i,0] += x_d_towertop
			partials['mom_grav_pitch', 'M_nacelle'][i,0] += 9.80665 * (CoG_nacelle - z_towernode[i])
			partials['mom_grav_pitch', 'CoG_nacelle'][i,0] += M_nacelle * 9.80665
			partials['mom_grav_pitch', 'z_towernode'][i,i] += -(M_nacelle + M_rotor) * 9.80665
			partials['mom_grav_pitch', 'M_rotor'][i,0] += 9.80665 * (CoG_rotor - z_towernode[i])
			partials['mom_grav_pitch', 'CoG_rotor'][i,0] += M_rotor * 9.80665
			partials['mom_grav_bend', 'M_nacelle'][i,0] += 9.80665 * (1. - x_towernode[i])
			partials['mom_grav_bend', 'x_towernode'][i,i] += -(M_nacelle + M_rotor) * 9.80665
			partials['mom_grav_bend', 'M_rotor'][i,0] += 9.80665 * (1. - x_towernode[i])
			partials['mom_rotspeed', 'dthrust_drotspeed'][i,0] += CoG_rotor - z_towernode[i]
			partials['mom_rotspeed', 'CoG_rotor'][i,0] += dthrust_drotspeed
			partials['mom_rotspeed', 'z_towernode'][i,i] += -dthrust_drotspeed
			partials['mom_bldpitch', 'dthrust_dbldpitch'][i,0] += CoG_rotor - z_towernode[i]
			partials['mom_bldpitch', 'CoG_rotor'][i,0] += dthrust_dbldpitch
			partials['mom_bldpitch', 'z_towernode'][i,i] += -dthrust_dbldpitch