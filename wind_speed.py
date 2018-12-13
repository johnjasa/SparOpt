import numpy as np

from openmdao.api import ExplicitComponent

class WindSpeed(ExplicitComponent):

	def initialize(self):
		self.options.declare('blades', types=dict)
		self.options.declare('freqs', types=dict)

	def setup(self):
		blades = self.options['blades']
		self.N_b_elem = blades['N_b_elem']
		self.cohfolder = blades['cohfolder']

		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('windspeed_0', val=0., units='m/s')
		self.add_input('rotspeed_0', val=0., units='rad/s')
		self.add_input('b_elem_r', val=np.zeros(self.N_b_elem), units='m')
		self.add_input('b_elem_dr', val=0., units='m')
		self.add_input('dFn_dv', val=np.zeros(self.N_b_elem), units='N*s/m**2')
		self.add_input('dFt_dv', val=np.zeros(self.N_b_elem), units='N*s/m**2')

		self.add_output('thrust_wind', val=np.zeros(N_omega), units='m/s')
		self.add_output('moment_wind', val=np.zeros(N_omega), units='m/s')
		self.add_output('torque_wind', val=np.zeros(N_omega), units='m/s')

	def compute(self,inputs,outputs):
		omega = self.omega
		Vhub = inputs['windspeed_0']
		"""
		rotspeed_0 = inputs['rotspeed_0']
		omega = self.omega

		N_b_elem = self.N_b_elem

		domega = omega[1] - omega[0]
		omega_gen = np.arange(freqs[0],freqs[-1] + 3. * rotspeed_0,domega) #internally generated frequencies that are needed due to frequency shift
		N_omega = len(omega)
		N_omega_gen = len(omega_gen)

		S_wind = wind.kaimal(Vhub, omega_gen)

		coh_file = self.cohfolder + 'windspeed' + str(int(Vhub)) + '.dat'

		all_omega, all_K0, all_K1, all_K2, all_K3, all_K4 = np.loadtxt(cohfile, skiprows=1, unpack=True)

		all_omega = np.unique(all_omega)
		all_K0 = np.reshape(all_K0,(len(all_omega),N_b_elem,N_b_elem))
		all_K1 = np.reshape(all_K0,(len(all_omega),N_b_elem,N_b_elem))
		all_K2 = np.reshape(all_K0,(len(all_omega),N_b_elem,N_b_elem))
		all_K3 = np.reshape(all_K0,(len(all_omega),N_b_elem,N_b_elem))
		all_K4 = np.reshape(all_K0,(len(all_omega),N_b_elem,N_b_elem))

		K0 = np.zeros((N_omega_gen,N_b_elem,N_b_elem))
		K1 = np.zeros((N_omega_gen,N_b_elem,N_b_elem))
		K2 = np.zeros((N_omega_gen,N_b_elem,N_b_elem))
		K3 = np.zeros((N_omega_gen,N_b_elem,N_b_elem))
		K4 = np.zeros((N_omega_gen,N_b_elem,N_b_elem))

		for i in xrange(Nfreq):
			for j in xrange(N_b_elem):
				for k in xrange(N_b_elem):
					K0[:,j,k] = np.interp(omega_gen, all_omega, all_K0)
					K1[:,j,k] = np.interp(omega_gen, all_omega, all_K1)
					K2[:,j,k] = np.interp(omega_gen, all_omega, all_K2)
					K3[:,j,k] = np.interp(omega_gen, all_omega, all_K3)
					K4[:,j,k] = np.interp(omega_gen, all_omega, all_K4)

		G0 = np.zeros((N_omega,N_b_elem,N_b_elem))
		Gm3 = np.zeros((N_omega,N_b_elem,N_b_elem))
		Gp3 = np.zeros((N_omega,N_b_elem,N_b_elem))

		Gm1 = np.zeros((N_omega,N_b_elem,N_b_elem))
		Gp1 = np.zeros((N_omega,N_b_elem,N_b_elem))
		Gm2 = np.zeros((N_omega,N_b_elem,N_b_elem))
		Gp2 = np.zeros((N_omega,N_b_elem,N_b_elem))
		Gm4 = np.zeros((N_omega,N_b_elem,N_b_elem))
		Gp4 = np.zeros((N_omega,N_b_elem,N_b_elem))

		for i in xrange(N_omega):
			idx0 = i
			idx_m3 = int(np.round(np.abs(omega_gen[i] + 3. * rotspeed_0) / domega))
			idx_p3 = int(np.round(np.abs(omega_gen[i] - 3. * rotspeed_0) / domega))
			
			G0[i] = K0[idx0] * S_wind[idx0]
			Gm3[i] = K3[idx_m3] * S_wind[idx_m3]
			Gp3[i] = K3[idx_p3] * S_wind[idx_p3]
			
			idx_m4 = int(np.round(np.abs(omega_gen[i] + 3. * rotspeed_0) / domega))
			idx_m2 = int(np.round(np.abs(omega_gen[i] + 3. * rotspeed_0) / domega))
			idx_m1 = i
			idx_p1 = i
			idx_p2 = int(np.round(np.abs(omega_gen[i] - 3. * rotspeed_0) / domega))
			idx_p4 = int(np.round(np.abs(omega_gen[i] - 3. * rotspeed_0) / domega))
			
			Gm1[i] = K1[idx_m1] * S_wind[idx_m1]
			Gp1[i] = K1[idx_p1] * S_wind[idx_p1]
			Gm2[i] = K2[idx_m2] * S_wind[idx_m2]
			Gp2[i] = K2[idx_p2] * S_wind[idx_p2]
			Gm4[i] = K4[idx_m4] * S_wind[idx_m4]
			Gp4[i] = K4[idx_p4] * S_wind[idx_p4]

		G_FQ = G0 + Gp3 + Gm3
		G_M = Gm1 + Gp1 + Gm2 + Gp2 + Gm4 + Gp4

		outputs['thrust_wind'] = np.zeros(N_omega)
		outputs['moment_wind'] = np.zeros(N_omega)
		outputs['torque_wind'] = np.zeros(N_omega)

		dthrust_dv_b = inputs['dFn_dv'] * inputs['b_elem_dr']
		dmoment_dv_b = inputs['dFn_dv'] * inputs['b_elem_r'] * inputs['b_elem_dr']
		dtorque_dv_b = inputs['dFt_dv'] * inputs['b_elem_r'] * inputs['b_elem_dr']

		for i in xrange(N_omega):
			outputs['thrust_wind'][i] = 3.**2. * np.linalg.multi_dot((dthrust_dv_b,G_FQ[i],np.transpose(dthrust_dv_b)))[0][0]
			outputs['moment_wind'][i] = (3./2.)**2. * np.linalg.multi_dot((dmoment_dv_b,G_M[i],np.transpose(dmoment_dv_b)))[0][0]
			outputs['torque_wind'][i] = 3.**2. * np.linalg.multi_dot((dtorque_dv_b,G_FQ[i],np.transpose(dtorque_dv_b)))[0][0]

			outputs['thrust_wind'][i] = np.sqrt(np.abs(thrust_wind[i]) / S_wind[i]) / (3. * np.sum(dthrust_dv_b)) #Equivalent wind speed transfer function
			outputs['moment_wind'][i] = np.sqrt(np.abs(moment_wind[i]) / S_wind[i]) / (3. / 2. * np.sum(dmoment_dv_b))
			outputs['torque_wind'][i] = np.sqrt(np.abs(torque_wind[i]) / S_wind[i]) / (3. * np.sum(dtorque_dv_b))
		"""
		omega_ws = np.linspace(0.014361566416410483,6.283185307179586,3493)
		thrust_wind, moment_wind, torque_wind = np.loadtxt('C:/Code/eq_wind_%d.dat' % Vhub, unpack=True)
		#omega_ws, thrust_wind, moment_wind, torque_wind = np.loadtxt('C:/Code/windspeeds/eq_wind_%d.dat' % Vhub, unpack=True)
		outputs['thrust_wind'] = np.interp(omega,omega_ws,thrust_wind)
		outputs['moment_wind'] = np.interp(omega,omega_ws,moment_wind)
		outputs['torque_wind'] = np.interp(omega,omega_ws,torque_wind)