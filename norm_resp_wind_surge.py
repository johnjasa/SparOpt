import numpy as np

from openmdao.api import ExplicitComponent

class NormRespWindSurge(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('thrust_wind', val=np.zeros(N_omega), units='m/s')
		self.add_input('torque_wind', val=np.zeros(N_omega), units='m/s')
		self.add_input('Re_H_feedbk', val=np.zeros((N_omega,11,6)))
		self.add_input('Im_H_feedbk', val=np.zeros((N_omega,11,6)))

		self.add_output('Re_RAO_wind_surge', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_output('Im_RAO_wind_surge', val=np.zeros(N_omega), units='m/(m/s)')

		Hcols = Hcols1 = np.array([0,2])
		for i in xrange(1,N_omega):
			Hcols = np.concatenate((Hcols,i*11*6+Hcols1),0)

		self.declare_partials('Re_RAO_wind_surge', 'thrust_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_surge', 'torque_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_surge', 'Re_H_feedbk', rows=np.repeat(np.arange(N_omega),2), cols=Hcols)
		self.declare_partials('Re_RAO_wind_surge', 'Im_H_feedbk', rows=np.repeat(np.arange(N_omega),2), cols=Hcols)
		self.declare_partials('Im_RAO_wind_surge', 'thrust_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_surge', 'torque_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_surge', 'Re_H_feedbk', rows=np.repeat(np.arange(N_omega),2), cols=Hcols)
		self.declare_partials('Im_RAO_wind_surge', 'Im_H_feedbk', rows=np.repeat(np.arange(N_omega),2), cols=Hcols)

	def compute(self, inputs, outputs):
		omega = self.omega

		thrust_wind = inputs['thrust_wind']
		torque_wind = inputs['torque_wind']

		H_feedbk = inputs['Re_H_feedbk'] + 1j * inputs['Im_H_feedbk']

		RAO_wind_surge = H_feedbk[:,0,0] * thrust_wind + H_feedbk[:,0,2] * torque_wind

		outputs['Re_RAO_wind_surge'] = np.real(RAO_wind_surge)
		outputs['Im_RAO_wind_surge'] = np.imag(RAO_wind_surge)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		partials['Re_RAO_wind_surge', 'thrust_wind'] = inputs['Re_H_feedbk'][:,0,0]
		partials['Re_RAO_wind_surge', 'torque_wind'] = inputs['Re_H_feedbk'][:,0,2]
		partials['Im_RAO_wind_surge', 'thrust_wind'] = inputs['Im_H_feedbk'][:,0,0]
		partials['Im_RAO_wind_surge', 'torque_wind'] = inputs['Im_H_feedbk'][:,0,2]

		partials['Re_RAO_wind_surge', 'Im_H_feedbk'] = np.zeros(2*N_omega)
		partials['Im_RAO_wind_surge', 'Re_H_feedbk'] = np.zeros(2*N_omega)

		for i in xrange(N_omega):
			partials['Re_RAO_wind_surge', 'Re_H_feedbk'][2*i] = inputs['thrust_wind'][i]
			partials['Re_RAO_wind_surge', 'Re_H_feedbk'][2*i+1] = inputs['torque_wind'][i]
			partials['Im_RAO_wind_surge', 'Im_H_feedbk'][2*i] = inputs['thrust_wind'][i]
			partials['Im_RAO_wind_surge', 'Im_H_feedbk'][2*i+1] = inputs['torque_wind'][i]