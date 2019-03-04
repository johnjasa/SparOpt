import numpy as np

from openmdao.api import ExplicitComponent

class RespSpectrumSurgeWF(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_surge', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_surge', val=np.zeros(N_omega), units='m/m')
		self.add_input('S_wave', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('S_wind', val=np.zeros(N_omega), units='m**2/(rad*s)')

		self.add_output('resp_surge_WF', val=np.zeros(N_omega), units='m**2*s/rad')

		self.declare_partials('resp_surge_WF', 'Re_RAO_wave_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_surge_WF', 'Im_RAO_wave_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_surge_WF', 'S_wave', rows=np.arange(N_omega), cols=np.arange(N_omega))

	def compute(self, inputs, outputs):
		outputs['resp_surge_WF'] = np.abs(inputs['Re_RAO_wave_surge'] + 1j * inputs['Im_RAO_wave_surge'])**2. * inputs['S_wave']

	def compute_partials(self, inputs, partials):
		partials['resp_surge_WF', 'Re_RAO_wave_surge'] = 2. * inputs['Re_RAO_wave_surge'] * inputs['S_wave']
		partials['resp_surge_WF', 'Im_RAO_wave_surge'] = 2. * inputs['Im_RAO_wave_surge'] * inputs['S_wave']
		partials['resp_surge_WF', 'S_wave'] = np.abs(inputs['Re_RAO_wave_surge'] + 1j * inputs['Im_RAO_wave_surge'])**2.