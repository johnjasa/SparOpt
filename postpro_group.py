import numpy as np

from openmdao.api import Group

from interp_wave_forces import InterpWaveForces
from norm_resp_wave_surge import NormRespWaveSurge
from norm_resp_wave_pitch import NormRespWavePitch
from norm_resp_wave_bend import NormRespWaveBend
from norm_resp_wave_rotspeed import NormRespWaveRotspeed
from norm_resp_wave_bldpitch import NormRespWaveBldpitch
from norm_resp_wind_surge import NormRespWindSurge
from norm_resp_wind_pitch import NormRespWindPitch
from norm_resp_wind_bend import NormRespWindBend
from norm_resp_wind_rotspeed import NormRespWindRotspeed
from norm_resp_wind_bldpitch import NormRespWindBldpitch
from norm_resp_Mwind_surge import NormRespMWindSurge
from norm_resp_Mwind_pitch import NormRespMWindPitch
from norm_resp_Mwind_bend import NormRespMWindBend
from norm_resp_Mwind_rotspeed import NormRespMWindRotspeed
from norm_resp_Mwind_bldpitch import NormRespMWindBldpitch
from norm_vel_wave_surge import NormVelWaveSurge
from norm_vel_wave_pitch import NormVelWavePitch
from norm_vel_wave_bend import NormVelWaveBend
from norm_vel_wind_surge import NormVelWindSurge
from norm_vel_wind_pitch import NormVelWindPitch
from norm_vel_wind_bend import NormVelWindBend
from norm_vel_Mwind_surge import NormVelMWindSurge
from norm_vel_Mwind_pitch import NormVelMWindPitch
from norm_vel_Mwind_bend import NormVelMWindBend
from norm_acc_wave_surge import NormAccWaveSurge
from norm_acc_wave_pitch import NormAccWavePitch
from norm_acc_wave_bend import NormAccWaveBend
from norm_acc_wind_surge import NormAccWindSurge
from norm_acc_wind_pitch import NormAccWindPitch
from norm_acc_wind_bend import NormAccWindBend
from norm_acc_Mwind_surge import NormAccMWindSurge
from norm_acc_Mwind_pitch import NormAccMWindPitch
from norm_acc_Mwind_bend import NormAccMWindBend
from resp_spectrum_surge import RespSpectrumSurge
from resp_spectrum_pitch import RespSpectrumPitch
from resp_spectrum_bend import RespSpectrumBend
from resp_spectrum_rotspeed import RespSpectrumRotspeed
from resp_spectrum_bldpitch import RespSpectrumBldpitch
from vel_spectrum_surge import VelSpectrumSurge
from vel_spectrum_pitch import VelSpectrumPitch
from vel_spectrum_bend import VelSpectrumBend
#from vel_distr import VelDistr
from std_dev_resp import StdDevResp
from std_dev_resp_vel import StdDevRespVel

class Postpro(Group):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']

		self.add_subsystem('interp_wave_forces', InterpWaveForces(freqs=freqs), promotes_inputs=['Re_wave_forces', 'Im_wave_forces'], promotes_outputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend'])

	 	self.add_subsystem('norm_resp_wave_surge', NormRespWaveSurge(freqs=freqs), promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wave_surge', 'Im_RAO_wave_surge'])

	 	self.add_subsystem('norm_resp_wave_pitch', NormRespWavePitch(freqs=freqs), promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wave_pitch', 'Im_RAO_wave_pitch'])

	 	self.add_subsystem('norm_resp_wave_bend', NormRespWaveBend(freqs=freqs), promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wave_bend', 'Im_RAO_wave_bend'])

	 	self.add_subsystem('norm_resp_wave_rotspeed', NormRespWaveRotspeed(freqs=freqs), promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wave_rotspeed', 'Im_RAO_wave_rotspeed'])

	 	self.add_subsystem('norm_resp_wave_bldpitch', NormRespWaveBldpitch(freqs=freqs), promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'gain_corr_factor'], promotes_outputs=['Re_RAO_wave_bldpitch', 'Im_RAO_wave_bldpitch'])

	 	self.add_subsystem('norm_resp_wind_surge', NormRespWindSurge(freqs=freqs), promotes_inputs=['thrust_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wind_surge', 'Im_RAO_wind_surge'])

	 	self.add_subsystem('norm_resp_wind_pitch', NormRespWindPitch(freqs=freqs), promotes_inputs=['thrust_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wind_pitch', 'Im_RAO_wind_pitch'])

	 	self.add_subsystem('norm_resp_wind_bend', NormRespWindBend(freqs=freqs), promotes_inputs=['thrust_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wind_bend', 'Im_RAO_wind_bend'])

	 	self.add_subsystem('norm_resp_wind_rotspeed', NormRespWindRotspeed(freqs=freqs), promotes_inputs=['thrust_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wind_rotspeed', 'Im_RAO_wind_rotspeed'])

	 	self.add_subsystem('norm_resp_wind_bldpitch', NormRespWindBldpitch(freqs=freqs), promotes_inputs=['thrust_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'gain_corr_factor'], promotes_outputs=['Re_RAO_wind_bldpitch', 'Im_RAO_wind_bldpitch'])

	 	self.add_subsystem('norm_resp_Mwind_surge', NormRespMWindSurge(freqs=freqs), promotes_inputs=['moment_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_Mwind_surge', 'Im_RAO_Mwind_surge'])

	 	self.add_subsystem('norm_resp_Mwind_pitch', NormRespMWindPitch(freqs=freqs), promotes_inputs=['moment_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_Mwind_pitch', 'Im_RAO_Mwind_pitch'])

	 	self.add_subsystem('norm_resp_Mwind_bend', NormRespMWindBend(freqs=freqs), promotes_inputs=['moment_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_Mwind_bend', 'Im_RAO_Mwind_bend'])

	 	self.add_subsystem('norm_resp_Mwind_rotspeed', NormRespMWindRotspeed(freqs=freqs), promotes_inputs=['moment_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_Mwind_rotspeed', 'Im_RAO_Mwind_rotspeed'])

	 	self.add_subsystem('norm_resp_Mwind_bldpitch', NormRespMWindBldpitch(freqs=freqs), promotes_inputs=['moment_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'gain_corr_factor'], promotes_outputs=['Re_RAO_Mwind_bldpitch', 'Im_RAO_Mwind_bldpitch'])

	 	self.add_subsystem('norm_vel_wave_surge', NormVelWaveSurge(freqs=freqs), promotes_inputs=['Re_RAO_wave_surge', 'Im_RAO_wave_surge'], promotes_outputs=['Re_RAO_wave_vel_surge', 'Im_RAO_wave_vel_surge'])

	 	self.add_subsystem('norm_vel_wave_pitch', NormVelWavePitch(freqs=freqs), promotes_inputs=['Re_RAO_wave_pitch', 'Im_RAO_wave_pitch'], promotes_outputs=['Re_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_pitch'])

	 	self.add_subsystem('norm_vel_wave_bend', NormVelWaveBend(freqs=freqs), promotes_inputs=['Re_RAO_wave_bend', 'Im_RAO_wave_bend'], promotes_outputs=['Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_bend'])

	 	self.add_subsystem('norm_vel_wind_surge', NormVelWindSurge(freqs=freqs), promotes_inputs=['Re_RAO_wind_surge', 'Im_RAO_wind_surge'], promotes_outputs=['Re_RAO_wind_vel_surge', 'Im_RAO_wind_vel_surge'])

	 	self.add_subsystem('norm_vel_wind_pitch', NormVelWindPitch(freqs=freqs), promotes_inputs=['Re_RAO_wind_pitch', 'Im_RAO_wind_pitch'], promotes_outputs=['Re_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_pitch'])

	 	self.add_subsystem('norm_vel_wind_bend', NormVelWindBend(freqs=freqs), promotes_inputs=['Re_RAO_wind_bend', 'Im_RAO_wind_bend'], promotes_outputs=['Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_bend'])

		self.add_subsystem('norm_vel_Mwind_surge', NormVelMWindSurge(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_surge', 'Im_RAO_Mwind_surge'], promotes_outputs=['Re_RAO_Mwind_vel_surge', 'Im_RAO_Mwind_vel_surge'])

	 	self.add_subsystem('norm_vel_Mwind_pitch', NormVelMWindPitch(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_pitch', 'Im_RAO_Mwind_pitch'], promotes_outputs=['Re_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_pitch'])

	 	self.add_subsystem('norm_vel_Mwind_bend', NormVelMWindBend(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_bend', 'Im_RAO_Mwind_bend'], promotes_outputs=['Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_bend'])

	 	self.add_subsystem('norm_acc_wave_surge', NormAccWaveSurge(freqs=freqs), promotes_inputs=['Re_RAO_wave_surge', 'Im_RAO_wave_surge'], promotes_outputs=['Re_RAO_wave_acc_surge', 'Im_RAO_wave_acc_surge'])

	 	self.add_subsystem('norm_acc_wave_pitch', NormAccWavePitch(freqs=freqs), promotes_inputs=['Re_RAO_wave_pitch', 'Im_RAO_wave_pitch'], promotes_outputs=['Re_RAO_wave_acc_pitch', 'Im_RAO_wave_acc_pitch'])

	 	self.add_subsystem('norm_acc_wave_bend', NormAccWaveBend(freqs=freqs), promotes_inputs=['Re_RAO_wave_bend', 'Im_RAO_wave_bend'], promotes_outputs=['Re_RAO_wave_acc_bend', 'Im_RAO_wave_acc_bend'])

	 	self.add_subsystem('norm_acc_wind_surge', NormAccWindSurge(freqs=freqs), promotes_inputs=['Re_RAO_wind_surge', 'Im_RAO_wind_surge'], promotes_outputs=['Re_RAO_wind_acc_surge', 'Im_RAO_wind_acc_surge'])

	 	self.add_subsystem('norm_acc_wind_pitch', NormAccWindPitch(freqs=freqs), promotes_inputs=['Re_RAO_wind_pitch', 'Im_RAO_wind_pitch'], promotes_outputs=['Re_RAO_wind_acc_pitch', 'Im_RAO_wind_acc_pitch'])

	 	self.add_subsystem('norm_acc_wind_bend', NormAccWindBend(freqs=freqs), promotes_inputs=['Re_RAO_wind_bend', 'Im_RAO_wind_bend'], promotes_outputs=['Re_RAO_wind_acc_bend', 'Im_RAO_wind_acc_bend'])

		self.add_subsystem('norm_acc_Mwind_surge', NormAccMWindSurge(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_surge', 'Im_RAO_Mwind_surge'], promotes_outputs=['Re_RAO_Mwind_acc_surge', 'Im_RAO_Mwind_acc_surge'])

	 	self.add_subsystem('norm_acc_Mwind_pitch', NormAccMWindPitch(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_pitch', 'Im_RAO_Mwind_pitch'], promotes_outputs=['Re_RAO_Mwind_acc_pitch', 'Im_RAO_Mwind_acc_pitch'])

	 	self.add_subsystem('norm_acc_Mwind_bend', NormAccMWindBend(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_bend', 'Im_RAO_Mwind_bend'], promotes_outputs=['Re_RAO_Mwind_acc_bend', 'Im_RAO_Mwind_acc_bend'])

		self.add_subsystem('resp_spectrum_surge', RespSpectrumSurge(freqs=freqs), promotes_inputs=['Re_RAO_wave_surge', 'Im_RAO_wave_surge', 'Re_RAO_wind_surge', 'Im_RAO_wind_surge', 'Re_RAO_Mwind_surge', 'Im_RAO_Mwind_surge', 'S_wave', 'S_wind'], promotes_outputs=['resp_surge'])

		self.add_subsystem('resp_spectrum_pitch', RespSpectrumPitch(freqs=freqs), promotes_inputs=['Re_RAO_wave_pitch', 'Im_RAO_wave_pitch', 'Re_RAO_wind_pitch', 'Im_RAO_wind_pitch', 'Re_RAO_Mwind_pitch', 'Im_RAO_Mwind_pitch', 'S_wave', 'S_wind'], promotes_outputs=['resp_pitch'])

		self.add_subsystem('resp_spectrum_bend', RespSpectrumBend(freqs=freqs), promotes_inputs=['Re_RAO_wave_bend', 'Im_RAO_wave_bend', 'Re_RAO_wind_bend', 'Im_RAO_wind_bend', 'Re_RAO_Mwind_bend', 'Im_RAO_Mwind_bend', 'S_wave', 'S_wind'], promotes_outputs=['resp_bend'])

		self.add_subsystem('resp_spectrum_rotspeed', RespSpectrumRotspeed(freqs=freqs), promotes_inputs=['Re_RAO_wave_rotspeed', 'Im_RAO_wave_rotspeed', 'Re_RAO_wind_rotspeed', 'Im_RAO_wind_rotspeed', 'Re_RAO_Mwind_rotspeed', 'Im_RAO_Mwind_rotspeed', 'S_wave', 'S_wind'], promotes_outputs=['resp_rotspeed'])

		self.add_subsystem('resp_spectrum_bldpitch', RespSpectrumBldpitch(freqs=freqs), promotes_inputs=['Re_RAO_wave_bldpitch', 'Im_RAO_wave_bldpitch', 'Re_RAO_wind_bldpitch', 'Im_RAO_wind_bldpitch', 'Re_RAO_Mwind_bldpitch', 'Im_RAO_Mwind_bldpitch', 'S_wave', 'S_wind'], promotes_outputs=['resp_bldpitch'])

	 	self.add_subsystem('vel_spectrum_surge', VelSpectrumSurge(freqs=freqs), promotes_inputs=['Re_RAO_wave_vel_surge', 'Im_RAO_wave_vel_surge', 'Re_RAO_wind_vel_surge', 'Im_RAO_wind_vel_surge', 'Re_RAO_Mwind_vel_surge', 'Im_RAO_Mwind_vel_surge', 'S_wave', 'S_wind'], promotes_outputs=['resp_vel_surge'])

	 	self.add_subsystem('vel_spectrum_pitch', VelSpectrumPitch(freqs=freqs), promotes_inputs=['Re_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_pitch', 'Re_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_pitch', 'Re_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_pitch', 'S_wave', 'S_wind'], promotes_outputs=['resp_vel_pitch'])

	 	self.add_subsystem('vel_spectrum_bend', VelSpectrumBend(freqs=freqs), promotes_inputs=['Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_bend', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_bend', 'Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_bend', 'S_wave', 'S_wind'], promotes_outputs=['resp_vel_bend'])

	 	#self.add_subsystem('vel_distr', VelDistr(freqs=freqs), promotes_inputs=['Re_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_surge', 'Im_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_bend', 'Re_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_surge', 'Im_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_bend', 'Re_RAO_Mwind_vel_surge', 'Re_RAO_Mwind_vel_pitch', 'Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_surge', 'Im_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_bend', 'S_wave', 'S_wind', 'z_sparnode', 'x_sparelem'], promotes_outputs=['stddev_vel_distr'])

	 	self.add_subsystem('std_dev_resp', StdDevResp(freqs=freqs), promotes_inputs=['resp_surge', 'resp_pitch', 'resp_bend', 'resp_rotspeed', 'resp_bldpitch'], promotes_outputs=['stddev_surge', 'stddev_pitch', 'stddev_bend', 'stddev_rotspeed', 'stddev_bldpitch'])

	 	self.add_subsystem('std_dev_resp_vel', StdDevRespVel(freqs=freqs), promotes_inputs=['resp_vel_surge', 'resp_vel_pitch', 'resp_vel_bend'], promotes_outputs=['stddev_vel_surge', 'stddev_vel_pitch', 'stddev_vel_bend'])