import numpy as np

from openmdao.api import Group

from norm_resp_wave_rotspeed import NormRespWaveRotspeed
from norm_resp_wave_bldpitch import NormRespWaveBldpitch
from norm_resp_wind_rotspeed import NormRespWindRotspeed
from norm_resp_wind_bldpitch import NormRespWindBldpitch
from norm_resp_Mwind_rotspeed import NormRespMWindRotspeed
from norm_resp_Mwind_bldpitch import NormRespMWindBldpitch
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
from std_dev_resp import StdDevResp
from tower_moment_gains import TowerMomentGains
from norm_tower_moment_wave import NormTowerMomentWave
from norm_tower_moment_wind import NormTowerMomentWind
from norm_tower_moment_Mwind import NormTowerMomentMWind
from tower_moment_spectrum import TowerMomentSpectrum
from tower_stress_spectrum import TowerStressSpectrum
from dirlik import Dirlik

class Postpro(Group):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']

	 	self.add_subsystem('norm_resp_wave_rotspeed', NormRespWaveRotspeed(freqs=freqs), promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wave_rotspeed', 'Im_RAO_wave_rotspeed'])

	 	self.add_subsystem('norm_resp_wave_bldpitch', NormRespWaveBldpitch(freqs=freqs), promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'gain_corr_factor'], promotes_outputs=['Re_RAO_wave_bldpitch', 'Im_RAO_wave_bldpitch'])

	 	self.add_subsystem('norm_resp_wind_rotspeed', NormRespWindRotspeed(freqs=freqs), promotes_inputs=['thrust_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wind_rotspeed', 'Im_RAO_wind_rotspeed'])

	 	self.add_subsystem('norm_resp_wind_bldpitch', NormRespWindBldpitch(freqs=freqs), promotes_inputs=['thrust_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'gain_corr_factor'], promotes_outputs=['Re_RAO_wind_bldpitch', 'Im_RAO_wind_bldpitch'])

	 	self.add_subsystem('norm_resp_Mwind_rotspeed', NormRespMWindRotspeed(freqs=freqs), promotes_inputs=['moment_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_Mwind_rotspeed', 'Im_RAO_Mwind_rotspeed'])

	 	self.add_subsystem('norm_resp_Mwind_bldpitch', NormRespMWindBldpitch(freqs=freqs), promotes_inputs=['moment_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'gain_corr_factor'], promotes_outputs=['Re_RAO_Mwind_bldpitch', 'Im_RAO_Mwind_bldpitch'])

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

	 	self.add_subsystem('std_dev_resp', StdDevResp(freqs=freqs), promotes_inputs=['resp_surge', 'resp_pitch', 'resp_bend', 'resp_rotspeed', 'resp_bldpitch'], promotes_outputs=['stddev_surge', 'stddev_pitch', 'stddev_bend', 'stddev_rotspeed', 'stddev_bldpitch'])

		self.add_subsystem('tower_moment_gains', TowerMomentGains(), promotes_inputs=['M_tower', 'M_nacelle', 'M_rotor', 'I_rotor', 'CoG_nacelle', 'CoG_rotor', 'z_towernode', 'x_towerelem', 'x_towernode', 'x_d_towertop', 'dthrust_dv', 'dmoment_dv', 'dthrust_drotspeed', 'dthrust_dbldpitch'], promotes_outputs=['mom_acc_surge', 'mom_acc_pitch', 'mom_acc_bend', 'mom_damp_surge', 'mom_damp_pitch', 'mom_damp_bend', 'mom_grav_pitch', 'mom_grav_bend', 'mom_rotspeed', 'mom_bldpitch'])

	 	self.add_subsystem('norm_tower_moment_wave', NormTowerMomentWave(freqs=freqs), promotes_inputs=['mom_acc_surge', 'mom_acc_pitch', 'mom_acc_bend', 'mom_damp_surge', 'mom_damp_pitch', 'mom_damp_bend', 'mom_grav_pitch', 'mom_grav_bend', 'mom_rotspeed', 'mom_bldpitch', 'Re_RAO_wave_pitch', 'Re_RAO_wave_bend', 'Re_RAO_wave_rotspeed', 'Re_RAO_wave_bldpitch', 'Im_RAO_wave_pitch', 'Im_RAO_wave_bend', 'Im_RAO_wave_rotspeed', 'Im_RAO_wave_bldpitch', 'Re_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_surge', 'Im_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_bend', 'Re_RAO_wave_acc_surge', 'Re_RAO_wave_acc_pitch', 'Re_RAO_wave_acc_bend', 'Im_RAO_wave_acc_surge', 'Im_RAO_wave_acc_pitch', 'Im_RAO_wave_acc_bend'], promotes_outputs=['Re_RAO_wave_tower_moment', 'Im_RAO_wave_tower_moment'])

	 	self.add_subsystem('norm_tower_moment_wind', NormTowerMomentWind(freqs=freqs), promotes_inputs=['mom_acc_surge', 'mom_acc_pitch', 'mom_acc_bend', 'mom_damp_surge', 'mom_damp_pitch', 'mom_damp_bend', 'mom_grav_pitch', 'mom_grav_bend', 'mom_rotspeed', 'mom_bldpitch', 'Re_RAO_wind_pitch', 'Re_RAO_wind_bend', 'Re_RAO_wind_rotspeed', 'Re_RAO_wind_bldpitch', 'Im_RAO_wind_pitch', 'Im_RAO_wind_bend', 'Im_RAO_wind_rotspeed', 'Im_RAO_wind_bldpitch', 'Re_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_surge', 'Im_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_bend', 'Re_RAO_wind_acc_surge', 'Re_RAO_wind_acc_pitch', 'Re_RAO_wind_acc_bend', 'Im_RAO_wind_acc_surge', 'Im_RAO_wind_acc_pitch', 'Im_RAO_wind_acc_bend', 'CoG_rotor', 'Z_tower', 'dthrust_dv', 'thrust_wind'], promotes_outputs=['Re_RAO_wind_tower_moment', 'Im_RAO_wind_tower_moment'])

	 	self.add_subsystem('norm_tower_moment_Mwind', NormTowerMomentMWind(freqs=freqs), promotes_inputs=['mom_acc_surge', 'mom_acc_pitch', 'mom_acc_bend', 'mom_damp_surge', 'mom_damp_pitch', 'mom_damp_bend', 'mom_grav_pitch', 'mom_grav_bend', 'mom_rotspeed', 'mom_bldpitch', 'Re_RAO_Mwind_pitch', 'Re_RAO_Mwind_bend', 'Re_RAO_Mwind_rotspeed', 'Re_RAO_Mwind_bldpitch', 'Im_RAO_Mwind_pitch', 'Im_RAO_Mwind_bend', 'Im_RAO_Mwind_rotspeed', 'Im_RAO_Mwind_bldpitch', 'Re_RAO_Mwind_vel_surge', 'Re_RAO_Mwind_vel_pitch', 'Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_surge', 'Im_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_bend', 'Re_RAO_Mwind_acc_surge', 'Re_RAO_Mwind_acc_pitch', 'Re_RAO_Mwind_acc_bend', 'Im_RAO_Mwind_acc_surge', 'Im_RAO_Mwind_acc_pitch', 'Im_RAO_Mwind_acc_bend', 'dmoment_dv', 'moment_wind'], promotes_outputs=['Re_RAO_Mwind_tower_moment', 'Im_RAO_Mwind_tower_moment'])

	 	self.add_subsystem('tower_moment_spectrum', TowerMomentSpectrum(freqs=freqs), promotes_inputs=['Re_RAO_wave_tower_moment', 'Im_RAO_wave_tower_moment', 'Re_RAO_wind_tower_moment', 'Im_RAO_wind_tower_moment', 'Re_RAO_Mwind_tower_moment', 'Im_RAO_Mwind_tower_moment', 'S_wave', 'S_wind'], promotes_outputs=['resp_tower_moment'])

	 	self.add_subsystem('tower_stress_spectrum', TowerStressSpectrum(freqs=freqs), promotes_inputs=['resp_tower_moment', 'D_tower_p', 'wt_tower_p'], promotes_outputs=['resp_tower_stress'])

	 	self.add_subsystem('dirlik', Dirlik(freqs=freqs), promotes_inputs=['resp_tower_stress', 'wt_tower_p'], promotes_outputs=['fatigue_damage'])