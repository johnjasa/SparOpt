import numpy as np

from openmdao.api import Group, DirectSolver

from viscous_damping import ViscousDamping
from global_damping import GlobalDamping
from A_str_damp import AstrDamp
from A_struct import Astruct
from A_feedbk import Afeedbk
#from transfer_function import TransferFunction
from transfer_function_pre import TransferFunctionPre
from transfer_function_pre_inv import TransferFunctionPreInv
from transfer_function2 import TransferFunction
from norm_resp_wave_surge import NormRespWaveSurge
from norm_resp_wave_pitch import NormRespWavePitch
from norm_resp_wave_bend import NormRespWaveBend
from norm_resp_wind_surge import NormRespWindSurge
from norm_resp_wind_pitch import NormRespWindPitch
from norm_resp_wind_bend import NormRespWindBend
from norm_resp_Mwind_surge import NormRespMWindSurge
from norm_resp_Mwind_pitch import NormRespMWindPitch
from norm_resp_Mwind_bend import NormRespMWindBend
from norm_vel_wave_surge import NormVelWaveSurge
from norm_vel_wave_pitch import NormVelWavePitch
from norm_vel_wave_bend import NormVelWaveBend
from norm_vel_wind_surge import NormVelWindSurge
from norm_vel_wind_pitch import NormVelWindPitch
from norm_vel_wind_bend import NormVelWindBend
from norm_vel_Mwind_surge import NormVelMWindSurge
from norm_vel_Mwind_pitch import NormVelMWindPitch
from norm_vel_Mwind_bend import NormVelMWindBend
from vel_distr import VelDistr

class Viscous(Group):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']

		self.add_subsystem('viscous_damping', ViscousDamping(), promotes_inputs=['Cd', 'x_sparelem', 'z_sparnode', 'Z_spar', 'D_spar', 'stddev_vel_distr'], promotes_outputs=['B_visc_11', 'B_visc_15', 'B_visc_17', 'B_visc_55', 'B_visc_57', 'B_visc_77'])
		
		self.add_subsystem('global_damping', GlobalDamping(), promotes_inputs=['B_aero_11', 'B_aero_15', 'B_aero_17', 'B_aero_55', 'B_aero_57', 'B_aero_77', 'B_struct_77', 'B_visc_11', 'B_visc_15', 'B_visc_17', 'B_visc_55', 'B_visc_57', 'B_visc_77'], promotes_outputs=['B_global'])

		A_str_damp = AstrDamp()
		A_str_damp.linear_solver = DirectSolver()

		self.add_subsystem('A_str_damp', A_str_damp, promotes_inputs=['M_global', 'A_global', 'B_global'], promotes_outputs=['Astr_damp'])

		self.add_subsystem('A_struct', Astruct(), promotes_inputs=['CoG_rotor', 'I_d', 'dtorque_dv', 'dtorque_drotspeed', 'Astr_stiff', 'Astr_damp', 'Astr_ext'], promotes_outputs=['A_struct'])

		self.add_subsystem('A_feedbk', Afeedbk(), promotes_inputs=['A_struct', 'A_contrl', 'BsCc', 'BcCs'], promotes_outputs=['A_feedbk'])

		#transfer_function = TransferFunction(freqs=freqs)
		#transfer_function.linear_solver = LinearBlockGS()

		#self.add_subsystem('transfer_function', transfer_function, promotes_inputs=['A_feedbk', 'B_feedbk'], promotes_outputs=['Re_H_feedbk', 'Im_H_feedbk'])

		self.add_subsystem('transfer_function_pre', TransferFunctionPre(freqs=freqs), promotes_inputs=['A_feedbk'], promotes_outputs=['Re_IA', 'Im_IA'])

		self.add_subsystem('transfer_function_pre_inv', TransferFunctionPreInv(freqs=freqs), promotes_inputs=['Re_IA', 'Im_IA'], promotes_outputs=['Re_IA_inv', 'Im_IA_inv'])

		self.add_subsystem('transfer_function', TransferFunction(freqs=freqs), promotes_inputs=['Re_IA_inv', 'Im_IA_inv', 'B_feedbk'], promotes_outputs=['Re_H_feedbk', 'Im_H_feedbk'])

		self.add_subsystem('norm_resp_wave_surge', NormRespWaveSurge(freqs=freqs), promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wave_surge', 'Im_RAO_wave_surge'])

		self.add_subsystem('norm_resp_wave_pitch', NormRespWavePitch(freqs=freqs), promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wave_pitch', 'Im_RAO_wave_pitch'])

		self.add_subsystem('norm_resp_wave_bend', NormRespWaveBend(freqs=freqs), promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wave_bend', 'Im_RAO_wave_bend'])

		self.add_subsystem('norm_resp_wind_surge', NormRespWindSurge(freqs=freqs), promotes_inputs=['thrust_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wind_surge', 'Im_RAO_wind_surge'])

		self.add_subsystem('norm_resp_wind_pitch', NormRespWindPitch(freqs=freqs), promotes_inputs=['thrust_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wind_pitch', 'Im_RAO_wind_pitch'])

		self.add_subsystem('norm_resp_wind_bend', NormRespWindBend(freqs=freqs), promotes_inputs=['thrust_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_wind_bend', 'Im_RAO_wind_bend'])

		self.add_subsystem('norm_resp_Mwind_surge', NormRespMWindSurge(freqs=freqs), promotes_inputs=['moment_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_Mwind_surge', 'Im_RAO_Mwind_surge'])

		self.add_subsystem('norm_resp_Mwind_pitch', NormRespMWindPitch(freqs=freqs), promotes_inputs=['moment_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_Mwind_pitch', 'Im_RAO_Mwind_pitch'])

		self.add_subsystem('norm_resp_Mwind_bend', NormRespMWindBend(freqs=freqs), promotes_inputs=['moment_wind', 'Re_H_feedbk', 'Im_H_feedbk'], promotes_outputs=['Re_RAO_Mwind_bend', 'Im_RAO_Mwind_bend'])

		self.add_subsystem('norm_vel_wave_surge', NormVelWaveSurge(freqs=freqs), promotes_inputs=['Re_RAO_wave_surge', 'Im_RAO_wave_surge'], promotes_outputs=['Re_RAO_wave_vel_surge', 'Im_RAO_wave_vel_surge'])

		self.add_subsystem('norm_vel_wave_pitch', NormVelWavePitch(freqs=freqs), promotes_inputs=['Re_RAO_wave_pitch', 'Im_RAO_wave_pitch'], promotes_outputs=['Re_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_pitch'])

		self.add_subsystem('norm_vel_wave_bend', NormVelWaveBend(freqs=freqs), promotes_inputs=['Re_RAO_wave_bend', 'Im_RAO_wave_bend'], promotes_outputs=['Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_bend'])

		self.add_subsystem('norm_vel_wind_surge', NormVelWindSurge(freqs=freqs), promotes_inputs=['Re_RAO_wind_surge', 'Im_RAO_wind_surge'], promotes_outputs=['Re_RAO_wind_vel_surge', 'Im_RAO_wind_vel_surge'])

		self.add_subsystem('norm_vel_wind_pitch', NormVelWindPitch(freqs=freqs), promotes_inputs=['Re_RAO_wind_pitch', 'Im_RAO_wind_pitch'], promotes_outputs=['Re_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_pitch'])

		self.add_subsystem('norm_vel_wind_bend', NormVelWindBend(freqs=freqs), promotes_inputs=['Re_RAO_wind_bend', 'Im_RAO_wind_bend'], promotes_outputs=['Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_bend'])

		self.add_subsystem('norm_vel_Mwind_surge', NormVelMWindSurge(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_surge', 'Im_RAO_Mwind_surge'], promotes_outputs=['Re_RAO_Mwind_vel_surge', 'Im_RAO_Mwind_vel_surge'])

		self.add_subsystem('norm_vel_Mwind_pitch', NormVelMWindPitch(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_pitch', 'Im_RAO_Mwind_pitch'], promotes_outputs=['Re_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_pitch'])

		self.add_subsystem('norm_vel_Mwind_bend', NormVelMWindBend(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_bend', 'Im_RAO_Mwind_bend'], promotes_outputs=['Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_bend'])

		self.add_subsystem('vel_distr', VelDistr(freqs=freqs), promotes_inputs=['Re_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_surge', 'Im_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_bend', 'Re_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_surge', 'Im_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_bend', 'Re_RAO_Mwind_vel_surge', 'Re_RAO_Mwind_vel_pitch', 'Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_surge', 'Im_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_bend', 'S_wave', 'S_wind', 'z_sparnode', 'x_sparelem'], promotes_outputs=['stddev_vel_distr'])