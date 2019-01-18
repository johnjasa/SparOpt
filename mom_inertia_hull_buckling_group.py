import numpy as np

from openmdao.api import Group

from net_pressure import NetPressure
from N_hull import NHull
from MyMz_hull import MyMzHull
from QyQz_hull import QyQzHull
from T_hull import THull
from hull_r_hull import HullRHull
from hull_axial_stress import HullAxialStress
from hull_bending_stress import HullBendingStress
from hull_beta import HullBeta
from hull_l_eo import HullLEo
from hull_alpha import HullAlpha
from hull_zeta import HullZeta
from hull_hoop_stress import HullHoopStress
from hull_shear_stress import HullShearStress
from hull_Z_l import HullZL
from hull_r_0 import HullR0
from hull_r_f import HullRF
from hull_z_t import HullZT
from hull_hoop_stress_ringstiff import HullHoopStressRingstiff
from hull_F_E import HullFE
from hull_sigma_0 import HullSigma0
from hull_mises_stress import HullMisesStress
from hull_lambda_s import HullLambdaS
from hull_f_ks import HullFKs
from hull_gamma_M import HullGammaM
from hull_f_ksd import HullFKsd
from hull_f_r import HullFR
from hull_delta_0 import HullDelta0
from hull_I_x import HullIX
from hull_I_xh import HullIXh
from hull_I_h import HullIH
from hull_I_R import HullIR
from hull_l_ef import HullLEf
from mom_inertia_ringstiff import MomInertiaRingstiff
from constr_mom_inertia_ringstiff import ConstrMomInertiaRingstiff

class MomInertiaHullBuckling(Group):

	def setup(self):
		
		self.add_subsystem('net_pressure', NetPressure(), promotes_inputs=['Z_spar'], promotes_outputs=['net_pressure'])
		
		self.add_subsystem('N_hull', NHull(), promotes_inputs=['D_spar_p', 'Z_spar', 'M_spar', 'M_ball', 'L_ball', 'spar_draft', 'M_moor', 'z_moor'], promotes_outputs=['N_hull'])
		
		#self.add_subsystem('MyMz_hull', MyMzHull(), promotes_inputs=['dthrust_dv'], promotes_outputs=['My_hull', 'Mz_hull']) #for angle = 0 deg, Mz = 0. My found from balance
		
		self.add_subsystem('QyQz_hull', QyQzHull(), promotes_inputs=['dthrust_dv'], promotes_outputs=['Qy_hull', 'Qz_hull'])
		
		self.add_subsystem('T_hull', THull(), promotes_inputs=['dmoment_dv'], promotes_outputs=['T_hull'])
		
		self.add_subsystem('hull_r_hull', HullRHull(), promotes_inputs=['D_spar_p', 'wt_spar_p'], promotes_outputs=['r_hull'])
		
		self.add_subsystem('hull_axial_stress', HullAxialStress(), promotes_inputs=['N_hull', 'r_hull', 'wt_spar_p'], promotes_outputs=['sigma_a'])
		
		self.add_subsystem('hull_bending_stress', HullBendingStress(), promotes_inputs=['My_hull', 'Mz_hull', 'r_hull', 'wt_spar_p', 'angle_hull'], promotes_outputs=['sigma_m'])
		
		self.add_subsystem('hull_beta', HullBeta(), promotes_inputs=['l_stiff', 'r_hull', 'wt_spar_p'], promotes_outputs=['beta'])
		
		self.add_subsystem('hull_l_eo', HullLEo(), promotes_inputs=['l_stiff', 'beta'], promotes_outputs=['l_eo'])
		
		self.add_subsystem('hull_alpha', HullAlpha(), promotes_inputs=['A_R', 'l_eo', 'wt_spar_p'], promotes_outputs=['alpha'])
		
		self.add_subsystem('hull_zeta', HullZeta(), promotes_inputs=['beta'], promotes_outputs=['zeta'])
		
		self.add_subsystem('hull_hoop_stress', HullHoopStress(), promotes_inputs=['net_pressure', 'r_hull', 'wt_spar_p', 'alpha', 'zeta', 'sigma_a', 'sigma_m'], promotes_outputs=['sigma_h'])
		
		#self.add_subsystem('hull_shear_stress', HullShearStress(), promotes_inputs=['T_hull', 'Qy_hull', 'Qz_hull', 'r_hull', 'wt_spar_p', 'angle_hull'], promotes_outputs=['tau']) #no shear for angle = 0 deg, ignore effect of torsion (verified OK)
		
		self.add_subsystem('hull_Z_l', HullZL(), promotes_inputs=['l_stiff', 'r_hull', 'wt_spar_p'], promotes_outputs=['Z_l'])
		
		self.add_subsystem('hull_r_0', HullR0(), promotes_inputs=['D_spar_p', 'wt_spar_p', 't_f_stiff', 't_w_stiff', 'b_stiff', 'h_stiff', 'l_eo', 'r_hull'], promotes_outputs=['r_0'])
		
		self.add_subsystem('hull_r_f', HullRF(), promotes_inputs=['D_spar_p', 'wt_spar_p', 't_f_stiff', 'h_stiff'], promotes_outputs=['r_f'])
		
		self.add_subsystem('hull_z_t', HullZT(), promotes_inputs=['r_0', 'r_f'], promotes_outputs=['z_t'])
		
		self.add_subsystem('hull_hoop_stress_ringstiff', HullHoopStressRingstiff(), promotes_inputs=['net_pressure', 'r_hull', 'wt_spar_p', 'alpha', 'r_f', 'sigma_a', 'sigma_m'], promotes_outputs=['sigma_hR'])
		
		self.add_subsystem('hull_F_E', HullFE(), promotes_inputs=['wt_spar_p', 'l_stiff', 'r_hull', 'Z_l'], promotes_outputs=['f_Ea', 'f_Em', 'f_Eh', 'f_Etau'])
		
		self.add_subsystem('hull_sigma_0', HullSigma0(), promotes_inputs=['sigma_a', 'sigma_m', 'sigma_h'], promotes_outputs=['sigma_a0', 'sigma_m0', 'sigma_h0'])
		
		self.add_subsystem('hull_mises_stress', HullMisesStress(), promotes_inputs=['sigma_a', 'sigma_m', 'sigma_h', 'tau'], promotes_outputs=['sigma_j'])
		
		self.add_subsystem('hull_lambda_s', HullLambdaS(), promotes_inputs=['f_y', 'sigma_j', 'sigma_a0', 'sigma_m0', 'sigma_h0', 'tau', 'f_Ea', 'f_Em', 'f_Eh', 'f_Etau'], promotes_outputs=['lambda_s'])
		
		self.add_subsystem('hull_f_ks', HullFKs(), promotes_inputs=['f_y', 'lambda_s'], promotes_outputs=['f_ks'])
		
		self.add_subsystem('hull_gamma_M', HullGammaM(), promotes_inputs=['lambda_s'], promotes_outputs=['gamma_M_hull'])
		
		self.add_subsystem('hull_f_ksd', HullFKsd(), promotes_inputs=['f_ks', 'gamma_M_hull'], promotes_outputs=['f_ksd'])
		
		self.add_subsystem('hull_f_r', HullFR(), promotes_inputs=['f_y'], promotes_outputs=['f_r'])
		
		self.add_subsystem('hull_delta_0', HullDelta0(), promotes_inputs=['r_hull'], promotes_outputs=['delta_0'])
		
		self.add_subsystem('hull_I_x', HullIX(), promotes_inputs=['sigma_a', 'sigma_m', 'wt_spar_p', 'r_0', 'l_stiff'], promotes_outputs=['I_x'])
		
		self.add_subsystem('hull_I_xh', HullIXh(), promotes_inputs=['tau', 'r_0', 'spar_draft', 'wt_spar_p', 'l_stiff'], promotes_outputs=['I_xh'])
		
		self.add_subsystem('hull_I_h', HullIH(), promotes_inputs=['net_pressure', 'r_hull', 'r_0', 'l_stiff', 'z_t', 'delta_0', 'f_r', 'sigma_hR'], promotes_outputs=['I_h'])
		
		self.add_subsystem('hull_I_R', HullIR(), promotes_inputs=['I_x', 'I_xh', 'I_h'], promotes_outputs=['I_R'])
		
		self.add_subsystem('hull_l_ef', HullLEf(), promotes_inputs=['r_hull', 'wt_spar_p', 'l_stiff'], promotes_outputs=['l_ef'])
		
		self.add_subsystem('mom_inertia_ringstiff', MomInertiaRingstiff(), promotes_inputs=['D_spar_p', 'wt_spar_p', 't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'r_hull', 'r_0', 'l_ef'], promotes_outputs=['I_stiff'])
		
		self.add_subsystem('constr_mom_inertia_ringstiff', ConstrMomInertiaRingstiff(), promotes_inputs=['I_R', 'I_stiff'], promotes_outputs=['constr_mom_inertia_ringstiff'])