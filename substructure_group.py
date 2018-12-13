import numpy as np

from openmdao.api import Group

from spar_diameter import SparDiameter
from spar_thickness import SparThickness
from tower_diameter import TowerDiameter
from tower_thickness import TowerThickness
from draft import Draft
from z_spar import ZSpar
from spar_mass import SparMass
from spar_total_mass import SparTotalMass
from spar_inertia import SparInertia
from spar_addedmass import SparAddedMass
from spar_cog import SparCoG
from z_tower import ZTower
from tower_mass import TowerMass
from tower_total_mass import TowerTotalMass
from turb_mass import TurbMass
from tower_inertia import TowerInertia
from tower_cog import TowerCoG
from turb_inertia import TurbInertia
from turb_cog import TurbCoG
from ballast_elem import BallastElem
from ballast_len import BallastLen
from ballast_mass import BallastMass
from ballast_cog import BallastCoG
from ball_inertia import BallInertia
from total_cog import TotalCoG
from volume import Volume
from buoyancy import Buoyancy
from aero_damping import AeroDamping
from modeshape_spar_nodes import ModeshapeSparNodes
from modeshape_tower_nodes import ModeshapeTowerNodes
from modeshape_elem_mass import ModeshapeElemMass
from modeshape_elem_EI import ModeshapeElemEI
from modeshape_elem_length import ModeshapeElemLength
from modeshape_elem_normforce import ModeshapeElemNormforce
from modeshape_elem_stiff import ModeshapeElemStiff
from modeshape_glob_mass import ModeshapeGlobMass
from modeshape_glob_stiff import ModeshapeGlobStiff
from modeshape_M_inv import ModeshapeMInv
from modeshape_eigmatrix import ModeshapeEigmatrix
from modeshape_eigvector import ModeshapeEigvector
from modeshape_disp import ModeshapeDisp
from tower_node_1_lhs import TowerNode1LHS
from tower_node_1_rhs import TowerNode1RHS
from tower_node_1_deriv import TowerNode1Deriv
from tower_top_deriv import TowerTopDeriv
from tower_elem_disp import TowerElemDisp
from tower_elem_1_deriv import TowerElem1Deriv
from tower_elem_2_deriv import TowerElem2Deriv
from spar_node_1_lhs import SparNode1LHS
from spar_node_1_rhs import SparNode1RHS
from spar_node_1_deriv import SparNode1Deriv
from spar_elem_disp import SparElemDisp
from spar_moor_disp import SparMoorDisp
from spar_elem_1_deriv import SparElem1Deriv
from spar_elem_2_deriv import SparElem2Deriv
from bending_mass import BendingMass
from bending_addedmass import BendingAddedMass
from bending_damping import BendingDamping
from bending_stiffness import BendingStiffness
from global_mass import GlobalMass
from global_addedmass import GlobalAddedMass
from global_stiffness import GlobalStiffness
from wave_number import WaveNumber
from wave_loads import WaveLoads

class Substructure(Group):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']

		self.add_subsystem('spar_diameter', SparDiameter(), promotes_inputs=['D_spar_p'], promotes_outputs=['D_spar'])

		self.add_subsystem('spar_thickness', SparThickness(), promotes_inputs=['wt_spar_p'], promotes_outputs=['wt_spar'])

		self.add_subsystem('tower_diameter', TowerDiameter(), promotes_inputs=['D_tower_p'], promotes_outputs=['D_tower'])

		self.add_subsystem('tower_thickness', TowerThickness(), promotes_inputs=['wt_tower_p'], promotes_outputs=['wt_tower'])

	 	self.add_subsystem('draft', Draft(), promotes_inputs=['L_spar'], promotes_outputs=['spar_draft'])

	 	self.add_subsystem('Z_spar', ZSpar(), promotes_inputs=['L_spar', 'spar_draft'], promotes_outputs=['Z_spar'])

	 	self.add_subsystem('spar_mass', SparMass(), promotes_inputs=['D_spar', 'L_spar', 'wt_spar'], promotes_outputs=['M_spar'])

	 	self.add_subsystem('spar_total_mass', SparTotalMass(), promotes_inputs=['M_spar'], promotes_outputs=['tot_M_spar'])

	 	self.add_subsystem('spar_inertia', SparInertia(), promotes_inputs=['L_spar', 'M_spar', 'spar_draft'], promotes_outputs=['I_spar'])

	 	self.add_subsystem('spar_cog', SparCoG(), promotes_inputs=['L_spar', 'M_spar', 'tot_M_spar', 'spar_draft'], promotes_outputs=['CoG_spar'])

	 	self.add_subsystem('Z_tower', ZTower(), promotes_inputs=['L_tower'], promotes_outputs=['Z_tower'])

	 	self.add_subsystem('tower_mass', TowerMass(), promotes_inputs=['D_tower', 'L_tower', 'wt_tower'], promotes_outputs=['M_tower'])

	 	self.add_subsystem('tower_total_mass', TowerTotalMass(), promotes_inputs=['M_tower'], promotes_outputs=['tot_M_tower'])

	 	self.add_subsystem('turb_mass', TurbMass(), promotes_inputs=['tot_M_tower', 'M_nacelle', 'M_rotor'], promotes_outputs=['M_turb'])

	 	self.add_subsystem('tower_inertia', TowerInertia(), promotes_inputs=['L_tower', 'M_tower'], promotes_outputs=['I_tower'])

	 	self.add_subsystem('tower_cog', TowerCoG(), promotes_inputs=['L_tower', 'M_tower', 'tot_M_tower'], promotes_outputs=['CoG_tower'])

	 	self.add_subsystem('turb_inertia', TurbInertia(), promotes_inputs=['I_tower', 'M_nacelle', 'CoG_nacelle', 'M_rotor', 'CoG_rotor', 'I_rotor'], promotes_outputs=['I_turb'])

		self.add_subsystem('turb_cog', TurbCoG(), promotes_inputs=['tot_M_tower', 'CoG_tower', 'M_nacelle', 'CoG_nacelle', 'M_rotor', 'CoG_rotor', 'M_turb'], promotes_outputs=['CoG_turb'])
	 	
	 	self.add_subsystem('volume', Volume(), promotes_inputs=['D_spar', 'L_spar'], promotes_outputs=['sub_vol'])

	 	self.add_subsystem('buoyancy', Buoyancy(), promotes_inputs=['D_spar', 'L_spar', 'spar_draft', 'sub_vol'], promotes_outputs=['buoy_spar', 'CoB'])

	 	#self.add_subsystem('ballast', Ballast(), promotes_inputs=['spar_draft', 'buoy_spar', 'tot_M_spar', 'D_spar', 'M_turb', 'M_moor_zero', 'rho_ball', 'wt_ball'], promotes_outputs=['M_ball', 'CoG_ball', 'L_ball'])

	 	self.add_subsystem('ballast_elem', BallastElem(), promotes_inputs=['buoy_spar', 'tot_M_spar', 'D_spar', 'L_spar', 'M_turb', 'M_moor_zero', 'rho_ball', 'wt_ball'], promotes_outputs=['L_ball_elem', 'M_ball_elem'])

	 	self.add_subsystem('ballast_len', BallastLen(), promotes_inputs=['L_ball_elem'], promotes_outputs=['L_ball'])

	 	self.add_subsystem('ballast_mass', BallastMass(), promotes_inputs=['M_ball_elem'], promotes_outputs=['M_ball'])

	 	self.add_subsystem('ballast_cog', BallastCoG(), promotes_inputs=['L_ball_elem', 'M_ball_elem', 'M_ball', 'spar_draft'], promotes_outputs=['CoG_ball'])

	 	self.add_subsystem('ball_inertia', BallInertia(), promotes_inputs=['M_ball', 'CoG_ball'], promotes_outputs=['I_ball'])

	 	self.add_subsystem('total_cog', TotalCoG(), promotes_inputs=['M_turb', 'tot_M_spar', 'M_ball', 'CoG_turb', 'CoG_spar', 'CoG_ball'], promotes_outputs=['CoG_total'])

	 	self.add_subsystem('modeshape_spar_nodes', ModeshapeSparNodes(), promotes_inputs=['Z_spar', 'spar_draft', 'L_ball', 'z_moor'], promotes_outputs=['z_sparnode'])

	 	self.add_subsystem('modeshape_tower_nodes', ModeshapeTowerNodes(), promotes_inputs=['Z_tower'], promotes_outputs=['z_towernode'])

	 	self.add_subsystem('modeshape_elem_length', ModeshapeElemLength(), promotes_inputs=['z_sparnode', 'z_towernode'], promotes_outputs=['L_mode_elem'])

	 	self.add_subsystem('modeshape_elem_mass', ModeshapeElemMass(), promotes_inputs=['D_spar', 'L_spar', 'M_spar', 'Z_spar', 'L_tower', 'M_tower', 'spar_draft', 'M_ball_elem', 'L_ball_elem', 'L_ball', 'z_sparnode', 'L_mode_elem'], promotes_outputs=['mel'])

	 	self.add_subsystem('modeshape_elem_EI', ModeshapeElemEI(), promotes_inputs=['D_tower', 'wt_tower'], promotes_outputs=['EI_mode_elem'])

	 	self.add_subsystem('modeshape_elem_normforce', ModeshapeElemNormforce(), promotes_inputs=['D_spar', 'L_spar', 'Z_spar', 'M_spar', 'L_ball', 'M_ball', 'M_ball_elem', 'M_moor', 'z_moor', 'z_sparnode', 'spar_draft', 'M_tower', 'M_nacelle', 'M_rotor', 'tot_M_tower'], promotes_outputs=['normforce_mode_elem'])

	 	self.add_subsystem('modeshape_elem_stiff', ModeshapeElemStiff(), promotes_inputs=['EI_mode_elem', 'L_mode_elem', 'normforce_mode_elem'], promotes_outputs=['kel'])

	 	self.add_subsystem('modeshape_glob_mass', ModeshapeGlobMass(), promotes_inputs=['M_nacelle', 'M_rotor', 'I_rotor', 'mel'], promotes_outputs=['M_mode'])
		
		self.add_subsystem('modeshape_glob_stiff', ModeshapeGlobStiff(), promotes_inputs=['K_moor', 'z_sparnode', 'z_moor', 'kel'], promotes_outputs=['K_mode'])

		self.add_subsystem('modeshape_M_inv', ModeshapeMInv(), promotes_inputs=['M_mode'], promotes_outputs=['M_mode_inv'])

		self.add_subsystem('modeshape_eigmatrix', ModeshapeEigmatrix(), promotes_inputs=['K_mode', 'M_mode_inv'], promotes_outputs=['A_eig'])

		self.add_subsystem('modeshape_eigvector', ModeshapeEigvector(), promotes_inputs=['A_eig', 'struct_damp_ratio'], promotes_outputs=['eig_vector', 'alpha_damp'])

		self.add_subsystem('modeshape_disp', ModeshapeDisp(), promotes_inputs=['eig_vector'], promotes_outputs=['x_sparnode', 'x_towernode'])

	 	self.add_subsystem('tower_node_1_lhs', TowerNode1LHS(), promotes_inputs=['z_towernode'], promotes_outputs=['tower_spline_lhs'])

	 	self.add_subsystem('tower_node_1_rhs', TowerNode1RHS(), promotes_inputs=['z_towernode', 'x_towernode'], promotes_outputs=['tower_spline_rhs'])

	 	self.add_subsystem('tower_node_1_deriv', TowerNode1Deriv(), promotes_inputs=['tower_spline_lhs', 'tower_spline_rhs'], promotes_outputs=['x_d_towernode'])
		
	 	self.add_subsystem('tower_top_deriv', TowerTopDeriv(), promotes_inputs=['x_d_towernode'], promotes_outputs=['x_d_towertop'])
	 	
	 	self.add_subsystem('tower_elem_disp', TowerElemDisp(), promotes_inputs=['z_towernode', 'x_towernode', 'x_d_towernode'], promotes_outputs=['x_towerelem'])

	 	self.add_subsystem('tower_elem_1_deriv', TowerElem1Deriv(), promotes_inputs=['z_towernode', 'x_towernode', 'x_d_towernode'], promotes_outputs=['x_d_towerelem'])

	 	self.add_subsystem('tower_elem_2_deriv', TowerElem2Deriv(), promotes_inputs=['z_towernode', 'x_d_towernode'], promotes_outputs=['x_dd_towerelem'])
		
	 	self.add_subsystem('spar_node_1_lhs', SparNode1LHS(), promotes_inputs=['z_sparnode'], promotes_outputs=['spar_spline_lhs'])

	 	self.add_subsystem('spar_node_1_rhs', SparNode1RHS(), promotes_inputs=['z_sparnode', 'x_sparnode'], promotes_outputs=['spar_spline_rhs'])

	 	self.add_subsystem('spar_node_1_deriv', SparNode1Deriv(), promotes_inputs=['spar_spline_lhs', 'spar_spline_rhs'], promotes_outputs=['x_d_sparnode'])

	 	self.add_subsystem('spar_elem_disp', SparElemDisp(), promotes_inputs=['z_sparnode', 'x_sparnode', 'x_d_sparnode'], promotes_outputs=['x_sparelem'])

	 	self.add_subsystem('spar_moor_disp', SparMoorDisp(), promotes_inputs=['z_sparnode', 'x_sparnode', 'z_moor'], promotes_outputs=['x_moor'])

	 	self.add_subsystem('spar_elem_1_deriv', SparElem1Deriv(), promotes_inputs=['z_sparnode', 'x_sparnode', 'x_d_sparnode'], promotes_outputs=['x_d_sparelem'])

	 	self.add_subsystem('spar_elem_2_deriv', SparElem2Deriv(), promotes_inputs=['z_sparnode', 'x_d_sparnode'], promotes_outputs=['x_dd_sparelem'])

	 	self.add_subsystem('spar_addedmass', SparAddedMass(), promotes_inputs=['z_sparnode', 'Z_spar', 'D_spar'], promotes_outputs=['A11', 'A15', 'A55'])

	 	self.add_subsystem('bending_mass', BendingMass(), promotes_inputs=['z_sparnode', 'x_sparelem', 'z_towernode', 'x_towerelem', 'x_d_towertop', 'M_spar', 'L_spar', 'Z_spar', 'M_tower', 'L_tower', 'Z_tower', 'L_ball_elem', 'M_ball_elem', 'L_ball', 'M_rotor', 'M_nacelle', 'I_rotor', 'spar_draft'], promotes_outputs=['M17', 'M57', 'M77'])
	
		self.add_subsystem('bending_addedmass', BendingAddedMass(), promotes_inputs=['x_sparelem', 'z_sparnode', 'Z_spar', 'D_spar'], promotes_outputs=['A17', 'A57', 'A77'])
		
		self.add_subsystem('bending_damping', BendingDamping(), promotes_inputs=['x_dd_sparelem', 'x_dd_towerelem', 'EI_mode_elem', 'z_sparnode', 'z_towernode', 'alpha_damp'], promotes_outputs=['B_struct_77'])
		
		self.add_subsystem('bending_stiffness', BendingStiffness(), promotes_inputs=['normforce_mode_elem', 'EI_mode_elem', 'x_moor', 'z_sparnode', 'z_towernode', 'x_d_sparelem', 'x_dd_sparelem', 'x_d_towerelem', 'x_dd_towerelem', 'K_moor', 'z_moor'], promotes_outputs=['K17', 'K57', 'K77'])

		self.add_subsystem('aero_damping', AeroDamping(), promotes_inputs=['CoG_rotor', 'dthrust_dv', 'dmoment_dv', 'x_d_towertop'], promotes_outputs=['B_aero_11', 'B_aero_15', 'B_aero_17', 'B_aero_55', 'B_aero_57', 'B_aero_77'])

	 	self.add_subsystem('global_mass', GlobalMass(), promotes_inputs=['tot_M_spar', 'M_ball', 'M_turb', 'CoG_turb', 'CoG_spar', 'CoG_ball', 'I_spar', 'I_ball', 'I_turb', 'M17', 'M57', 'M77'], promotes_outputs=['M_global'])

	 	self.add_subsystem('global_addedmass', GlobalAddedMass(), promotes_inputs=['A11', 'A15', 'A17', 'A55', 'A57', 'A77'], promotes_outputs=['A_global'])

	 	self.add_subsystem('global_stiffness', GlobalStiffness(), promotes_inputs=['D_spar', 'tot_M_spar', 'M_turb', 'M_ball', 'CoG_total', 'M_moor', 'K_moor', 'z_moor', 'K17', 'K57', 'K77', 'buoy_spar', 'CoB'], promotes_outputs=['K_global'])

	 	self.add_subsystem('wave_num', WaveNumber(freqs=freqs), promotes_inputs=['water_depth'], promotes_outputs=['wave_number'])

	 	self.add_subsystem('wave_loads', WaveLoads(freqs=freqs), promotes_inputs=['D_spar', 'Z_spar', 'wave_number', 'water_depth', 'z_sparnode', 'x_sparelem'], promotes_outputs=['Re_wave_forces', 'Im_wave_forces'])