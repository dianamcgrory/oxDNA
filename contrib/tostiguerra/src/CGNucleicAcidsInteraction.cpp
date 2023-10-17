#include "CGNucleicAcidsInteraction.h"
#include "ParticleFTG.h"

#include <Particles/Molecule.h>
#include <fstream>
using namespace std;

CGNucleicAcidsInteraction::CGNucleicAcidsInteraction() :
				BaseInteraction() {
	ADD_INTERACTION_TO_MAP(BONDED, pair_interaction_bonded);
	ADD_INTERACTION_TO_MAP(NONBONDED, pair_nonbonded_WCA);
	ADD_INTERACTION_TO_MAP(STICKY, pair_nonbonded_sticky);

}

CGNucleicAcidsInteraction::~CGNucleicAcidsInteraction() {

}

void CGNucleicAcidsInteraction::get_settings(input_file &inp) {
	BaseInteraction::get_settings(inp);

	getInputInt(&inp, "DPS_n", &_PS_n, 0);
	getInputInt(&inp, "DPS_chain_size", &_chain_size, 1);

	getInputNumber(&inp, "DPS_alpha", &_PS_alpha, 0);

	if(_PS_alpha != 0.) {
		throw oxDNAException("DetailedPolymerSwap: PS_alpha != 0 is not supported yet!");
	}

	getInputNumber(&inp, "DPS_3b_sigma", &_3b_sigma, 0);
	getInputNumber(&inp, "DPS_3b_range", &_3b_range, 0);
	getInputNumber(&inp, "DPS_3b_lambda", &_3b_lambda, 0);
	getInputNumber(&inp, "DPS_mu", &_mu, 1.0);
	getInputNumber(&inp, "DPS_dS_mod", &dS_mod, 1.0);
	getInputNumber(&inp, "DPS_alpha_mod", &alpha_mod, 1.0);
  getInputNumber(&inp, "DPS_bdG_threshold", &bdG_threshold, 1.0);

	getInputNumber(&inp, "DPS_deltaPatchMon", &_deltaPatchMon, 0);

	getInputString(&inp, "DPS_interaction_matrix_file", _interaction_matrix_file, 1);

	getInputBool(&inp, "DPS_semiflexibility", &_enable_semiflexibility, 0);
	if(_enable_semiflexibility) {
		getInputNumber(&inp, "DPS_semiflexibility_k", &_semiflexibility_k, 1);
		getInputNumber(&inp, "DPS_semiflexibility_a1", &_semiflexibility_a1, 1);
	}
	getInputBool(&inp, "DPS_semiflexibility_3b", &_enable_semiflexibility_3b, 0);

	if(_enable_semiflexibility_3b) {
		getInputNumber(&inp, "DPS_semiflexibility_3b_k", &_semiflexibility_3b_k, 1);
		getInputNumber(&inp, "DPS_semiflexibility_a1", &_semiflexibility_a1, 1);
	}

	getInputBool(&inp, "DPS_stacking", &_enable_patch_stacking, 0);
	if(_enable_patch_stacking) {
		getInputNumber(&inp, "DPS_stacking_eta", &_stacking_eta, 1);
	}

	getInputNumber(&inp, "DPS_rfene", &_rfene, 0);
	getInputNumber(&inp, "DPS_Kfene", &_Kfene, 0);
	getInputNumber(&inp, "DPS_WCA_sigma", &_WCA_sigma, 0);
}

void CGNucleicAcidsInteraction::init() {
	_sqr_rfene = SQR(_rfene);
	_PS_sqr_rep_rcut = pow(2. * _WCA_sigma, 2. / this->_PS_n);

	_rcut = sqrt(_PS_sqr_rep_rcut);

	_3b_rcut = _3b_range * _3b_sigma;
	_sqr_3b_rcut = SQR(_3b_rcut);
	number B_ss = 1. / (1. + 4. * SQR(1. - _3b_range));
	_3b_A_part = -1. / (B_ss - 1.) / exp(1. / (1. - _3b_range));
	_3b_B_part = B_ss * pow(_3b_sigma, 4.);
	_3b_prefactor = _3b_lambda;

	_rcut = _3b_rcut + 2 * (_deltaPatchMon);

	if(_PS_alpha > 0.) {
		if(_rfene > _rcut) {
			_rcut = _rfene;
		}
	}
	_sqr_rcut = SQR(_rcut);

	OX_LOG(Logger::LOG_INFO, "PolymerSwap: A_part: %lf, B_part: %lf, total rcut: %lf (%lf)", _3b_A_part, _3b_B_part, _rcut, _sqr_rcut);

	if(_PS_alpha != 0) {
		if(_PS_alpha < 0.) {
			throw oxDNAException("MG_alpha may not be negative");
		}
		_PS_gamma = M_PI / (_sqr_rfene - pow(2., 1. / 3.));
		_PS_beta = 2 * M_PI - _sqr_rfene * _PS_gamma;
		OX_LOG(Logger::LOG_INFO, "MG: alpha = %lf, beta = %lf, gamma = %lf", _PS_alpha, _PS_beta, _PS_gamma);
	}
}

void CGNucleicAcidsInteraction::_update_inter_chain_stress_tensor(int chain, int ref_chain, LR_vector group_force) {
	LR_vector chain_pos = _chain_coms[ref_chain];
	if(chain != ref_chain) {
		chain_pos += _box->min_image(_chain_coms[ref_chain], _chain_coms[chain]);
	}

	_inter_chain_stress_tensor[0] += chain_pos[0] * group_force[0];
	_inter_chain_stress_tensor[1] += chain_pos[1] * group_force[1];
	_inter_chain_stress_tensor[2] += chain_pos[2] * group_force[2];
	_inter_chain_stress_tensor[3] += chain_pos[0] * group_force[1];
	_inter_chain_stress_tensor[4] += chain_pos[0] * group_force[2];
	_inter_chain_stress_tensor[5] += chain_pos[1] * group_force[2];
}

number CGNucleicAcidsInteraction::P_inter_chain() {
	number V = CONFIG_INFO->box->V();
	return CONFIG_INFO->temperature() * (_N_chains / V) + (_inter_chain_stress_tensor[0] + _inter_chain_stress_tensor[1] + _inter_chain_stress_tensor[2]) / (3. * V);
}

void CGNucleicAcidsInteraction::begin_energy_computation() {
	BaseInteraction::begin_energy_computation();

	std::fill(_inter_chain_stress_tensor.begin(), _inter_chain_stress_tensor.end(), 0.);
	_chain_coms.resize(_N_chains, LR_vector(0., 0., 0.));

	for(int i = 0; i < _N_chains; i++) {
		CONFIG_INFO->molecules()[i]->update_com();
		_chain_coms[i] = CONFIG_INFO->molecules()[i]->com;
	}

	_bonds.clear();
}

bool CGNucleicAcidsInteraction::has_custom_stress_tensor() const {
	return true;
}

number CGNucleicAcidsInteraction::_fene(BaseParticle *p, BaseParticle *q, bool update_forces) {
	number sqr_r = _computed_r.norm();

	if(sqr_r > _sqr_rfene) {
		if(update_forces) {
			throw oxDNAException("The distance between particles %d and %d (r: %lf) exceeds the FENE distance (%lf)", p->index, q->index, sqrt(sqr_r), _rfene);
		}
		set_is_infinite(true);
		return 10e10;
	}

	number energy = -_Kfene * _sqr_rfene * log(1. - sqr_r / _sqr_rfene);

	if(update_forces) {
		// this number is the module of the force over r, so we don't have to divide the distance vector by its module
		number force_mod = -2 * _Kfene * _sqr_rfene / (_sqr_rfene - sqr_r);
		auto force = -_computed_r * force_mod;
		p->force += force;
		q->force -= force;

		_update_stress_tensor(p->pos, force);
		_update_stress_tensor(p->pos + _computed_r, -force);
	}

	return energy;
}

number CGNucleicAcidsInteraction::_WCA(BaseParticle *p, BaseParticle *q, bool update_forces) {
	number sqr_r = _computed_r.norm();
	if(sqr_r > _PS_sqr_rep_rcut) {
		return (number) 0.;
	}

	number energy = 0;
	// this number is the module of the force over r, so we don't have to divide the distance vector for its module
	number force_mod = 0;

	// cut-off for all the repulsive interactions
	if(sqr_r < _PS_sqr_rep_rcut) {
		number part = 1.;
		number ir2_scaled = SQR(_WCA_sigma) / sqr_r;
		for(int i = 0; i < _PS_n / 2; i++) {
			part *= ir2_scaled;
		}
		energy += 4. * (part * (part - 1.)) + 1. - _PS_alpha;
		if(update_forces) {
			force_mod += 4. * _PS_n * part * (2. * part - 1.) / sqr_r;
		}
	}
	// attraction
	/*else if(int_type == 0 && _PS_alpha != 0.) {
	 energy += 0.5 * _PS_alpha * (cos(_PS_gamma * sqr_r + _PS_beta) - 1.0);
	 if(update_forces) {
	 force_mod += _PS_alpha * _PS_gamma * sin(_PS_gamma * sqr_r + _PS_beta);
	 }
	 }*/

	if(update_forces) {
		auto force = -_computed_r * force_mod;
		p->force += force;
		q->force -= force;

		if(p->strand_id != q->strand_id) {
			_update_inter_chain_stress_tensor(p->strand_id, p->strand_id, force);
			_update_inter_chain_stress_tensor(q->strand_id, p->strand_id, -force);
		}

		_update_stress_tensor(p->pos, force);
		_update_stress_tensor(p->pos + _computed_r, -force);
	}

	return energy;
}

number CGNucleicAcidsInteraction::_sticky(BaseParticle *p, BaseParticle *q, bool update_forces) {
	LR_vector p_patch_pos = p->int_centers[0];
	LR_vector q_patch_pos = q->int_centers[0];
	LR_vector patch_dist = _computed_r + q_patch_pos - p_patch_pos;
	number sqr_r = patch_dist.norm();
	number energy = 0.;

	// sticky-sticky
	if(_sticky_interaction(p->btype, q->btype)) {
		if(sqr_r  < _sqr_3b_rcut) {
			number epsilon = _3b_epsilon[p->btype * _interaction_matrix_size + q->btype];

			number r_mod = sqrt(sqr_r);
			number exp_part = exp(_3b_sigma / (r_mod - _3b_rcut));
			number Vradial = epsilon * _3b_A_part * exp_part * (_3b_B_part / SQR(sqr_r) - 1.);

			number cost_a3 = (p->orientationT.v3 * q->orientationT.v3);
			number Vteta_a3 = (1. - cost_a3)/2.;         // axis defining the 3'-5' direction
			// Vteta_a3 = 1;

			//number cost_a1 = (p->orientationT.v1 * q->orientationT.v1);
			//number Vteta_a1 = (1. - cost_a1)/2.;         // axis on which the patch lies
			//number Vteta_a1 = pow((1. - cost_a1), _semiflexibility_a1)/( pow(2., _semiflexibility_a1) );
			number Vteta_a1 = 1.;

			number tmp_energy = Vradial * Vteta_a3 * Vteta_a1;
			energy += tmp_energy;

			number tb_energy = (r_mod < _3b_sigma) ? Vteta_a3 * Vteta_a1: -tmp_energy / epsilon;

			PSBond p_bond(q, tb_energy, epsilon, 1, 1, patch_dist, r_mod, _computed_r);
			PSBond q_bond(p, tb_energy, epsilon, 1, 1, -patch_dist, r_mod, -_computed_r);

			if(update_forces) {
				number force_mod = ( epsilon * _3b_A_part * exp_part * (4. * _3b_B_part / (SQR(sqr_r) * r_mod)) + _3b_sigma * Vradial / SQR(r_mod - _3b_rcut) ) * Vteta_a3 * Vteta_a1;
				LR_vector tmp_force = patch_dist * (-force_mod / r_mod);

				//number torque_a1 = _semiflexibility_a1 * pow((1. - cost_a1), (_semiflexibility_a1 - 1.)) / (pow(2, _semiflexibility_a1));
				number torque_a1 = 0.;
				//LR_vector torque_tetaTerm = Vradial * ( p->orientationT.v3.cross(q->orientationT.v3) )/2. * Vteta_a1 + Vradial * ( p->orientationT.v1.cross(q->orientationT.v1) )/2. * Vteta_a3;
				LR_vector torque_tetaTerm = Vradial * ( p->orientationT.v3.cross(q->orientationT.v3) )/2. * Vteta_a1 + Vradial * torque_a1 * ( p->orientationT.v1.cross(q->orientationT.v1) ) * Vteta_a3;
				// torque_tetaTerm = LR_vector();
				LR_vector p_torque = p->orientationT * ( p_patch_pos.cross(tmp_force) + torque_tetaTerm);
				LR_vector q_torque = q->orientationT * ( q_patch_pos.cross(tmp_force) + torque_tetaTerm);

				p->force += tmp_force;
				q->force -= tmp_force;

				p->torque += p_torque;
				q->torque -= q_torque;

				if(r_mod > _3b_sigma) {
						p_bond.force = tmp_force / epsilon;
						p_bond.p_torque = p_torque / epsilon;
						p_bond.q_torque = q_torque / epsilon;

						q_bond.force = -tmp_force / epsilon;
						q_bond.p_torque = -q_torque / epsilon;
						q_bond.q_torque = -p_torque / epsilon;
				}
				else {
					p_bond.force = LR_vector();
					p_bond.p_torque = -p->orientationT * torque_tetaTerm / Vradial;
					p_bond.q_torque = -q->orientationT * torque_tetaTerm / Vradial;

					q_bond.force = LR_vector();
					q_bond.p_torque = q->orientationT * torque_tetaTerm / Vradial;
					q_bond.q_torque = p->orientationT * torque_tetaTerm / Vradial;
				}


				if(p->strand_id != q->strand_id) {
					_update_inter_chain_stress_tensor(p->strand_id, p->strand_id, tmp_force);
					_update_inter_chain_stress_tensor(q->strand_id, p->strand_id, -tmp_force);
				}

				_update_stress_tensor(p->pos, tmp_force);
				_update_stress_tensor(p->pos + _computed_r, -tmp_force);
			}

			if(!no_three_body) {
				energy += _patchy_three_body(p, p_bond, update_forces);
				energy += _patchy_three_body(q, q_bond, update_forces);

				_bonds[p->index].insert(p_bond);
				_bonds[q->index].insert(q_bond);
			}
		}
	}

	return energy;
}

number CGNucleicAcidsInteraction::_patchy_three_body(BaseParticle *p, PSBond &new_bond, bool update_forces) {
	number energy = 0.;

	number curr_energy = new_bond.energy;
	for(auto &other_bond : _bonds[p->index]) {
		if(other_bond.other != new_bond.other) {
			number other_energy = other_bond.energy;
			number smallest_epsilon = std::min(new_bond.epsilon, other_bond.epsilon);
			number prefactor = _3b_prefactor * smallest_epsilon;

			energy += prefactor * curr_energy * other_energy;

			LR_vector check;
			if(update_forces) {
				// if(new_bond.r_mod > _3b_sigma)
				{
					BaseParticle *other = new_bond.other;

					number factor = -prefactor * other_energy;
					LR_vector tmp_force = factor * new_bond.force;

					p->force += tmp_force;
					other->force -= tmp_force;

					p->torque += factor * new_bond.p_torque;
					other->torque -= factor * new_bond.q_torque;

					check = new_bond.r_part.cross(-tmp_force) + p->orientation * factor * new_bond.p_torque - other->orientation * factor * new_bond.q_torque;

					if(p->strand_id != other->strand_id) {
						_update_inter_chain_stress_tensor(p->strand_id, p->strand_id, tmp_force);
						_update_inter_chain_stress_tensor(other->strand_id, p->strand_id, -tmp_force);
					}

					_update_stress_tensor(p->pos, tmp_force);
					_update_stress_tensor(p->pos + new_bond.r, -tmp_force);
				}

				// if(other_bond.r_mod > _3b_sigma)
				{
					BaseParticle *other = other_bond.other;

					number factor = -prefactor * curr_energy;
					LR_vector tmp_force = factor * other_bond.force;

					p->force += tmp_force;
					other->force -= tmp_force;

					p->torque += factor * other_bond.p_torque;
					other->torque -= factor * other_bond.q_torque;

					check += other_bond.r_part.cross(-tmp_force) + p->orientation * factor * other_bond.p_torque - other->orientation * factor * other_bond.q_torque;

					if(p->strand_id != other->strand_id) {
						_update_inter_chain_stress_tensor(p->strand_id, p->strand_id, tmp_force);
						_update_inter_chain_stress_tensor(other->strand_id, p->strand_id, -tmp_force);
					}


					_update_stress_tensor(p->pos, tmp_force);
					_update_stress_tensor(p->pos + other_bond.r, -tmp_force);
				}
			}
		}
	}
	return energy;
}

number CGNucleicAcidsInteraction::pair_interaction(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
	if(p->is_bonded(q)) {
		return pair_interaction_bonded(p, q, compute_r, update_forces);
	}
	else {
		return pair_interaction_nonbonded(p, q, compute_r, update_forces);
	}
}

//// to align the direction on which the patches of consecutive beads lie
number CGNucleicAcidsInteraction::_patch_stacking(BaseParticle *p, BaseParticle *q, bool update_forces) {
	number cost_a1 = (p->orientationT.v1 * q->orientationT.v1);

	if (update_forces) {
		LR_vector torque_term = _stacking_eta * p->orientationT.v1.cross(q->orientationT.v1);
		p->torque += p->orientationT * torque_term;
		q->torque -= q->orientationT * torque_term;
	}

	return _stacking_eta * (1. - cost_a1);
}
////

number CGNucleicAcidsInteraction::_semiflexibility_two_body(BaseParticle *p, BaseParticle *q, bool update_forces) {
	number cost_a3 = (p->orientationT.v3 * q->orientationT.v3);

	if (update_forces) {
		LR_vector torque_term = _semiflexibility_k * p->orientationT.v3.cross(q->orientationT.v3);
		p->torque += p->orientationT * torque_term;
		q->torque -= q->orientationT * torque_term;
	}

	return _semiflexibility_k * (1. - cost_a3);
}

number CGNucleicAcidsInteraction::_semiflexibility_three_body(BaseParticle *middle, BaseParticle *n1, BaseParticle *n2, bool update_forces) {
	LR_vector dist_pn1 = _box->min_image(middle->pos, n1->pos);
	LR_vector dist_pn2 = _box->min_image(n2->pos, middle->pos);

	number sqr_dist_pn1 = dist_pn1.norm();
	number sqr_dist_pn2 = dist_pn2.norm();
	number i_pn1_pn2 = 1. / sqrt(sqr_dist_pn1 * sqr_dist_pn2);
	number cost = (dist_pn1 * dist_pn2) * i_pn1_pn2;

	if(update_forces) {
		number cost_n1 = cost / sqr_dist_pn1;
		number cost_n2 = cost / sqr_dist_pn2;
		number force_mod_n1 = i_pn1_pn2 + cost_n1;
		number force_mod_n2 = i_pn1_pn2 + cost_n2;

		middle->force += dist_pn1 * (force_mod_n1 * _semiflexibility_3b_k) - dist_pn2 * (force_mod_n2 * _semiflexibility_3b_k);
		n1->force -= dist_pn1 * (cost_n1 * _semiflexibility_3b_k) - dist_pn2 * (i_pn1_pn2 * _semiflexibility_3b_k);
		n2->force -= dist_pn1 * (i_pn1_pn2 * _semiflexibility_3b_k) - dist_pn2 * (cost_n2 * _semiflexibility_3b_k);
	}

	return _semiflexibility_3b_k * (1. - cost);
}


number CGNucleicAcidsInteraction::pair_interaction_bonded(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
	number energy = (number) 0.f;

	if(p->is_bonded(q)) {
		if(compute_r) {
			if(q != P_VIRTUAL && p != P_VIRTUAL) {
				_computed_r = _box->min_image(p->pos, q->pos);
			}
		}

		energy = _fene(p, q, update_forces);
		energy += _WCA(p, q, update_forces);

		if(_enable_semiflexibility) {
			energy += _semiflexibility_two_body(p, q, update_forces);
		}

		if(_enable_patch_stacking) {
			energy += _patch_stacking(p, q, update_forces);
		}

		if(_enable_semiflexibility_3b) {
			// here things get complicated. We first check whether p is the middle particle of a three-body interaction
			for(auto pair : p->affected) {
				// q has always a smaller index than p, so that this condition selects all the other ParticlePair's
				if(pair.second != q) {
					BaseParticle *third_p = (pair.first == p) ? pair.second : pair.first;
					// if third_p's index is smaller than p's then pair_interaction_bonded won't be called with p as its first argument any more, which means
					// that this is the only possibility of computing the three-body interaction with p being the middle particle
					if(third_p->index < p->index) {
						energy += _semiflexibility_three_body(p, q, third_p, update_forces);
					}
					// otherwise p will be passed as the first argument one more time, which means we have to make sure that the (p, q, third_p) three-body
					// interaction is called only once. We do so by adding a condition on the relation between q's and third_p's indexes
					else if(third_p->index < q->index) {
						energy += _semiflexibility_three_body(p, q, third_p, update_forces);
					}

				}
			}
			// we have another possibility: q is the middle particle of a three-body interaction, but its index is such that it will never be passed as the
			// first argument of this method. If this is the case then we need to make sure that its interaction is computed
			for(auto pair : q->affected) {
				// we don't consider the pair formed by q and p
				if(pair.first != p) {
					BaseParticle *third_p = (pair.first == q) ? pair.second : pair.first;
					// if this condition is true then q won't be passed as first argument for this pair, which means that we may have to compute the three body
					// interaction involving q, p and third_p here
					if(third_p->index < q->index) {
						// now we have to choose whether this three-body interaction gets computed as (p, q, third_p) or as (third_p, q, p)
						if(third_p->index < p->index) {
							energy += _semiflexibility_three_body(q, p, third_p, update_forces);
						}
					}
				}
			}
		}
	}

	return energy;
}

bool CGNucleicAcidsInteraction::_sticky_interaction(int p_btype, int q_btype) {
	return _3b_epsilon[p_btype * _interaction_matrix_size + q_btype] != 0.;
}

number CGNucleicAcidsInteraction::pair_interaction_nonbonded(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
	if(p->is_bonded(q)) {
		return (number) 0.f;
	}

	if(compute_r) {
		_computed_r = _box->min_image(p->pos, q->pos);
	}

	number energy = pair_nonbonded_WCA(p, q, false, update_forces);
	energy += pair_nonbonded_sticky(p, q, false, update_forces);

	return energy;
}

number CGNucleicAcidsInteraction::pair_nonbonded_WCA(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
	if(p->is_bonded(q)) {
		return (number) 0.f;
	}

	if(compute_r) {
		_computed_r = _box->min_image(p->pos, q->pos);
	}

	number energy = _WCA(p, q, update_forces);

	return energy;
}	// switch(N_int_centers()) {
	// case 0:
	// 	break;
	// case 1: {
	// 	_base_patches[0] = LR_vector(1, 0, 0);
	// 	break;
	// }
  // default:
	// 	throw oxDNAException("Unsupported number of patches %d\n", N_int_centers());
	// }
  // for(uint i = 0; i < N_int_centers(); i++) {
  //   _base_patches[i].normalize();
  //   _base_patches[i] *= deltaPM;
  // }

number CGNucleicAcidsInteraction::pair_nonbonded_sticky(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
	if(p->is_bonded(q)) {
		return (number) 0.f;
	}

	if(compute_r) {
		_computed_r = _box->min_image(p->pos, q->pos);
	}

	number energy = 0.;
	if(_sticky_interaction(p->btype, q->btype)) {
		energy += _sticky(p, q, update_forces);
	}

	return energy;
}

void CGNucleicAcidsInteraction::check_input_sanity(std::vector<BaseParticle*> &particles) {
	for(auto p : particles) {
		if(_enable_semiflexibility && p->affected.size() > 2) {
			throw oxDNAException("CGNucleicAcidsInteraction: PS_enable_semiflexibility can be set to true only if no particle has more than 2 bonded neighbours");
		}
	}
}

void CGNucleicAcidsInteraction::allocate_particles(std::vector<BaseParticle*> &particles) {
	for(uint i = 0; i < particles.size(); i++) {
		particles[i] = new ParticleFTG(1,0,1,_deltaPatchMon);
	}
}

void CGNucleicAcidsInteraction::_parse_interaction_matrix() {
        // parse the interaction matrix file
        input_file inter_matrix_file;
        inter_matrix_file.init_from_filename(_interaction_matrix_file);
        const number _t37_ = 310.15;
        const number _kB_ = 1.9872036;
        //const number dS_mod = 1.87;
        if(inter_matrix_file.state == ERROR) {
                throw oxDNAException("Caught an error while opening the interaction matrix file '%s'", _interaction_matrix_file.c_str());
        }

        _interaction_matrix_size = _N_attractive_types + 1;
        _3b_epsilon.resize(_interaction_matrix_size * _interaction_matrix_size, 0.);

        ofstream myfile;
        myfile.open ("beta_eps_matrix.dat");
        for(int i = 1; i <= _N_attractive_types; i++) {
                for(int j = 1; j <= _N_attractive_types; j++) {
                        number valueH;
                        number valueS;
                        std::string keyH = Utils::sformat("dH[%d][%d]", i, j);
                        std::string keyS = Utils::sformat("dS[%d][%d]", i, j);
                        if(getInputNumber(&inter_matrix_file, keyH.c_str(), &valueH, 0) == KEY_FOUND && getInputNumber(&inter_matrix_file, keyS.c_str(), &valueS, 0) == KEY_FOUND) {
                                number beta_dG = (_mu * valueH * 1000 / _t37_ - valueS)/_kB_;
                                number beta_eps = -(beta_dG + dS_mod) / alpha_mod;
                                if(beta_dG<bdG_threshold && abs(i-j)>2) {
                                        _3b_epsilon[i + _interaction_matrix_size * j] = _3b_epsilon[j + _interaction_matrix_size * i] = beta_eps;
                                        myfile << "beta_eps[" << i << "][" << j << "]=" << beta_eps << "\n";
                                }
                        }
                }
        }
        myfile.close();
}

void CGNucleicAcidsInteraction::read_topology(int *N_strands, std::vector<BaseParticle*> &particles) {
	std::string line;
	unsigned int N_from_conf = particles.size();
	BaseInteraction::read_topology(N_strands, particles);

	std::ifstream topology;
	topology.open(_topology_filename, std::ios::in);
	if(!topology.good()) {
		throw oxDNAException("Can't read topology file '%s'. Aborting", _topology_filename);
	}
	unsigned int N_from_topology;
	topology >> N_from_topology >> *N_strands;

	if(N_from_conf != N_from_topology) {
		throw oxDNAException("The number of particles found in the configuration file (%u) and specified in the topology (%u) are different", N_from_conf, N_from_topology);
	}

	for(unsigned int i = 0; i < N_from_conf; i++) {
		unsigned int p_idx;
		topology >> p_idx;

		if(!topology.good()) {
			throw oxDNAException("The topology should contain two lines per particle, but it seems there is info for only %d particles\n", i);
		}

		ParticleFTG *p = static_cast<ParticleFTG*>(particles[p_idx]);
		topology >> p->btype;
		p->type = (p->btype > 0) ? STICKY_ANY : MONOMER;
		if(p->btype > _N_attractive_types) {
			_N_attractive_types = p->btype;
		}

		int n_bonds;
		topology >> n_bonds;

		if(i != p_idx) {
			throw oxDNAException("There is something wrong with the topology file. Expected index %d, found %d\n", i, p_idx);
		}

		if(p_idx >= N_from_conf) {
			throw oxDNAException("There is a mismatch between the configuration and topology files: the latter refers to particle %d, which is larger than the largest possible index (%d)", p_idx,
					N_from_conf - 1);
		}

		p->strand_id = p_idx / _chain_size;
		p->n3 = p->n5 = P_VIRTUAL;
		for(int j = 0; j < n_bonds; j++) {
			unsigned int n_idx;
			topology >> n_idx;
			// the >> operator always leaves '\n', which is subsequently picked up by getline if we don't explicitly ignore it
			topology.ignore();

			if(n_idx >= N_from_conf) {
				throw oxDNAException("The topology file contains a link between particles {} and {}, but the largest possible index in the configuration is %d", p->index, n_idx, N_from_conf - 1);
			}

			ParticleFTG *q = static_cast<ParticleFTG*>(particles[n_idx]);
			p->add_bonded_neigh(q);
		}
	}

	topology.close();

	*N_strands = N_from_conf / _chain_size;
	if(*N_strands == 0) {
		*N_strands = 1;
	}
	_N_chains = *N_strands;

	_parse_interaction_matrix();

	return;
}

extern "C" CGNucleicAcidsInteraction* make_CGNucleicAcidsInteraction() {
	return new CGNucleicAcidsInteraction();
}
