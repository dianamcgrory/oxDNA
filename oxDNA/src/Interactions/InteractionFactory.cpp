/*
 * InteractionFactory.cpp
 *
 *  Created on: Feb 13, 2013
 *      Author: rovigatti
 */

#include "InteractionFactory.h"
#include "DNAInteraction.h"
#include "DNAInteraction_nomesh.h"
#include "RNAInteraction.h"
#include "LJInteraction.h"
#include "PatchyInteraction.h"
#include "DNAInteraction_relax.h"
#include "TSPInteraction.h"
#include "HSInteraction.h"
#include "DHSInteraction.h"
#include "BoxInteraction.h"
#include "HardCylinderInteraction.h"
#include "HardSpheroCylinderInteraction.h"
#include "DirkInteraction.h"
#include "DirkInteraction2.h"

InteractionFactory::InteractionFactory() {

}

InteractionFactory::~InteractionFactory() {

}

template<typename number>
IBaseInteraction<number> *InteractionFactory::make_interaction(input_file &inp) {
	// The default interaction is DNAInteraction
	char inter_type[512] = "DNA";
	getInputString(&inp, "interaction_type", inter_type, 0);

	if(!strncmp(inter_type, "DNA", 512)) {
		// in order to avoid small mismatches between potential energies computed on the GPU and 
		// on the CPU we enforce the DNAInteraction_nomesh interaction class
		char backend[512];
		getInputString(&inp, "backend", backend, 0);
		if(!strncmp(backend, "CUDA", 512)) return new DNAInteraction_nomesh<number>();
		else return new DNAInteraction<number>();
	}
	else if(!strncmp(inter_type, "DNA_nomesh", 512)) return new DNAInteraction_nomesh<number>();
	else if(!strncmp(inter_type, "LJ", 512)) return new LJInteraction<number>();
	else if(!strncmp(inter_type, "DNA_relax", 512)) return new DNAInteraction_relax<number>();
	else if(!strncmp(inter_type, "RNA", 512)) return new RNAInteraction<number>();
	else if(!strncmp(inter_type, "patchy", 512)) return new PatchyInteraction<number>();
	else if(!strncmp(inter_type, "TSP", 512)) return new TSPInteraction<number>();
	else if(!strncmp(inter_type, "HS", 512)) return new HSInteraction<number>();
	else if(!strncmp(inter_type, "Box", 512)) return new BoxInteraction<number>();
	else if(!strncmp(inter_type, "HardCylinder", 512)) return new HardCylinderInteraction<number>();
	else if(!strncmp(inter_type, "HardSpheroCylinder", 512)) return new HardSpheroCylinderInteraction<number>();
	else if(!strncmp(inter_type, "DHS", 512)) return new DHSInteraction<number>();
	else if(!strncmp(inter_type, "Dirk", 512)) return new DirkInteraction<number>();
	else if(!strncmp(inter_type, "Dirk2", 512)) return new DirkInteraction2<number>();
	else throw oxDNAException("Invalid interaction '%s'", inter_type);
}

template IBaseInteraction<float> *InteractionFactory::make_interaction<float>(input_file &inp);
template IBaseInteraction<double> *InteractionFactory::make_interaction<double>(input_file &inp);
