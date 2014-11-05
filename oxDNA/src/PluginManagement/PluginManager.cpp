/*
 * PluginManager.cpp
 *
 *  Created on: 09/dic/2013
 *      Author: lorenzo
 */

#include <dlfcn.h>

#include "PluginManager.h"

#include "../Utilities/Logger.h"
#include "../Utilities/Utils.h"
#include "../Utilities/oxDNAException.h"

using std::string;
using std::vector;

PluginManager *PluginManager::_manager = NULL;

typedef BaseObservable<float>* make_float_obs();
typedef BaseObservable<double>* make_double_obs();

typedef IBaseInteraction<float>* make_float_inter();
typedef IBaseInteraction<double>* make_double_inter();

PluginManager::PluginManager() {
	_path.push_back(string("."));
	_initialised = false;

	// these are the default entry point names
	_obs_entry_points.push_back(string("make_"));
	_obs_entry_points.push_back(string("make_observable_"));

	_inter_entry_points.push_back(string("make_"));
	_inter_entry_points.push_back(string("make_interaction_"));
}

PluginManager::~PluginManager() {

}

void PluginManager::init(input_file &sim_inp) {
	_initialised = true;

	// load the plugin path from the input file
	string ppath;
	if(getInputString(&sim_inp, "plugin_search_path", ppath, 0) == KEY_FOUND) {
		vector<string> paths = Utils::split(ppath, ':');
		for(vector<string>::iterator it = paths.begin(); it < paths.end(); it++) add_to_path(*it);
	}

	string entries;
	// for observables
	if(getInputString(&sim_inp, "plugin_observable_entry_points", entries, 0) == KEY_FOUND) {
		vector<string> v_entries = Utils::split(entries, ':');
		for(vector<string>::iterator it = v_entries.begin(); it < v_entries.end(); it++) _obs_entry_points.push_back(*it);
	}
	// for interactions
	if(getInputString(&sim_inp, "plugin_interaction_entry_points", entries, 0) == KEY_FOUND) {
		vector<string> v_entries = Utils::split(entries, ':');
		for(vector<string>::iterator it = v_entries.begin(); it < v_entries.end(); it++) _inter_entry_points.push_back(*it);
	}
}

void PluginManager::add_to_path(string s) {
	_path.push_back(s);
}

void *PluginManager::_get_handle(string &name) {
	if(!_initialised) throw oxDNAException("PluginManager not initialised, aborting");

	void *handle = NULL;
	for(vector<string>::iterator it = _path.begin(); it != _path.end() && handle == NULL; it++) {
		string path = *it + "/" + name + ".so";
		OX_DEBUG("Looking for plugin '%s' in '%s'", name.c_str(), it->c_str());
		handle = dlopen(path.c_str(), RTLD_LAZY);
	}
	const char *dl_error = dlerror();
	if(dl_error != NULL) throw oxDNAException("Caught an error while opening shared library '%s.so': %s", name.c_str(), dl_error);

	_handles.push(handle);

	return handle;
}

void *PluginManager::_get_entry_point(void *handle, vector<string> &entry_points, string suffix) {
	void *res = NULL;

	bool found = false;
	for(vector<string>::iterator it = entry_points.begin(); it != entry_points.end() && !found; it++) {
		string to_try = *it + suffix;
		res = dlsym(handle, to_try.c_str());
		const char *dlsym_error = dlerror();
		found = (!dlsym_error);
	}

	return res;
}

template<typename number>
BaseObservable<number> *PluginManager::get_observable(string name) {
	void *handle = _get_handle(name);

	// we do this c-like because dynamic linking can be done only in c and thus
	// we have no way of using templates
	void *temp_obs;
	bool found = false;
	// choose between float and double
	if(sizeof(number) == 4) {
		make_float_obs *make_obs = (make_float_obs *) _get_entry_point(handle, _obs_entry_points, string("float"));
		if(make_obs != NULL) {
			temp_obs = (void *)make_obs();
			found = true;
		}
	}
	else {
		make_double_obs *make_obs = (make_double_obs *) _get_entry_point(handle, _obs_entry_points, string("double"));
		if(make_obs != NULL) {
			temp_obs = (void *)make_obs();
			found = true;
		}
	}

	if(!found) {
		OX_LOG(Logger::LOG_WARNING, "Cannot load symbol from plugin observable library '%s'", name.c_str());
		return NULL;
	}

	// now we cast it back to the type required by the code
	return static_cast<BaseObservable<number> *>(temp_obs);
}

template<typename number>
IBaseInteraction<number> *PluginManager::get_interaction(string name) {
	void *handle = _get_handle(name);

	// we do this c-like because dynamic linking can be done only in c and thus
	// we have no way of using templates
	void *temp_inter;
	bool found = false;
	// choose between float and double
	if(sizeof(number) == 4) {
		make_float_inter *make_inter = (make_float_inter *) _get_entry_point(handle, _inter_entry_points, string("float"));
		if(make_inter != NULL) {
			temp_inter = (void *)make_inter();
			found = true;
		}
	}
	else {
		make_double_inter *make_inter = (make_double_inter *) _get_entry_point(handle, _inter_entry_points, string("double"));
		if(make_inter != NULL) {
			temp_inter = (void *)make_inter();
			found = true;
		}
	}

	if(!found) {
		OX_LOG(Logger::LOG_WARNING, "Cannot load symbol from plugin interaction library '%s'", name.c_str());
		return NULL;
	}

	// now we cast it back to the type required by the code
	return static_cast<IBaseInteraction<number> *>(temp_inter);
}

void PluginManager::clear() {
	if(_manager != NULL) {
		while(!_manager->_handles.empty()) {
			dlclose(_manager->_handles.top());
			_manager->_handles.pop();
		}
		delete _manager;
	}
}

PluginManager *PluginManager::instance() {
	if(_manager == NULL) _manager = new PluginManager();

	return _manager;
}

template BaseObservable<float> *PluginManager::get_observable(string name);
template BaseObservable<double> *PluginManager::get_observable(string name);

template IBaseInteraction<float> *PluginManager::get_interaction(string name);
template IBaseInteraction<double> *PluginManager::get_interaction(string name);
