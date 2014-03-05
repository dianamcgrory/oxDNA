/**
 * @file    Utils.h
 * @date    04/set/2010
 * @author  lorenzo
 *
 *
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <algorithm>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <cctype>
#include <string>
#include <cstdarg>
#include <cstdio>
#include <vector>

#include "../defs.h"

extern "C" {
#include "../Utilities/parse_input/parse_input.h"
}

/**
 * @brief Utility class. It mostly contains static methods.
 */
class Utils {
public:
	Utils();
	virtual ~Utils();

	static int decode_base(char c);
	static char encode_base(int b);

	template<typename number> static number gaussian();
	template<typename number> static number sum(number *v, int N) {
		number res = (number) 0.;
		for(int i = 0; i < N; i++) res += v[i];
		return res;
	}

	/**
	 * @brief split a string into tokens, according to the given delimiter
	 *
	 * @param s string to be splitted
	 * @param delim delimiter, defaults to a space
	 * @return a vector of strings containing all the tokens
	 */
	static std::vector<std::string> split(const std::string &s, char delim=' ');

	// trim from both ends, it works like Python's own trim
	static inline std::string &trim(std::string &s) {
			return ltrim(rtrim(s));
	}

	// trim from start
	static inline std::string &ltrim(std::string &s) {
			s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
			return s;
	}

	// trim from end
	static inline std::string &rtrim(std::string &s) {
			s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
			return s;
	}

	/**
	 * @brief sprintf c++ wrapper (I love c++...).
	 *
	 * @param fmt
	 * @return
	 */
	static std::string sformat(const std::string &fmt, ...);
	/**
	 * @brief vsprintf c++ wrapper.
	 *
	 * This method can be called by other variadic. It is used, for example, by oxDNAException
	 * @param fmt format string
	 * @param ap variadic parameter list initialized by the caller
	 * @return
	 */
	static std::string sformat_ap(const std::string &fmt, va_list &ap);

	template<typename number> static LR_vector<number> get_random_vector();
	template<typename number> static LR_vector<number> get_random_vector_in_sphere(number r);
	template<typename number> static void orthonormalize_matrix(LR_matrix<number> &M);
	template<typename number> static LR_matrix<number> get_random_rotation_matrix(number max_angle=2*M_PI);
	template<typename number> static LR_matrix<number> get_random_rotation_matrix_from_angle (number angle);

	/**
	 * @brief Creates a temporary file and loads it in an input_file
	 *
	 * If the string parameter starts with a curly opening bracket, this method will print in the temporary file
	 * only the part of the string which is enclosed by the outer bracket pair.
	 * The calling method is responsible for the calling of cleanInputFile and the deletion of the returned pointer
	 * @param inp string to be written in the temporary file and then loaded in the input_file
	 * @return pointer to the newly loaded input_file
	 */
	static input_file *get_input_file_from_string(const std::string &inp);

	template<typename number>
	static number get_temperature(char * raw_T);

	/**
	 * @brief fills the memory pointed to by seedptr with the current
	 * state of the random number generator.
	 *
	 * This method does not handle the memory: it assumes that it can overwrite the first 48 bit of the
	 * memory.
	 *
	 * @param seedptr the memory address to store the 48 bits of the
	 * seed into.
	 */
	static void get_seed(unsigned short * seedptr);

};

template<typename number>
inline LR_vector<number> Utils::get_random_vector() {
	number ransq = 1.;
	number ran1, ran2;

	while(ransq >= 1) {
		ran1 = 1. - 2. * drand48();
		ran2 = 1. - 2. * drand48();
		ransq = ran1*ran1 + ran2*ran2;
	}

	number ranh = 2. * sqrt(1. - ransq);
	return LR_vector<number>(ran1*ranh, ran2*ranh, 1. - 2. * ransq);
}

template<typename number>
inline LR_vector<number> Utils::get_random_vector_in_sphere(number r) {
	number r2 = SQR(r);
	LR_vector<number> res = LR_vector<number>(r, r, r);

	while(res.norm() > r2) {
		res = LR_vector<number>(2 * r * (drand48() - 0.5), 2 * r * (drand48() - 0.5), 2 * r * (drand48() - 0.5));
	}

	return res;
}

template<typename number>
void Utils::orthonormalize_matrix(LR_matrix<number> &m) {
    number v1_norm2 = m.v1 * m.v1;
    number v2_v1 = m.v2 * m.v1;

    m.v2 -= (v2_v1/v1_norm2) * m.v1;

    number v3_v1 = m.v3 * m.v1;
    number v3_v2 = m.v3 * m.v2;
    number v2_norm2 = m.v2 * m.v2;

    m.v3 -= (v3_v1/v1_norm2) * m.v1 + (v3_v2/v2_norm2) * m.v2;

    m.v1.normalize();
    m.v2.normalize();
    m.v3.normalize();
}

template<typename number>
LR_matrix<number> Utils::get_random_rotation_matrix_from_angle (number angle) {
	LR_vector<number> axis = Utils::get_random_vector<number>();

	number t = angle;
	number sintheta = sin(t);
	number costheta = cos(t);
	number olcos = 1. - costheta;

	number xyo = axis.x * axis.y * olcos;
	number xzo = axis.x * axis.z * olcos;
	number yzo = axis.y * axis.z * olcos;
	number xsin = axis.x * sintheta;
	number ysin = axis.y * sintheta;
	number zsin = axis.z * sintheta;

	LR_matrix<number> R(axis.x * axis.x * olcos + costheta, xyo - zsin, xzo + ysin,
				xyo + zsin, axis.y * axis.y * olcos + costheta, yzo - xsin,
				xzo - ysin, yzo + xsin, axis.z * axis.z * olcos + costheta);

	return R;
}


template<typename number>
LR_matrix<number> Utils::get_random_rotation_matrix(number max_angle) {
	number t = max_angle * (drand48() - 0.5);
	return get_random_rotation_matrix_from_angle (t);
}

template<typename number>
inline number Utils::gaussian() {
	static unsigned int isNextG = 0;
	static number nextG;
	number toRet;
	number u, v, w;

	if(isNextG) {
		isNextG = 0;
		return nextG;
	}

	w = 2.;
	while(w >= 1.0) {
		u = 2. * drand48() - 1.0;
		v = 2. * drand48() - 1.0;
		w = u*u + v*v;
	}

	w = sqrt((-2. * log(w)) / w);
	toRet = u * w;
	nextG = v * w;
	isNextG = 1;

	return toRet;
}

#endif /* UTILS_H_ */
