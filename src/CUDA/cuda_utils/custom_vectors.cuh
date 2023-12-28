#ifndef CUSTOM_VECTORS_CUH
#define CUSTOM_VECTORS_CUH

#include <cuda.h>

/**
 * @brief We need this struct because the fourth element of such a structure must be a float or _float_as_int will not work.
 */
typedef struct
__align__(16) {
	double x, y, z;
	float w;
} LR_double4;

template<typename T>
struct ox_number3 {
    T x, y, z;

    inline __host__ __device__ ox_number3() : x(0), y(0), z(0) {

    }

    inline __host__ __device__ ox_number3(T nx, T ny, T nz) : x(nx), y(ny), z(nz) {

    }

    __host__ __device__ ox_number3(const float4 &from) : x(from.x), y(from.y), z(from.z) {

    }

    inline __host__ __device__ ox_number3(const double4 &from) : x(from.x), y(from.y), z(from.z) {

    }

    inline __host__ __device__ operator float4() const {
        return make_float4(x, y, z, 0.f);
    }

    inline __host__ __device__ operator double4() const {
        return make_double4(x, y, z, 0.);
    }

};

#ifdef CUDA_DOUBLE_PRECISION
using c_number4 = LR_double4;
using c_number = double;
using GPU_quat = double4;
#else
using c_number4 = float4;
using c_number3 = ox_number3<float>;
using c_number = float;
using GPU_quat = float4;
#endif

template<typename T>
inline __host__ __device__ ox_number3<T> operator*(float c, ox_number3<T> v) {
	return ox_number3<T>(v.x * c, v.y * c, v.z * c);
}

template<typename T>
inline __host__ __device__ ox_number3<T> operator*(ox_number3<T> lhs, T rhs) {
    return ox_number3<T>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}

template<typename T>
inline __host__ __device__ ox_number3<T> operator/(ox_number3<T> lhs, T rhs) {
    T inv = 1.f / rhs;
    return lhs * inv;
}

template<typename T>
inline __host__ __device__ ox_number3<T> operator+(ox_number3<T> lhs, ox_number3<T> rhs) {
    return ox_number3<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

template<typename T>
inline __host__ __device__ ox_number3<T> operator-(ox_number3<T> lhs, ox_number3<T> rhs) {
    return ox_number3<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

template<typename T>
inline __host__ __device__ ox_number3<T> operator+(ox_number3<T> lhs, float4 rhs) {
    return ox_number3<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

template<typename T>
inline __host__ __device__ ox_number3<T> operator-(ox_number3<T> lhs, float4 rhs) {
    return ox_number3<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

template<typename T>
inline __host__ __device__ void operator-=(ox_number3<T> &lhs, ox_number3<T> rhs) {
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
}

template<typename T>
inline __host__ __device__ void operator+=(ox_number3<T> &lhs,ox_number3<T> rhs) {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
}

template<typename T>
inline __host__ __device__ void operator*=(ox_number3<T> &lhs,T rhs) {
    lhs.x *= rhs;
    lhs.y *= rhs;
    lhs.z *= rhs;
}

template<typename T>
inline __host__ __device__ void operator/=(ox_number3<T> &lhs,T rhs) {
    T inv = 1.f / rhs;
    lhs *= inv;
}

template<typename T>
inline __device__ void operator+=(c_number4 &a, ox_number3<T> b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

template<typename T>
inline __device__ void operator-=(c_number4 &a, ox_number3<T> b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

template<typename vector>
inline __device__ vector _cross(vector v, vector w) {
	vector res;
	res.x = v.y * w.z - v.z * w.y;
	res.y = v.z * w.x - v.x * w.z;
	res.z = v.x * w.y - v.y * w.x;
	return res;
}

template<typename T>
inline __device__ ox_number3<T> _cross(ox_number3<T> v, float4 w) {
	ox_number3<T> res;
	res.x = v.y * w.z - v.z * w.y;
	res.y = v.z * w.x - v.x * w.z;
	res.z = v.x * w.y - v.y * w.x;
	return res;
}

template<typename vector>
inline __device__ c_number _module(vector v) {
	return sqrtf(CUDA_DOT(v, v));
}

template<typename vector>
inline __device__ vector stably_normalised(vector v) {
	c_number max = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fabsf(v.z));
	vector res = v / max;
	return res / _module(res);
}

#endif
