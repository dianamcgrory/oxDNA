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

    __host__ __device__ ox_number3() : x(0), y(0), z(0) {

    }

    __host__ __device__ ox_number3(T nx, T ny, T nz) : x(nx), y(ny), z(nz) {

    }

    __host__ __device__ ox_number3(const float4 &from) : x(from.x), y(from.y), z(from.z) {

    }

    __host__ __device__ ox_number3(const double4 &from) : x(from.x), y(from.y), z(from.z) {

    }

    __host__ __device__ operator float4() const {
        return make_float4(x, y, z, 0.f);
    }

    __host__ __device__ operator double4() const {
        return make_double4(x, y, z, 0.);
    }

    __host__ __device__ ox_number3 operator+(const ox_number3<T> &rhs) const {
        return ox_number3(x + rhs.x, y + rhs.y, z + rhs.z);
    }

    __host__ __device__ ox_number3 operator-(const ox_number3<T> &rhs) const {
        return ox_number3(x - rhs.x, y - rhs.y, z - rhs.z);
    }

    __host__ __device__ ox_number3 operator+(const float4 &rhs) const {
        return ox_number3(x + rhs.x, y + rhs.y, z + rhs.z);
    }

    __host__ __device__ ox_number3 operator-(const float4 &rhs) const {
        return ox_number3(x - rhs.x, y - rhs.y, z - rhs.z);
    }

    __host__ __device__ ox_number3 operator/(T rhs) const {
        T inv = 1.f / rhs;
        return ox_number3(x * inv, y * inv, z * inv);
    }

    __host__ __device__ ox_number3 operator*(T rhs) const {
        return ox_number3(x * rhs, y * rhs, z * rhs);
    }

    __host__ __device__ T operator*(const ox_number3<T> &rhs) const {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }

    __host__ __device__ ox_number3 &operator-=(const ox_number3 &rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    __host__ __device__ ox_number3 &operator+=(const ox_number3 &rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    __host__ __device__ ox_number3 &operator*=(T rhs) {
        x *= rhs;
        y *= rhs;
        z *= rhs;
        return *this;
    }

    __host__ __device__ ox_number3 &operator/=(T rhs) {
        T inv = 1.f / rhs;
        x *= inv;
        y *= inv;
        z *= inv;
        return *this;
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
__forceinline__ __host__ __device__ ox_number3<T> operator*(float c, const ox_number3<T> &v) {
	return ox_number3<T>(v.x * c, v.y * c, v.z * c);
}

template<typename T>
__forceinline__ __device__ void operator+=(c_number4 &a, const ox_number3<T> &b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

template<typename T>
__forceinline__ __device__ void operator-=(c_number4 &a, const ox_number3<T> &b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

template<typename vector>
__forceinline__ __device__ vector _cross(const vector &v, const vector &w) {
	vector res;
	res.x = v.y * w.z - v.z * w.y;
	res.y = v.z * w.x - v.x * w.z;
	res.z = v.x * w.y - v.y * w.x;
	return res;
}

template<typename T>
__forceinline__ __device__ ox_number3<T> _cross(const ox_number3<T> &v, const float4 &w) {
	ox_number3<T> res;
	res.x = v.y * w.z - v.z * w.y;
	res.y = v.z * w.x - v.x * w.z;
	res.z = v.x * w.y - v.y * w.x;
	return res;
}

template<typename vector>
__forceinline__ __device__ c_number _module(const vector &v) {
	return sqrtf(SQR(v.x) + SQR(v.y) + SQR(v.z));
}

template<typename vector>
__forceinline__ __device__ vector stably_normalised(const vector &v) {
	c_number max = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fabsf(v.z));
	vector res = v / max;
	return res / _module(res);
}

#endif
