// includes, cuda
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <vector_types.h>

#ifdef __INTELLISENSE__
#define ADD_INTELLISENSE(fname, ...) __VA_ARGS__
#else
#define ADD_INTELLISENSE(fname, ...) fname(__VA_ARGS__)
#endif

#if defined(__CUDACC_RTC__)
#define __VECTOR_FUNCTIONS_DECL__ __host__ __device__
#else /* !__CUDACC_RTC__ */
#define __VECTOR_FUNCTIONS_DECL__ static __inline__ __host__ __device__
#endif /* __CUDACC_RTC__ */

//#define sqrtf ADD_INTELLISENSE(sqrtf)
//#define rsqrtf ADD_INTELLISENSE(rsqrtf)
//#define sinf ADD_INTELLISENSE(sinf)
//#define cosf ADD_INTELLISENSE(cosf)
//#define tanf ADD_INTELLISENSE(tanf)
//#define asinf ADD_INTELLISENSE(asinf)
//#define acosf ADD_INTELLISENSE(acosf)
//#define atanf ADD_INTELLISENSE(atanf)
//#define atan2f ADD_INTELLISENSE(atan2f)
//#define expf ADD_INTELLISENSE(expf)

// basic
__VECTOR_FUNCTIONS_DECL__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__VECTOR_FUNCTIONS_DECL__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__VECTOR_FUNCTIONS_DECL__ float3 operator*(float3 a, float3 b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__VECTOR_FUNCTIONS_DECL__ float3 operator/(float3 a, float3 b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__VECTOR_FUNCTIONS_DECL__ float4 operator+(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__VECTOR_FUNCTIONS_DECL__ float4 operator-(float4 a, float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__VECTOR_FUNCTIONS_DECL__ float4 operator*(float4 a, float4 b)
{
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__VECTOR_FUNCTIONS_DECL__ float4 operator/(float4 a, float4 b)
{
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

// scalar operations
__VECTOR_FUNCTIONS_DECL__ float3 operator*(float3 v, float s)
{
	return make_float3(v.x * s, v.y * s, v.z * s);
}

__VECTOR_FUNCTIONS_DECL__ float3 operator*(float s, float3 v)
{
	return make_float3(v.x * s, v.y * s, v.z * s);
}

__VECTOR_FUNCTIONS_DECL__ float3 operator/(float3 v, float s)
{
	return make_float3(v.x / s, v.y / s, v.z / s);
}

__VECTOR_FUNCTIONS_DECL__ float3 operator/(float s, float3 v)
{
	return make_float3(s / v.x, s / v.y, s / v.z);
}

__VECTOR_FUNCTIONS_DECL__ float4 operator*(float4 v, float s)
{
	return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

__VECTOR_FUNCTIONS_DECL__ float4 operator*(float s, float4 v)
{
	return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

__VECTOR_FUNCTIONS_DECL__ float4 operator/(float4 v, float s)
{
	return make_float4(v.x / s, v.y / s, v.z / s, v.w / s);
}

__VECTOR_FUNCTIONS_DECL__ float4 operator/(float s, float4 v)
{
	return make_float4(s / v.x, s / v.y, s / v.z, s / v.w);
}


// dot product
__VECTOR_FUNCTIONS_DECL__ constexpr float dot(float4 a, float4 b) noexcept
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__VECTOR_FUNCTIONS_DECL__ constexpr float dot(float3 a, float3 b) noexcept
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}


// cross product
__VECTOR_FUNCTIONS_DECL__ float3 cross(float3 a, float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__VECTOR_FUNCTIONS_DECL__ float3 cross(float4 a, float4 b) // ignores w component
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}


// length
__VECTOR_FUNCTIONS_DECL__ float length(float3 v) noexcept
{
	return sqrtf(dot(v, v));
}

__VECTOR_FUNCTIONS_DECL__ float length(float4 v) noexcept
{
	return sqrtf(dot(v, v));
}


/* normalize Can't implement here for some stupid reason
__VECTOR_FUNCTIONS_DECL__ float3 normalize(float3 v)
{
	//float invLen = rsqrtf(dot(v, v));
	return v * rnorm3df(v.x, v.y, v.z);
}

__VECTOR_FUNCTIONS_DECL__ float4 normalize(float4 v)
{
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}
*/

// trig
__VECTOR_FUNCTIONS_DECL__ float3 sin(float3 v)
{
	return make_float3(sinf(v.x), sinf(v.y), sinf(v.z));
}

__VECTOR_FUNCTIONS_DECL__ float3 cos(float3 v)
{
	return make_float3(cosf(v.x), cosf(v.y), cosf(v.z));
}

__VECTOR_FUNCTIONS_DECL__ float3 tan(float3 v)
{
	return make_float3(tanf(v.x), tanf(v.y), tanf(v.z));
}

__VECTOR_FUNCTIONS_DECL__ float3 asin(float3 v)
{
	return make_float3(asinf(v.x), asinf(v.y), asinf(v.z));
}

__VECTOR_FUNCTIONS_DECL__ float3 acos(float3 v)
{
	return make_float3(acosf(v.x), acosf(v.y), acosf(v.z));
}

__VECTOR_FUNCTIONS_DECL__ float3 atan(float3 v)
{
	return make_float3(atanf(v.x), atanf(v.y), atanf(v.z));
}

__VECTOR_FUNCTIONS_DECL__ float3 atan2(float3 a, float3 b)
{
	return make_float3(atan2f(a.x, b.x), atan2f(a.y, b.y), atan2f(a.z, b.z));
}

__VECTOR_FUNCTIONS_DECL__ float4 sin(float4 v)
{
	return make_float4(sinf(v.x), sinf(v.y), sinf(v.z), sinf(v.w));
}

__VECTOR_FUNCTIONS_DECL__ float4 cos(float4 v)
{
	return make_float4(cosf(v.x), cosf(v.y), cosf(v.z), cosf(v.w));
}

__VECTOR_FUNCTIONS_DECL__ float4 tan(float4 v)
{
	return make_float4(tanf(v.x), tanf(v.y), tanf(v.z), tanf(v.w));
}

__VECTOR_FUNCTIONS_DECL__ float4 asin(float4 v)
{
	return make_float4(asinf(v.x), asinf(v.y), asinf(v.z), asinf(v.w));
}

__VECTOR_FUNCTIONS_DECL__ float4 acos(float4 v)
{
	return make_float4(acosf(v.x), acosf(v.y), acosf(v.z), acosf(v.w));
}

__VECTOR_FUNCTIONS_DECL__ float4 atan(float4 v)
{
	return make_float4(atanf(v.x), atanf(v.y), atanf(v.z), atanf(v.w));
}

__VECTOR_FUNCTIONS_DECL__ float4 atan2(float4 a, float4 b)
{
	return make_float4(atan2f(a.x, b.x), atan2f(a.y, b.y), atan2f(a.z, b.z), atan2f(a.w, b.w));
}

__VECTOR_FUNCTIONS_DECL__ float3 convert(float4 v)
{
	return make_float3(v.x, v.y, v.z);
}