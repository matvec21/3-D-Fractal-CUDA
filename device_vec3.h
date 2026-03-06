#pragma once

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct vec3
{
public:
	float x;
	float y;
	float z;

	__device__ vec3() {};

	__device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}

	__device__ float length()
	{
		return sqrt(x * x + y * y + z * z);
	}

	__device__ vec3 normalized()
	{
		return vec3(x, y, z) / length();
	}

	__device__ static float dot(vec3 a, vec3 b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	__device__ vec3 operator-()
	{
		return vec3(-x, -y, -z);
	}

	__device__ vec3 operator+=(vec3 v)
	{
		x += v.x; y += v.y; z += v.z;
	}

	__device__ vec3 operator-=(vec3 v)
	{
		x -= v.x; y -= v.y; z -= v.z;
	}

	__device__ vec3 operator+(vec3 v)
	{
		return vec3(x + v.x, y + v.y, z + v.z);
	}

	__device__ vec3 operator-(vec3 v)
	{
		return vec3(x - v.x, y - v.y, z - v.z);
	}

	__device__ vec3 operator+(float a)
	{
		return vec3(x + a, y + a, z + a);
	}

	__device__ vec3 operator-(float a)
	{
		return vec3(x - a, y - a, z - a);
	}

	__device__ vec3 operator*(float a)
	{
		return vec3(x * a, y * a, z * a);
	}

	__device__ vec3 operator/(float a)
	{
		return vec3(x / a, y / a, z / a);
	}
};

struct mat3x3
{
public:
	float v[9];

	__device__ float operator[](int i)
	{
		return v[i];
	}

	__device__ mat3x3 operator*(mat3x3 m)
	{
		return{
			v[0] * m[0] + v[1] * m[3] + v[2] * m[6], v[0] * m[1] + v[1] * m[4] + v[2] * m[7], v[0] * m[2] + v[1] * m[5] + v[2] * m[8],
			v[3] * m[0] + v[4] * m[3] + v[5] * m[6], v[3] * m[1] + v[4] * m[4] + v[5] * m[7], v[3] * m[2] + v[4] * m[5] + v[5] * m[8],
			v[6] * m[0] + v[7] * m[3] + v[8] * m[6], v[6] * m[1] + v[7] * m[4] + v[8] * m[7], v[6] * m[2] + v[7] * m[5] + v[8] * m[8]
		};
	}

	__device__ vec3 operator*(vec3 m)
	{
		return vec3(v[0] * m.x + v[1] * m.y + v[2] * m.z, v[3] * m.x + v[4] * m.y + v[5] * m.z, v[6] * m.x + v[7] * m.y + v[8] * m.z);
	}
};