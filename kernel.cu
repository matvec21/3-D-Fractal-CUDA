#include "bitmap_image.hpp"
#include "device_vec3.h"

#define PI 3.14159265358979323846

#define THREADS 512

#define WIDTH 10000
#define HEIGHT 10000

#define FOV 4 * PI / 180.f
#define MAXDIST 100
#define MINSTEP 0.000001f
#define STEPS 2000
#define ITERATIONS 2000

__device__ float _min(float a, float b)
{
	return a < b ? a : b;
}

__device__ float _max(float a, float b)
{
	return a > b ? a : b;
}

__device__ vec3 rotate(vec3 v, float a[6])
{
	mat3x3 Rx = { 1, 0, 0, 0, cos(a[0]), -sin(a[0]), 0, sin(a[0]), cos(a[0]) };
	mat3x3 Ry = { cos(a[1]), 0, sin(a[1]), 0, 1, 0, -sin(a[1]), 0, cos(a[1]) };
	mat3x3 Rz = { cos(a[2]), -sin(a[2]), 0, sin(a[2]), cos(a[2]), 0, 0, 0, 1 };

	return Rz * Ry * Rx * v;
}

__device__ float getDistance(vec3 pos)
{
	float n = 2.f; float r = 0.f; float dr = 1.f;
	vec3 c = -pos;
	float y = c.y;
	c.y = c.z;
	c.z = y;
	c.x = -c.x;
	vec3 z = c;

	for (int i = 0; i < ITERATIONS; i++)
	{
		r = z.length();
		if (r >= 4)
			break;
		else if (i == ITERATIONS - 1)
			return 0;

		dr = pow(r, n - 1.f) * n * dr + 1.f;
//		float phi = atan2(z.x, z.y) * n;
	//	float theta = atan2(sqrt(z.x * z.x + z.y * z.y), z.z) * n;
		float phi = atan(z.y / z.x) * n;
		float theta = atan(z.z / sqrt(z.x * z.x + z.y * z.y)) * n;

		z = vec3(cos(phi) * cos(theta), sin(phi) * cos(theta), sin(theta)) * pow(r, n);
		z += c;
	}

	return 0.5f * log(r) * r / dr;
}

__device__ vec3 getNormal(vec3 pos)
{
	vec3 a = vec3(MINSTEP, 0, 0);
	vec3 b = vec3(0, MINSTEP, 0);
	vec3 c = vec3(0, 0, MINSTEP);

	float gx = getDistance(pos + a) - getDistance(pos - a);
	float gy = getDistance(pos + b) - getDistance(pos - b);
	float gz = getDistance(pos + c) - getDistance(pos - c);
	return vec3(gx, gy, gz).normalized();
}

__device__ bool trace(vec3 start, vec3 dir, vec3* hit)
{
	*hit = start;
	for (int i = 0; i < STEPS; i++)
	{
		float distance = getDistance(*hit);
		if (distance <= MINSTEP)
			return true;

		*hit += dir * distance;
		if ((*hit - start).length() >= MAXDIST)
			return false;
	}
	return false;
}

__global__ void kernelProcess(unsigned char* pixels)
{
	int index = blockDim.x * gridDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= WIDTH * HEIGHT)
		return;

	static const float fov = FOV;
	static float camRot[6] = { 0, 0, 0, 0, 0, 0 };
	vec3 camPos = { -1.94106, 0, -0.0666499 };
	vec3 lightPos = { 100, 100, 100 };

	float x = (float)(index % WIDTH);
	float y = (float)(index / WIDTH);
	index *= 3;

	float angle[6];
	angle[0] = fov * (x / WIDTH - 0.5f);
	angle[1] = fov * (y - HEIGHT * 0.5f) / WIDTH;
	angle[2] = PI / 2;
	angle[3] = 0.f;
	angle[4] = 0.f;
	angle[5] = 0.f;

	vec3 dir = rotate(vec3(0.f, 0.f, 1.f), angle);
	dir = rotate(dir, camRot);

	float3 color = { 0.2f, 0.2f, 0.2f };

	vec3 hit;
	if (trace(camPos, dir, &hit))
	{
		color = { 0.6f, 0.f, 0.f };
		vec3 normal = getNormal(hit);
		vec3 lightDir = (lightPos - hit).normalized();

	//	if (!trace(hit + normal * (1.1f * MINSTEP), lightDir, &hit))
		{
			float t = vec3::dot(normal, lightDir);
			color.x += 0.4f * t;
			color.y += 0.2f * t;
			color.z += 0.2f * t;
		}
	}

	color.x = _max(_min(color.x, 1.f), 0.f);
	color.y = _max(_min(color.y, 1.f), 0.f);
	color.z = _max(_min(color.z, 1.f), 0.f);

	pixels[index + 0] = color.x * 255;
	pixels[index + 1] = color.y * 255;
	pixels[index + 2] = color.z * 255;
}

int main(void)
{
	const int size = WIDTH * HEIGHT * 3;
	unsigned char* pixels = new unsigned char[size];
	unsigned char* dev_pixels;

	cudaSetDevice(0);
	cudaMalloc(&dev_pixels, size * sizeof(unsigned char));

	float x = ceil((float)(WIDTH * HEIGHT) / THREADS);
	dim3 blocks(65535, ceil(x / 65535.f), 1);
	kernelProcess << < blocks, THREADS >> >(dev_pixels);

	cudaDeviceSynchronize();
	cudaMemcpy(pixels, dev_pixels, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(dev_pixels);
	cudaDeviceReset();

	bitmap_image img(WIDTH, HEIGHT);
	for (int i = 0; i < WIDTH * HEIGHT; i++)
	{
		rgb_t color = { pixels[i * 3 + 0], pixels[i * 3 + 1], pixels[i * 3 + 2] };
		img.set_pixel(i % WIDTH, i / WIDTH, color);
	}
	img.save_image("result.bmp");
	return 0;
}
