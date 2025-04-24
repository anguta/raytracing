#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Updated resolution to 1024×1024
constexpr int WIDTH     = 1024;
constexpr int HEIGHT    = 1024;
constexpr int N_SPHERES = 10;

// Simple 3-float vector
struct Vec {
    float x, y, z;
    __host__ __device__ Vec() : x(0), y(0), z(0) {}
    __host__ __device__ Vec(float X, float Y, float Z) : x(X), y(Y), z(Z) {}

    __host__ __device__ Vec operator+(Vec const &o) const { return {x+o.x, y+o.y, z+o.z}; }
    __host__ __device__ Vec operator-(Vec const &o) const { return {x-o.x, y-o.y, z-o.z}; }
    __host__ __device__ Vec operator*(float k)      const { return {x*k,   y*k,   z*k  }; }
    __host__ __device__ Vec operator-()             const { return Vec{-x, -y, -z}; }
};
__host__ __device__ inline Vec operator*(float k, Vec const &v) { return v * k; }

__host__ __device__ inline float dot(Vec const &a, Vec const &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__host__ __device__ inline Vec norm(Vec const &v) {
    float invLen = rsqrtf(dot(v,v));
    return {v.x * invLen, v.y * invLen, v.z * invLen};
}

// Sphere definition
struct Sphere {
    Vec   center;
    float radius;
    Vec   color;
};

// ray–sphere intersection
__device__ bool hit_sphere(Vec const &orig, Vec const &dir,
                           Sphere const &s, float &t) {
    Vec oc = orig - s.center;
    float b = dot(oc, dir);
    float c = dot(oc, oc) - s.radius * s.radius;
    float disc = b*b - c;
    if (disc < 0.0f) return false;
    t = -b - sqrtf(disc);
    return t > 0.0f;
}

// Phong shading
__host__ __device__ Vec phong_shade(
    Vec const &P, Vec const &N, Vec const &V,
    Vec const &lightPos, Vec const &Kd, Vec const &Ks, float shininess) {
    Vec ambient = 0.1f * Kd;
    Vec L       = norm(lightPos - P);
    float diff  = fmaxf(dot(N,L), 0.0f);
    Vec diffuse = diff * Kd;
    Vec R       = norm(2.0f * dot(N,L) * N - L);
    float spec  = powf(fmaxf(dot(R,V), 0.0f), shininess);
    Vec specular= spec * Ks;
    return ambient + diffuse + specular;
}

// render kernel: one thread per pixel
__global__ void render_kernel(Vec *fb, Sphere const *spheres) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    Vec origin   {0.0f,  0.0f,  2.0f};
    Vec lightPos{-5.0f, -5.0f, 10.0f};

    float u = -1.0f + 2.0f * x / (WIDTH  - 1);
    float v = -1.0f + 2.0f * y / (HEIGHT - 1);
    Vec dir = norm(Vec{u,v,0.0f} - origin);

    float best_t = 1e20f;
    int   hit_id = -1;
    #pragma unroll
    for(int i = 0; i < N_SPHERES; ++i) {
        float t;
        if (hit_sphere(origin, dir, spheres[i], t) && t < best_t) {
            best_t = t;
            hit_id = i;
        }
    }

    Vec pixelColor;
    if (hit_id >= 0) {
        Vec P  = origin + dir * best_t;
        Vec N  = norm(P - spheres[hit_id].center);
        Vec V  = norm(-dir);
        Vec Kd = spheres[hit_id].color;
        pixelColor = phong_shade(P, N, V, lightPos, Kd, {1,1,1}, 32.0f);
    } else {
        int ix = int(floor((u+1.0f)*5.0f));
        int iy = int(floor((v+1.0f)*5.0f));
        pixelColor = ((ix + iy)&1)==0 ? Vec{0.9f,0.9f,0.9f}
                                     : Vec{0.1f,0.1f,0.1f};
    }

    fb[y*WIDTH + x] = pixelColor;
}

inline int toByte(float c) {
    c = fminf(fmaxf(c,0.0f),1.0f);
    return int(c * 255.999f);
}

int main() {
    // prepare host spheres
    Sphere h_spheres[N_SPHERES];
    for(int i = 0; i < N_SPHERES; ++i) {
        float y = -1.0f + i * (2.0f/(N_SPHERES-1));
        float z = -2.0f - i * 0.5f;
        h_spheres[i].center = {0.0f, y, z};
        h_spheres[i].radius = 0.75f;
        float t = float(N_SPHERES - i) / N_SPHERES;
        h_spheres[i].color  = {t, 0.5f, 1.0f - t};
    }

    // allocate device spheres in global memory
    Sphere *d_spheres;
    cudaMalloc(&d_spheres, N_SPHERES * sizeof(Sphere));
    cudaMemcpy(d_spheres, h_spheres,
               N_SPHERES * sizeof(Sphere),
               cudaMemcpyHostToDevice);

    // allocate device framebuffer
    Vec *d_fb;
    cudaMalloc(&d_fb, WIDTH*HEIGHT * sizeof(Vec));

    // allocate pinned host memory
    Vec *h_fb;
    cudaMallocHost(&h_fb, WIDTH*HEIGHT * sizeof(Vec));

    // launch kernel
    dim3 block(16,16);
    dim3 grid((WIDTH+block.x-1)/block.x,
              (HEIGHT+block.y-1)/block.y);
    render_kernel<<<grid,block>>>(d_fb, d_spheres);
    cudaDeviceSynchronize();

    // copy framebuffer back to host
    cudaMemcpy(h_fb, d_fb,
               WIDTH*HEIGHT * sizeof(Vec),
               cudaMemcpyDeviceToHost);

    // output PPM
    std::cout << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255\n";
    for(int y = HEIGHT-1; y >= 0; --y) {
        for(int x = 0; x < WIDTH; ++x) {
            Vec c = h_fb[y*WIDTH + x];
            std::cout
                << toByte(c.x) << ' '
                << toByte(c.y) << ' '
                << toByte(c.z) << '\n';
        }
    }

    // cleanup
    cudaFree(d_spheres);
    cudaFree(d_fb);
    cudaFreeHost(h_fb);
    return 0;
}
