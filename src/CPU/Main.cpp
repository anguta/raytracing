#include "Vec.h"
#include "Ray.h"
#include "Sphere.h"
#include <vector>
#include <iostream>
#include <cmath>

// resolution
static const int WIDTH      = 1024;
static const int HEIGHT     = 1024;
// number of spheres in vertical stack
static const int N_SPHERES  = 10;

// light and camera
Vec LIGHT{-5.0f, -5.0f, 10.0f};
Vec ORIGIN{0.0f,  0.0f,  2.0f};

// scene storage
std::vector<Sphere> scene;

// ray-sphere intersection (returns t or INFINITY)
float intersect_sphere(const Sphere &s, const Ray &r) {
    Vec  Oc = r.A() - s.c();
    float b = dot(r.B(), Oc);
    float c = dot(Oc, Oc) - s.r()*s.r();
    float disc = b*b - c;
    if (disc < 0.0f) return INFINITY;
    float t = -b - sqrtf(disc);
    return (t > 0.0f) ? t : INFINITY;
}

// Phong shading (ambient + diffuse + specular)
Vec phong_shade(const Vec &P, const Vec &N, const Vec &V,
                const Vec &Lpos, const Vec &Kd, const Vec &Ks, float shin) {
    // ambient
    Vec ambient = 0.1f * Kd;
    // diffuse
    Vec L = norm(Lpos - P);
    float diff = fmaxf(dot(N,L), 0.0f);
    Vec diffuse = diff * Kd;
    // specular
    Vec R = norm(2.0f * dot(N,L) * N - L);
    float spec = powf(fmaxf(dot(R,V), 0.0f), shin);
    Vec specular = spec * Ks;
    return ambient + diffuse + specular;
}

// clamp to [0,255]
int clip(float c) {
    if (c <= 0.0f) return 0;
    if (c >= 1.0f) return 255;
    return int(c * 255.999f);
}

int main() {
    // build vertical stack of spheres
    scene.reserve(N_SPHERES);
    for (int i = 0; i < N_SPHERES; ++i) {
        float y = -1.0f + i * (2.0f / (N_SPHERES - 1));
        float z = -2.0f - i * 0.5f;
        float t = float(N_SPHERES - i) / N_SPHERES;
        Vec  color{ t, 0.5f, 1.0f - t };
        scene.emplace_back(Sphere{ Vec{0.0f, y, z}, 0.75f, color, 0.0f });
    }

    // output PPM header
    std::cout << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255\n";

    // loop over every pixel
    for (int j = HEIGHT - 1; j >= 0; --j) {
        for (int i = 0; i < WIDTH; ++i) {
            // normalized screen coords
            float u = -1.0f + 2.0f * i / (WIDTH  - 1);
            float v = -1.0f + 2.0f * j / (HEIGHT - 1);
            Vec   dir = norm(Vec{u, v, 0.0f} - ORIGIN);
            Ray   r(ORIGIN, dir);

            // find closest sphere
            float best_t = INFINITY;
            int   hit_id = -1;
            for (int k = 0; k < N_SPHERES; ++k) {
                float t = intersect_sphere(scene[k], r);
                if (t < best_t) {
                    best_t = t;
                    hit_id = k;
                }
            }

            Vec pixelColor;
            if (hit_id >= 0) {
                // compute Phong shading
                Vec P  = r.P(best_t);
                Vec N  = norm(P - scene[hit_id].c());
                Vec Vc = norm(ORIGIN - P);
                Vec Kd = scene[hit_id].col();
                Vec Ks{1,1,1};
                pixelColor = phong_shade(P, N, Vc, LIGHT, Kd, Ks, 32.0f);
            } else {
                // checkered background
                int ix = int(floor((u + 1.0f) * 5.0f));
                int iy = int(floor((v + 1.0f) * 5.0f));
                bool white = ((ix + iy) & 1) == 0;
                pixelColor = white ? Vec{0.9f,0.9f,0.9f}
                                   : Vec{0.1f,0.1f,0.1f};
            }

            // write pixel
            std::cout
                << clip(pixelColor.x()) << ' '
                << clip(pixelColor.y()) << ' '
                << clip(pixelColor.z()) << '\n';
        }
    }

    return 0;
}
