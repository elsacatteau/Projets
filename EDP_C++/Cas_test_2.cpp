#include "TransportDiffusion.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace::std;
// Constructeur
TransportDiffusion::TransportDiffusion(double C_, double D_, double L_, int Nx_, int Nt_): C(C_), D(D_), L(L_), Nx(Nx_), Nt(Nt_){
    dx = L / Nx;
    gamma = 0.25;
    v = 0.5;
    double dt1 = gamma * dx * dx / D;
    double dt2 = v * dx / std::abs(C);
    dt = std::min(dt1, dt2);
    T = Nt * dt;

    x.resize(Nx);
    t.resize(Nt);
    u_calcule.resize(Nx, std::vector<double>(Nt, 0.0));

    for (int i = 0; i < Nx; ++i) x[i] = i * dx;
    for (int n = 0; n < Nt; ++n) t[n] = n * dt;

    for (int i = 0; i < Nx; ++i)
        u_calcule[i][0] = u_exacte(x[i], 0);
}

// Destructeur
TransportDiffusion::~TransportDiffusion() {}

// Solution exacte
double TransportDiffusion::u_exacte(double x, double t) {
    return std::sin(M_PI*x)*std::exp(-t);
}

// Terme source
double TransportDiffusion::f(double x, double t) {
    return -std::sin(M_PI*x)*std::exp(-t)+D*(M_PI*M_PI)*std::sin(M_PI*x)*std::exp(-t)
    +C*M_PI*std::cos(M_PI*x)*std::exp(-t);
}

// Calcul
void TransportDiffusion::calculer() {
    for (int n = 0; n < Nt-1; ++n) {
        for (int i = 1; i < Nx-1; ++i) {
            double du = (C > 0) ? (u_calcule[i][n] - u_calcule[i-1][n])
                                 : (u_calcule[i+1][n] - u_calcule[i][n]);
            u_calcule[i][n+1] = u_calcule[i][n] + gamma*(u_calcule[i+1][n] - 2*u_calcule[i][n] + u_calcule[i-1][n])
                                 - v*du + f(x[i], t[n])*dt;
        }
        u_calcule[0][n+1] = 0;
        u_calcule[Nx-1][n+1] = 0;
    }
}

// Erreurs
double TransportDiffusion::erreur_L2() {
    double sum = 0.0;
    for (int i = 0; i < Nx; ++i) {
        double e = std::abs(u_calcule[i][Nt-1] - u_exacte(x[i], T));
        sum += e*e;
    }
    return std::sqrt(dx * sum);
}

double TransportDiffusion::erreur_Linf() {
    double max_err = 0.0;
    for (int i = 0; i < Nx; ++i) {
        double e = std::abs(u_calcule[i][Nt-1] - u_exacte(x[i], T));
        if (e > max_err) max_err = e;
    }
    return max_err;
}

// Export CSV
void TransportDiffusion::exporter_csv(const std::string& filename) {
    std::ofstream file(filename);
    for (int n = 0; n < Nt; ++n) {
        for (int i = 0; i < Nx; ++i) {
            file << u_calcule[i][n];
            if (i != Nx-1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

void TransportDiffusion::exportData(const std::string& filename) {
    std::ofstream file(filename);
    for (int i = 0; i < Nx; ++i) {
        file << x[i] << " " << u_exacte(x[i], T) << " " << u_calcule[i][Nt-1] << "\n";
    }
    file.close();
}
