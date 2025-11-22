#ifndef TRANSPORTDIFFUSION_H
#define TRANSPORTDIFFUSION_H

#include <vector>
#include <string>

class TransportDiffusion {
private:
    double C, D, L;
    int Nx, Nt;
    double dx, dt, T;
    double gamma, v;
    std::vector<double> x;
    std::vector<double> t;
    std::vector<std::vector<double>> u_calcule;

public:
    // Constructeur et destructeur
    TransportDiffusion(double C_, double D_, double L_, int Nx_, int Nt_);
    ~TransportDiffusion();

    // Méthodes
    double u_exacte(double xi, double ti);
    double f(double xi, double ti);
    void calculer();
    double erreur_L2();
    double erreur_Linf();
    void exporter_csv(const std::string& filename);
    void exportData(const std::string& filename);
};

#endif
