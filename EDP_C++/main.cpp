#include "TransportDiffusion.h"
#include <iostream>

int main() {
    double C = 1.0, D = 1.0, L = 1.0;
    int Nx = 10000, Nt = 50;

    TransportDiffusion td(C, D, L, Nx, Nt);
    td.calculer();

    std::cout << "Erreur L2 : " << td.erreur_L2() << std::endl;
    std::cout << "Erreur Linf : " << td.erreur_Linf() << std::endl;

    td.exporter_csv("resultats.csv");

    // Export pour gnuplot
    td.exportData("C:/Users/elsac/Documents/4a/Cpp/solution.dat");

    std::cout << "Fichier solution.dat cree" << std::endl;

    return 0;
}

