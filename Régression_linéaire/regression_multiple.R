A=mtcars
F1=as.factor(A[,8]) #prend la 8e colonnne
A[,8]=F1
set.seed(13) #permet de fixer les données
set.seed(13*floor(100*runif(1,0,3)))
set1=sample(1:32,1) #donne un nombre aléatoire entre 1 et 32
B=A[-set1,] #enlève la set1-ème ligne de A
Y=B[,1] #1ère colonne de B
u=1:11 #vecteur 1 à 11
v=u[-c(1,8,9)] #correspond au vecteur u sans les valeurs 1,8 et 9
set2=c(8,sample(v,6,replace=FALSE)) #vecteur commençant par 8 puis 6 valeurs aléatoires qui appartiennent au vecteur v
X=B[,set2] #prend les colonnes de B qui correspondent aux coefficients du vecteur set2
X1=X[,-c(1,2,6,7)] #on enlève les variables qualitatives

Yn_barre=mean(Y)
X_matrice=model.matrix(~ ., data = X1) # Crée la matrice X à partir de X1 
n=nrow(X_matrice) #taille

# Vérifie l'inversibilité de la matrice
if (abs(det(t(X_matrice) %*% X_matrice)) > 1e-10) {
  beta_chapeau = solve(t(X_matrice) %*% X_matrice) %*% t(X_matrice) %*% Y
} else {
  stop("Matrice non inversible — arrêt du programme.")
}

beta_chapeau = solve(t(X_matrice) %*% X_matrice) %*% t(X_matrice) %*% Y
Y_chapeau=X_matrice%*%beta_chapeau
plot (Y,Y_chapeau,main="^Y en fonction de Y") # Graphe de Y chapeau en fonction de Y
abline(a=0,b=1,col="red") #trace la droite

rang=qr(X_matrice)$rank #rang de la matrice
sigman_carre_chapeau=(1/(n-rang))*sum((Y-Y_chapeau)^2)
sigman_chapeau=sqrt(sigman_carre_chapeau)

#Formule de F
F=(sum((Y_chapeau-Yn_barre)^2)/(rang-1))/((sum((Y-Y_chapeau)^2)/(n-rang)))

#Résidus standardisés
H = X_matrice %*% solve(t(X_matrice) %*% X_matrice) %*% t(X_matrice) #matrice H
hi = diag(H)  
residus_sd = (Y - Y_chapeau) / (sigman_chapeau * sqrt(1 - hi)) #formule des résidus standardisés
qqnorm(residus_sd, main = "Normal Q-Q",ylab="Résidus Standardisés",xlab="Quantités théoriques")

#Résidus studentisés
X1=as.matrix(X1) #transforme X1 en matrice
residus_st = rstudent(lm(Y ~ X1)) #récupère les résidus de student
kolmogorov = ks.test(residus_st, "pt",df=n-rang-1) #test de kolmogorov
plot(ecdf(residus_st),main="Test de Kolmogorv", verticals = TRUE, do.points = FALSE, col = "red", xlab = "valeurs", ylab = "probabilités cumulées")
curve(pt(x, df = n - rang - 1), col = "blue", add = TRUE)

#p-valeur
p_value = 1 - pf(F, df1 = rang - 1, df2 = n - rang)
alpha = 0.05
if (p_value < alpha) {
  cat("On décide H1\n")
} else {
  cat("On ne rejette pas H0\n")
}

#Intervalle de confiance
x_new = A[set1, set2[-c(1,2,6,7)]] #on prend la ligne enlevée dans A[-set1,] qui correspond à une nouvelle observation
x_new_mat = model.matrix(~ ., data = x_new) 
y_new_chapeau = x_new_mat %*% beta_chapeau
val = sigman_chapeau*sqrt((x_new_mat %*% solve(t(X_matrice) %*% X_matrice) %*% t(x_new_mat)))
alpha = 0.05
t = qt(1 - alpha/2, df = n - rang) # Quantile 
borne_inf = y_new_chapeau - t * val
borne_sup = y_new_chapeau + t * val
cat("Valeur de y_new_chapeau est :", y_new_chapeau, "\n") #affiche 27.70598
cat("Intervalle de confiance à 95% : [", borne_inf, ",", borne_sup, "]\n") #affiche l'intervalle [ 23.72286 , 31.68909 ]

#Analyse de la variance à 1 facteur
#On récupère nos variables qualitatives
vs=B$vs 
L1=lm(Y ~ vs)  
boxplot(Y ~ vs, main = "Boîtes à moustache de la variable vs", xlab = "vs", ylab = "Y", col = c("lightpink", "lightblue"), names = levels(vs)) #boîte à moustache de vs
moyennes = tapply(Y, vs, mean)
points(1:length(moyennes), moyennes, col = "red", pch = 19) 
summary(aov(L1)) #avo(L1) convertit le modèle linéaire L1 en modèle d'analyse de la variance
#summary(aov(L1)) affiche un tableau de décomposition de la variance
TUKEY_vs=TukeyHSD(aov(L1)) #applique le test de Tukey 
print(TUKEY_vs)
plot(TUKEY_vs)

carb=as.factor(B$carb)
L2=lm(Y ~ carb)  
boxplot(Y ~ carb, main = "Boîtes à moustache de la variable carb", xlab = "carb", ylab = "Y", col = c("lightpink", "lightblue","lightgreen", "orange","yellow" ), names = levels(carb)) #boîte à moustache de carb
moyennes = tapply(Y, carb, mean)
points(1:length(moyennes), moyennes, col = "red", pch = 19)
summary(aov(L2))
TUKEY_carb=TukeyHSD(aov(L2))
plot(TUKEY_carb)

gear=as.factor(B$gear)
L3=lm(Y ~ gear) 
boxplot(Y ~ gear, main = "Boîtes à moustache de la variable gear", xlab = "gear", ylab = "Y", col = c("lightpink", "lightblue","lightgreen"), names = levels(gear)) #boîte à moustache de gear
moyennes = tapply(Y, gear, mean)
points(1:length(moyennes), moyennes, col = "red", pch = 19)
summary(aov(L3))
TUKEY_gear=TukeyHSD(aov(L3))
plot(TUKEY_gear)

cyl=as.factor(B$cyl)
L4=lm(Y ~ cyl)
boxplot(Y ~ cyl, main = "Boîtes à moustache de la variable cyl", xlab = "cyl", ylab = "Y", col = c("lightpink", "lightblue","lightgreen"), names = levels(cyl)) #boîte à moustache de cyl
moyennes = tapply(Y, cyl, mean)
points(1:length(moyennes), moyennes, col = "red", pch = 19)
summary(aov(L4))
TUKEY_cyl=TukeyHSD(aov(L4))
plot(TUKEY_cyl)

#Graphe pour comparer R_carre et Ra_carre
library(leaps)
R_carre = 1-((sum((Y_chapeau - Y)^2) )/(sum((Y - Yn_barre)^2)))
Ra_carre=1-((n-1)*(1-R_carre)/(n-rang))
reg = regsubsets(X1, Y, nvmax = 7, method = "exhaustive") #fonction de leaps pour effectuer une sélection de variables
summary = summary(reg)
nb = which.max(summary$adjr2) #nombre de variables à sélectionner pour avoir le meilleur Ra_carre
var = summary$which[nb,-1] #summary$which est une matrice booléenne indiquant quelles variables sont incluses dans chaque modèle
#var récupère la ligne de la matrice summary$which qui correspond aux variables à sélectionner pour avoir le meilleur Ra_carre et on enlève la colonne Intercept
print(var) #3 variables carb, disp et gear
plot(summary$rsq, type = "l", col = "blue", pch = 16, ylim = c(0.6, 0.85), xlab = "Nombre de variables", ylab = "Valeurs",main = "Comparaison de R² et R² ajusté") #graphe avec tracé de R_carre
lines(summary$adjr2, type = "l", col = "red", pch = 16) #on trace Ra_carre
legend("bottomright", legend = c("R²", "R² ajusté"),col = c("blue", "red"), lty=1, pch = 16, bty = "n")
