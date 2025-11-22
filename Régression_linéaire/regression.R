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

xi = X[,4]
Yi = Y
xn_barre=mean(xi)
Yn_barre=mean(Yi)
n=length(xi)
an_chapeau = (sum(xi * Yi) - n * xn_barre * Yn_barre) / sum((xi - xn_barre)^2)
bn_chapeau=Yn_barre-an_chapeau*xn_barre
cov=cov(an_chapeau,bn_chapeau)
Yi_chapeau=an_chapeau*xi+bn_chapeau
plot(xi, Yi, main = "Régression linéaire", xlab = "variable explicative", ylab = "variable réponse")
abline(a=bn_chapeau, b=an_chapeau, col = "red")
abline(lm(Yi ~ xi), col = "blue")

#Intervalle de confiance
df=data.frame(xi,Yi)
plot(df$xi, df$Yi, main = "Régression linéaire (intervalle de confiance à 95%)", xlab = "variable explicative", ylab = "variable réponse")
x_seq=seq(min(df$xi),max(df$xi),length.out = 100)
abline(lm(Yi ~ xi,data=df), col = "red") #droite de régression linéaire
x_new= data.frame(xi=x_seq)
pred = predict(lm(Yi~xi), newdata = x_new, interval = "confidence", level = 0.95)
lines(x_seq, pred[,"lwr"], col = "blue", lty=2) # borne inférieure
lines(x_seq, pred[,"upr"], col = "blue", lty=2) # borne supérieure
#Prédiction
pred = predict(lm(Yi~xi), newdata = x_new, interval = "prediction", level = 0.95)
lines(x_seq, pred[,"lwr"], col = "green", lty=2) # borne inférieure
lines(x_seq, pred[,"upr"], col = "green", lty=2) # borne supérieure

R_carre = 1 - ( (sum((Yi_chapeau - Yi)^2) )/(sum((Yi - Yn_barre)^2)))
sigman_carre_chapeau=(1/(n-2))*sum((Yi-Yi_chapeau)^2)
sigman_chapeau=sqrt(sigman_carre_chapeau)

#Résidus standardisés
X_matrice = cbind(1, xi) #Crée la matrice X avec sur la première colonne que des 1 puis les xi
X_transpose=t(X_matrice)
H = X_matrice %*% solve(t(X_matrice) %*% X_matrice) %*% t(X_matrice)
hi = diag(H)  
residus_sd = (Yi - Yi_chapeau) / (sigman_chapeau * sqrt(1 - hi)) #formule des résidus standardisés
qqnorm(residus_sd , main = "Normal Q-Q",ylab="Résidus standardisés",xlab="")

#Résidus studentisés
residus_st = rstudent(lm(Yi ~ xi)) #on récupère les résidus studentisés
kolmogorov = ks.test(residus_st , "pt", df = n - 3) #test de Kolomogorov
print(kolmogorov)
plot(ecdf(residus_st), col = "red", main = "Test de Kolmogorov", xlab = "Valeurs", ylab = "Probabilités cumulées", verticals=TRUE, do.points = FALSE)
curve(pt(x, df = n - 3), col = "blue", add = TRUE)
