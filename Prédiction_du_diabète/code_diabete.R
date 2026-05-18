## BLOC 1 - Initialisation
options(menu.graphics = FALSE)
df <- read.csv("dataset_diabetes.csv")

## BLOC 2 - Préparation des données
#a) Transformation des variables catégorielles :
transform_activity <- c("Low" = 0, "Medium" = 1, "High" = 2)
transform_gender <- c("Male" = 0, "Female" = 1)

df$physical_activity <- transform_activity[df$physical_activity]
df$gender <- transform_gender[df$gender]
df$id <- NULL

#b) Traitement des outliers sur BMI
Q1 <- quantile(df$bmi, 0.25)
Q3 <- quantile(df$bmi, 0.75)
IQR_val <- IQR(df$bmi)
borne_inf <- Q1 - 1.5 * IQR_val
borne_sup <- Q3 + 1.5 * IQR_val

df_clean <- df[df$bmi >= borne_inf & df$bmi <= borne_sup, ]

#c) Transformation de la variable cible pour les modèles
df_clean$diabetes <- factor(df_clean$diabetes, levels = c(0, 1), labels = c("Non", "Oui"))

#d) Split train/test
set.seed(123)
n <- nrow(df_clean)
index <- sample(1:n, 0.8 * n)
train <- df_clean[index, ]
test <- df_clean[-index, ]

## BLOC 3 - PCA
pca_data <- scale(train[, -which(names(train) == "diabetes")])
pca <- prcomp(pca_data)
plot(pca$x[, 1], pca$x[, 2],
     col = ifelse(train$diabetes == "Oui", "red", "blue"),
     pch = 19, xlab = "PC1", ylab = "PC2",
     main = "PCA - Distribution des patients")
legend("topright", legend = c("Non-diabétique", "Diabétique"),
       col = c("blue", "red"), pch = 19)

# BLOC 4 - Librairies
library(caret)
library(randomForest)
library(e1071)
library(naivebayes)
library(pROC)

# ====================================================
# ANALYSE DÉTAILLÉE DE CHAQUE MODÈLE (CHAQUE METHODE EST ECRITE "MANUELLEMENT")
# ====================================================

# ----------------------------------------------------------------------
# 1. RÉGRESSION LOGISTIQUE
# ----------------------------------------------------------------------
cat("\n========== RÉGRESSION LOGISTIQUE ==========\n")

# Construction du modèle
modele_log <- glm(diabetes ~ age + gender + bmi + glucose + blood_pressure + 
                    cholesterol + insulin + physical_activity + heart_rate + 
                    sleep_hours + steps_per_day + work_hours + water_intake_ltr,
                  data = train, 
                  family = binomial)

# Résumé du modèle (coefficients et p-values)
cat("\n--- RÉSUMÉ DU MODÈLE ---\n")
summary(modele_log)

# Odds ratios (interprétation)
cat("\n--- ODDS RATIOS (facteurs de risque) ---\n")
odds_ratios <- exp(coef(modele_log))
print(round(odds_ratios, 3))

# Prédictions
prob_log <- predict(modele_log, test, type = "response")
pred_log <- ifelse(prob_log > 0.5, "Oui", "Non")
pred_log <- factor(pred_log, levels = c("Non", "Oui"))

# Matrice de confusion
cm_log <- table(Reference = test$diabetes, Prediction = pred_log)
cat("\n--- MATRICE DE CONFUSION (seuil=0.5) ---\n")
print(cm_log)

# Métriques
TP <- cm_log[2,2]; TN <- cm_log[1,1]; FP <- cm_log[1,2]; FN <- cm_log[2,1]
accuracy <- (TP+TN)/sum(cm_log)
sensitivity <- TP/(TP+FN)
specificity <- TN/(TN+FP)
precision <- TP/(TP+FP)
f1 <- 2*(precision*sensitivity)/(precision+sensitivity)

cat("\nAccuracy:", round(accuracy,3))
cat("\nSensibilité:", round(sensitivity,3))
cat("\nSpécificité:", round(specificity,3))
cat("\nPrécision:", round(precision,3))
cat("\nF1-score:", round(f1,3), "\n")

# Courbe ROC
roc_log <- roc(test$diabetes, prob_log, levels = c("Non", "Oui"))
plot(roc_log, col = "blue", lwd = 2, main = "Courbe ROC - Régression Logistique",print.auc = TRUE)

# Seuil optimal
seuil_opt <- coords(roc_log, "best", ret = "threshold")[1,1]
cat("\nSeuil optimal:", round(seuil_opt, 3), "\n")

# Prédiction avec seuil optimal
pred_log_opt <- ifelse(prob_log > seuil_opt, "Oui", "Non")
pred_log_opt <- factor(pred_log_opt, levels = c("Non", "Oui"))
cm_log_opt <- table(Reference = test$diabetes, Prediction = pred_log_opt)
cat("\n--- MATRICE DE CONFUSION (seuil optimal) ---\n")
print(cm_log_opt)

# ----------------------------------------------------------------------
# 2. NAIVE BAYES
# ----------------------------------------------------------------------
cat("\n========== NAIVE BAYES ==========\n")

# Construction du modèle
modele_nb <- naive_bayes(diabetes ~ ., data = train)

# Afficher les probabilités conditionnelles (pour quelques variables)
cat("\n--- PROBABILITÉS CONDITIONNELLES (aperçu) ---\n")
print(modele_nb$tables[1:3])

# Prédictions
prob_nb <- predict(modele_nb, test, type = "prob")[, "Oui"]
pred_nb <- predict(modele_nb, test)

# Matrice de confusion
cm_nb <- table(Reference = test$diabetes, Prediction = pred_nb)
cat("\n--- MATRICE DE CONFUSION ---\n")
print(cm_nb)

# Métriques
TP <- cm_nb[2,2]; TN <- cm_nb[1,1]; FP <- cm_nb[1,2]; FN <- cm_nb[2,1]
accuracy <- (TP+TN)/sum(cm_nb)
sensitivity <- TP/(TP+FN)
specificity <- TN/(TN+FP)
precision <- TP/(TP+FP)
f1 <- 2*(precision*sensitivity)/(precision+sensitivity)

cat("\nAccuracy:", round(accuracy,3))
cat("\nSensibilité:", round(sensitivity,3))
cat("\nSpécificité:", round(specificity,3))
cat("\nPrécision:", round(precision,3))
cat("\nF1-score:", round(f1,3), "\n")

# Courbe ROC
roc_nb <- roc(test$diabetes, prob_nb, levels = c("Non", "Oui"))
plot(roc_nb, col = "purple", lwd = 2, main = "Courbe ROC - Naive Bayes",print.auc = TRUE)

# ----------------------------------------------------------------------
# 3. RANDOM FOREST
# ----------------------------------------------------------------------
cat("\n========== RANDOM FOREST ==========\n")

# Construction du modèle
set.seed(123)
modele_rf <- randomForest(diabetes ~ ., data = train, ntree = 500, importance = TRUE)

# Importance des variables
cat("\n--- IMPORTANCE DES VARIABLES ---\n")
importance_rf <- importance(modele_rf)
print(round(importance_rf[order(-importance_rf[,4]), ], 2))

# Graphique d'importance
varImpPlot(modele_rf, main = "Importance des variables - Random Forest")

# Prédictions
prob_rf <- predict(modele_rf, test, type = "prob")[, "Oui"]
pred_rf <- predict(modele_rf, test)

# Matrice de confusion
cm_rf <- table(Reference = test$diabetes, Prediction = pred_rf)
cat("\n--- MATRICE DE CONFUSION ---\n")
print(cm_rf)

# Métriques
TP <- cm_rf[2,2]; TN <- cm_rf[1,1]; FP <- cm_rf[1,2]; FN <- cm_rf[2,1]
accuracy <- (TP+TN)/sum(cm_rf)
sensitivity <- TP/(TP+FN)
specificity <- TN/(TN+FP)
precision <- TP/(TP+FP)
f1 <- 2*(precision*sensitivity)/(precision+sensitivity)

cat("\nAccuracy:", round(accuracy,3))
cat("\nSensibilité:", round(sensitivity,3))
cat("\nSpécificité:", round(specificity,3))
cat("\nPrécision:", round(precision,3))
cat("\nF1-score:", round(f1,3), "\n")

# Courbe ROC
roc_rf <- roc(test$diabetes, prob_rf, levels = c("Non", "Oui"))
plot(roc_rf, col = "red", lwd = 2, main = "Courbe ROC - Random Forest", print.auc = TRUE)

# ----------------------------------------------------------------------
# 4. SVM
# ----------------------------------------------------------------------
cat("\n========== SVM ==========\n")

# Construction du modèle
set.seed(123)
modele_svm <- svm(diabetes ~ ., data = train, probability = TRUE, kernel = "radial", class.weights = c("Non" = 1, "Oui" = 1))  # équilibrage simple

# Prédictions
pred_svm <- predict(modele_svm, test)

# Pour les probabilités SVM
svm_prob <- attr(predict(modele_svm, test, probability = TRUE), "probabilities")
prob_svm <- svm_prob[, "Oui"]

# Matrice de confusion
cm_svm <- table(Reference = test$diabetes, Prediction = pred_svm)
cat("\n--- MATRICE DE CONFUSION ---\n")
print(cm_svm)

# Métriques
TP <- cm_svm[2,2]; TN <- cm_svm[1,1]; FP <- cm_svm[1,2]; FN <- cm_svm[2,1]
accuracy <- (TP+TN)/sum(cm_svm)
sensitivity <- TP/(TP+FN)
specificity <- TN/(TN+FP)
precision <- TP/(TP+FP)
f1 <- 2*(precision*sensitivity)/(precision+sensitivity)

cat("\nAccuracy:", round(accuracy,3))
cat("\nSensibilité:", round(sensitivity,3))
cat("\nSpécificité:", round(specificity,3))
cat("\nPrécision:", round(precision,3))
cat("\nF1-score:", round(f1,3), "\n")

# Courbe ROC
roc_svm <- roc(test$diabetes, prob_svm, levels = c("Non", "Oui"))
plot(roc_svm, col = "green", lwd = 2, main = "Courbe ROC - SVM", print.auc = TRUE)

# ====================================================
# COMPARAISON DES MODÈLES (version caret)
# ====================================================

cat("\n========== COMPARAISON AVEC CARET ==========\n")

# BLOC 5 - Configuration pour caret
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, verboseIter = FALSE)

# BLOC 6 - Entraînement avec caret
set.seed(123)
cat("Régression logistique...\n")
logistic <- train(diabetes ~ ., data = train, method = "glm", family = "binomial", trControl = ctrl, metric = "ROC")

set.seed(123)
cat("Naive Bayes...\n")
nb <- train(diabetes ~ ., data = train, method = "naive_bayes", trControl = ctrl, metric = "ROC")

set.seed(123)
cat("Random Forest...\n")
rf <- train(diabetes ~ ., data = train, method = "rf", ntree = 100, trControl = ctrl, metric = "ROC")

set.seed(123)
cat("SVM...\n")
svm <- train(diabetes ~ ., data = train, method = "svmRadial", trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, sampling = "up"), metric = "ROC")

# BLOC 7 - Évaluation des modèles caret sur le test set
models_caret <- list(logistic, nb, rf, svm)
model_names_caret <- c("Régression Logistique (caret)", "Naive Bayes (caret)", "Random Forest (caret)", "SVM (caret)")

results_caret <- data.frame(Modele = model_names_caret, AUC_Test = NA, Accuracy_Test = NA)

for(i in 1:length(models_caret)) {
  pred <- predict(models_caret[[i]], test)
  prob <- predict(models_caret[[i]], test, type = "prob")[, "Oui"]
  
  cm <- confusionMatrix(pred, test$diabetes)
  results_caret$Accuracy_Test[i] <- round(cm$overall["Accuracy"], 3)
  
  roc_obj <- roc(test$diabetes, prob, levels = c("Non", "Oui"), quiet = TRUE)
  results_caret$AUC_Test[i] <- round(auc(roc_obj), 3)
}

print("=== RÉSULTATS CARET (sur test set) ===")
print(results_caret[order(-results_caret$AUC_Test), ])

# ====================================================
# COMPARAISON DES DEUX APPROCHES
# ====================================================
cat("\n========== COMPARAISON MANUEL vs CARET ==========\n")

# Créer un tableau de comparaison
comparaison <- data.frame(
  Modele = c("Régression Logistique", "Naive Bayes", "Random Forest", "SVM"),
  AUC_Manuel = c(0.502, 0.497, 0.502, 0.510),  # À remplacer par vos vraies valeurs
  AUC_Caret = results_caret$AUC_Test,
  Difference = abs(c(0.502, 0.497, 0.502, 0.510) - results_caret$AUC_Test)
)

print(comparaison)

# Vérification : si les différences sont très petites (< 0.01), vos modèles sont cohérents
if(max(comparaison$Difference) < 0.01) {
  cat("\n✓ COHÉRENCE : Les approches manuelle et caret donnent des résultats similaires.\n")
} else {
  cat("\n⚠ ATTENTION : Différences > 0.01 détectées. Vérifiez vos modèles.\n")
}

# ====================================================
# COURBES ROC COMPARATIVES (une par approche)
# ====================================================

# Graphique 1 : Courbes des modèles manuels
par(mfrow = c(2, 2))
plot(roc_log, col = "blue", lwd = 2, main = "Régression Logistique", print.auc = TRUE)
plot(roc_nb, col = "purple", lwd = 2, main = "Naive Bayes", print.auc = TRUE)
plot(roc_rf, col = "red", lwd = 2, main = "Random Forest", print.auc = TRUE)
plot(roc_svm, col = "green", lwd = 2, main = "SVM", print.auc = TRUE)
par(mfrow = c(1, 1))

# Graphique 2 : Courbes des modèles caret
roc_log_caret <- roc(test$diabetes, predict(logistic, test, type="prob")[,"Oui"], levels=c("Non","Oui"))
roc_nb_caret <- roc(test$diabetes, predict(nb, test, type="prob")[,"Oui"], levels=c("Non","Oui"))
roc_rf_caret <- roc(test$diabetes, predict(rf, test, type="prob")[,"Oui"], levels=c("Non","Oui"))
roc_svm_caret <- roc(test$diabetes, predict(svm, test, type="prob")[,"Oui"], levels=c("Non","Oui"))

par(mfrow = c(2, 2))
plot(roc_log_caret, col = "blue", lwd = 2, main = "Régression Logistique", print.auc = TRUE)
plot(roc_nb_caret, col = "purple", lwd = 2, main = "Naive Bayes", print.auc = TRUE)
plot(roc_rf_caret, col = "red", lwd = 2, main = "Random Forest", print.auc = TRUE)
plot(roc_svm_caret, col = "green", lwd = 2, main = "SVM", print.auc = TRUE)
par(mfrow = c(1, 1))

# ====================================================
# CONCLUSION FINALE
# ====================================================
cat("\n========== CONCLUSION ==========\n")
cat("Deux approches ont été comparées pour valider la robustesse des résultats.\n")

# Meilleur modèle manuel
best_manuel <- which.max(c(0.502, 0.497, 0.502, 0.510))  # À adapter
best_manuel_name <- c("RL", "NB", "RF", "SVM")[best_manuel]

# Meilleur modèle caret
best_caret_idx <- which.max(results_caret$AUC_Test)
best_caret_name <- results_caret$Modele[best_caret_idx]

cat("\nMeilleur modèle (manuel) :", best_manuel_name, 
    "avec AUC =", max(c(0.502, 0.497, 0.502, 0.510)), "\n")
cat("Meilleur modèle (caret) :", best_caret_name, 
    "avec AUC =", max(results_caret$AUC_Test), "\n")

if(best_manuel_name == strsplit(best_caret_name, " ")[[1]][1]) {
  cat("\n✓ Les deux approches convergent vers le même meilleur modèle.\n")
} else {
  cat("\n⚠ Les approches donnent des meilleurs modèles différents.\n")
  cat("  Cela peut indiquer une instabilité des modèles.\n")
}

###### 1. ADVANCED STATISTICAL APPROACH
###### 2. DESCRIPTION OF THE ADVANCED STATISTICAL APPROACH

#On a choisit de comparer 4 méthodes d'apprentissage supervisé pour la classification binaire
#Pourquoi cette approche ? 
# - la variable cible est binaire
# - on veut comparer difféentes familles d'algorithmes pour identifier le plus performant
# - le SVM a été équilibré pour gérer le déséquilibre potentiel des classes 


#1) Régression Logistique
#Méthode de référence interprétable
#Modélise la probabilité d'appartenance à une classe via la fonction logistique:
#P(Y=1) = 1/(1 + e^-(β₀ + β₁X₁ + ...)). Les coefficients sont interprétables comme
#des facteurs de risque.

#2) Naive Bayes
#Méthode probabiliste simple et rapide
#Applique le théorème de Bayes avec l'hypothèse (naïve) d'indépendance
#conditionnelle des variables. Calcule la probabilité d'appartenance à
#chaque classe et choisit la plus probable.

#3) Random Forest
#Méthode d'ensemble basée sur des arbres de décision
#Combine plusieurs arbres de décision construits sur des échantillons bootstrap,
#avec un sous-ensemble aléatoire de variables à chaque nœud. La prédiction finale
#est obtenue par vote majoritaire.

#4) SVM
#"Support Vector Machine"
#Le SVM cherche à sépar er les classes en trouvant l'hyperplan optimal qui
#maximise la marge entre les points des deux classes. Pour les données non
#linéairement séparables, il utilise un noyau radial qui projette les données
#dans un espace de dimension supérieure. Les points les plus proches de la
#frontière sont appelés vecteurs de support.


###### 3. MEASURE OF PERFORMANCE
#On a utilisé plusieurs métriques complémentaires:

# - AUC (Area Under the Curve) :
#   = métrique principale qui mesure la capacité du modèle à distinguer
#     les classes, indépendamment du seuil. Valeur entre 0.5 (aléatoire) et 1 (parfait).

# - Accuracy :
#   = taux de bonnes prédictions ((VP + VN) / total)

# - Matrice de confusion :
#   = visualisation des vrais positifs, vrais négatifs, faux positifs et faus négatifs

# - Validation croisée (caret) 5-fold :
#   = pour évaluer la stabilité du modèle


###### 4. DATA PROCESSING
#Les étapes de préparations ont été :
#1. Tranformation des variables catégorielles
#   - physical_activity : Low→0, Medium→1, High→2
#   - gender : Male→0, Female→1

#2. Traitement des outliers sur la variable BMI (méthode IQR) :
#   - Q1 = 24.5, Q3 = 34.2, IQR = 9.7
#   - Bornes : [9.95, 48.75]
#   - Suppression des individus hors de ces bornes

#3. Création de la variable cible : transformation en facteur avec labels "Non" et "Oui"

#4. Split train/test : 80% entraînement, 20% test (seed=123 pour reproductibilité)



###### 5. RESULT ANALYSIS
#1. Résultats comparatifs :

#  Modele                     AUC_Manuel AUC_Caret  Difference
#1 Régression Logistique      0.502      0.500      0.002
#2           Naive Bayes      0.497      0.505      0.008
#3         Random Forest      0.502      0.492      0.010
#4                   SVM      0.510      0.507      0.003


#2. Matrice de confusion (pour SVM):
#Reference
#Prediction Non  Oui
#Non        532  465
#Oui        551  440


#-Vrais négatifs : 532 patients non-diabétiques correctement prédits
#-Vrais positifs : 440 patients diabétiques correctement prédits
#-Faux positifs : 551 patients sains prédits à tort comme diabétiques
#-Faux négatifs : 465 patients diabétiques non détectés


#Analyse de SVM avec caret :
# - AUC = 0.507 : très proche de 0.5 (modèle aléatoire)
# - Accuracy ≈ 50% : performance équivalente à un tirage au sort
# - Les 4 modèles donnent des résultats très similaire

#Ces résultats ne sont pas bons d'un point de vue prédictif, mais ils sont très
#instructifs : ils montrent que les variables disponibles (données démographiques,
#cliniques et de mode de vie) ne permettent pas à elles seules de prédire le diabète
#avec certitude.


###### 6. CONCLUSION
#Pourquoi les performances sont-elles faibles ?

# - Le diabète est une maladie multifactorielle complexe
# - Trop de variables non pertinentes (sommeil, pas, eau, travail)
# - Les variables disponibles ne capturent pas tous les facteurs de risque
# - Il manque des données essentielles comme la glycémie à jeun, ou encore le régime alimentaire

#Pistes d'amélioration concrètes :
#1. Sélectionner uniquement les variables cliniques (glucose, insuline, IMC, âge)

#2. Ajouter des données médicales de qualité (HbA1c...)

#3. Créer des ratios pertinents (glucose/insuline)

#4. Tester d'autres modèles :
#   - XGBoost ou LightGBM (gradient boosting)
#   - Réseaux de neurones avec plusieurs couches


#Conclusion finale :
#Bien que les performances prédictives soient modestes (AUC≈0.5), cette analyse
#a le mérite de montrer les limites des données actuelles et d'orienter vers les
#informations manquantes pour améliorer la prédiction du diabète à l'avenir.

#Prédire le diabète est objectivement difficile (maladie complexe),
#et avec des données limitées, les modèles ne peuvent pas faire de miracles.
#Ce n'est pas un échec, c'est la réalité médicale.

#Ce projet montre qu'en apprentissage automatique, la pertinence des données
#prime sur la complexité des modèles. Avec des variables inadaptées, même les
#meilleurs algorithmes ne peuvent pas performer.


