rm(list=ls())

setwd("C:/Users/loryg/OneDrive/Desktop/DATA MINING M/progetto")
dati <- read.csv("training.csv", stringsAsFactors = T)

str(dati)
#skimr::skim(dati)


#dando un'occhiata alle 21 variabili comprendo subito che ce ne sono due che sono inutili
#ID
#pickup_month (tutte le osservazioni fanno riferimento al mese di maggio)
table(dati$pickup_month)

#elimino quindi queste due variabili dal dataset
dati <- subset(dati, select=-c(ID, pickup_month))

#pickup_hour, pickup_week, pickup_doy e pickup wday, sebbene codificate come numeriche, sono
#di fatto delle factor

dati$pickup_hour <- as.factor(dati$pickup_hour)
dati$pickup_week <- as.factor(dati$pickup_week)
dati$pickup_doy <- as.factor(dati$pickup_doy)
dati$pickup_wday <- as.factor(dati$pickup_wday)

#a questo punto decido di dividere il dataset in tre insiemi
#training set: che utilizzerò per stimare i modelli
#validation set: che utilizzerò per stimare l'errore di previsione e scegliere il modello
#test set: che utilizzerò SOLO alla fine per la valutazione finale dell'errore di previsione
n <- nrow(dati)
set.seed(123)
id_train <- sort(sample(1:n, floor(0.5*n), replace=F))
id_no_train <- setdiff(1:n, id_train)

#head(id_train, 20)
#head(id_no_train, 20)

n <- length(id_no_train)
id_valid <- sort(sample(id_no_train, floor(0.5*n), replace=F))
id_test <- setdiff(id_no_train, id_valid)


#verifico di aver fatto tutto correttamente
n <- nrow(dati)
length(id_train)/n
length(id_valid)/n
length(id_test)/n
length(id_train) + length(id_valid) + length(id_test) == n

head(id_train, 20)
head(id_valid, 20)
head(id_test, 20)

#mean(sort(c(id_train, id_valid, id_test)) == 1:243179)

train <- dati[id_train,]
valid <- dati[id_valid,]
test <- dati[id_test,]


#definisco fin da subito funzione di perdita che utilizzerò
mae <- function(val.veri, val.previsti){
  mean(abs(val.veri-val.previsti))
}


#FASE DI PRE PROCESSING
#considero innanzitutto le variabili numeriche
correlazione <- cor(train[,c("length_time","trip_distance", "fare_amount", "tip_amount")])
correlazione
#noto che c'è altissima correlazione tra fare_amount e trip_distance
#decido comunque di tenere in considerazione tutte le variabili

#creo variabile trip_distance_km (per facilità di interpretazione)
train$trip_distance_km <- round(train$trip_distance * 1.60934, 2)
train <- subset(train, select=-trip_distance)


#considero variabili length_time e trip_distance_km
summary(train$length_time)
summary(train$trip_distance_km)

#noto subito che ci sono osservazioni nulle, e ciò non è sensato
#provo ad analizzare
dim(train[train$trip_distance_km==0 & train$length_time==0,])[1]
#nessuna osservazione ha sia tempo che lunghezza nulla

dim(train[train$trip_distance_km==0 | train$length_time==0,])[1]
#192 (=189+3) osservazioni hanno una delle due nulla

ind.0 <- which(train$trip_distance_km==0 | train$length_time==0)

train[ind.0, "fare_amount"]
train[train$trip_distance_km==0, "fare_amount"]
train[train$length_time==0, "fare_amount"]
#non sembra esserci una connessione nemmeno con la variabile fare_amount
#concludo che le osservazioni nulle sono errori di misurazione e decido di escluderle
train <- train[-ind.0,]

#considero ora la variabile length_time
summary(train$length_time)

#osservazioni sopra i 10800 secondi (=3 ore) e sotto i 10 secondi sono outliers
ind <- which(train$length_time<10 | train$length_time>10800)
train[ind, c("length_time", "trip_distance_km", "fare_amount")]
train <- train[-ind,]

#considero ora la variabile trip_distance_km
summary(train$trip_distance_km)
#potrebbero esserci anche qui valori eccessivamente alti o bassi ma decido al momento
#di tenere tutto

#considero ora la variabile fare_amount
summary(train$fare_amount)

#analizzo i valori più bassi
sort(train$fare_amount)
ind <- which(train$fare_amount<2.5)
#osservazione con tariffa 0.01 è un outlier/errore, la elimino
train[ind, c("length_time", "trip_distance_km", "fare_amount")]
train <- train[,-ind]

#analizzo i valori più alti
sort(train$fare_amount, decreasing = T)
ind <- which(train$fare_amount>=80)
train[ind, c("length_time", "trip_distance_km", "fare_amount")]
#a osservazioni con tariffe alte corrispondono tempi e distanze abbastanza elevate
#decido, per il momento, di non escluderle


#a questo punto decido di utilizzare il concetto di distanza di mahalanobis (al quadrato)
#per verificare presenza di outliers multivariati, per poi eventualmente escluderli
#dal training set

#creo un train ridotto, contenente le sole variabili numeriche (ANCHE tip_amount)
train_n <- subset(train, select=c(length_time, trip_distance_km, fare_amount, tip_amount))

#implemento a mano il calcolo della distanza di mahalanobis (al quadrato)
#calcolo vettore delle medie e lo metto in un vettore colonna
xbar <- apply(train_n, 2, mean)
xbar <- matrix(xbar, nrow=4, ncol=1)
xbar

#calcolo matrice varianza e relativa inversa
S <- var(train_n)
invS <- solve(S)

#prova con prima riga
x <- as.numeric(train_n[1,])
x <- matrix(x, nrow=4, ncol=1)
x

t(x - xbar) %*% invS %*% (x - xbar)
rm(x)

#creo funzione e calcolo distanza (al quadrato) per tutte le osservazioni
dist_m <- function(x, xbar, invS){
  x <- matrix(as.numeric(x), ncol=1)
  t(x - xbar) %*% invS %*% (x - xbar)
}

d <- apply(train_n, 1, dist_m, xbar, invS)

#soglia (quantile 0.95 di una chi quadro con p gradi di libertà)
q.95 = qchisq(0.95, df=4)

ind <- which(d>q.95)
#secondo questo criterio 8351 elementi sono outliers

#plot(d, pch=19, cex=0.5)
#abline(h=q.95, col="red", lty=2)


n <- nrow(train_n)
n*0.05
#il numero di outliers che ci si aspetterebbe (assumendo normalità) è 6064

#elimino tutti questi outliers
train <- train[-ind,]

summary(train$length_time)
summary(train$trip_distance_km)
summary(train$fare_amount)
summary(train$tip_amount)
#abbiamo range di valori plausibili

#plot(length_time ~ trip_distance_km, data=train, pch=19, cex=0.5)
#plot(fare_amount ~ trip_distance_km, data=train, pch=19, cex=0.5)
#plot(fare_amount ~ length_time, data=train, pch=19, cex=0.5)
#grafici ottimi

#con anche tip_amount
#plot(tip_amount ~ trip_distance_km, data=train, pch=19, cex=0.5)
#plot(tip_amount ~ fare_amount, data=train, pch=19, cex=0.5)
#plot(tip_amount ~ length_time, data=train, pch=19, cex=0.5)
#qua andamento non sembra essere lineare


#analisi variabile risposta
summary(train$tip_amount)

boxplot(train$tip_amount)
hist(train$tip_amount)

#sebbene siano stati rimossi tutti gli outliers multivariati, secondo il criterio della
#distanza di mahalanobis, la variabile risposta sembra averne ancora

#con trasformazione logaritmica
boxplot(log(train$tip_amount))
hist(log(train$tip_amount))
#non molto belli. trasformazione logaritmica non è utile


#decido di considerare il rapporto tra mancia e tariffa della corsa
tip_perc <- (train$tip_amount/train$fare_amount)*100

summary(tip_perc)
boxplot(tip_perc)
#ci sono outliers sia in eccesso, sia in difetto. 
#si decide di trascurare le osservazioni la cui percentuale di mancia sul totale
#è superiore al 60% oppure inferiore al 10%

ind <- which(tip_perc>60 | tip_perc<10)
train <- train[-ind,]

#altre variabili
#variabili di "tempo"
boxplot(tip_amount ~ pickup_hour, data=train)
tapply(train$tip_amount, train$pickup_hour, mean) |> sort(decreasing=T)

boxplot(tip_amount ~ pickup_week, data=train)
tapply(train$tip_amount, train$pickup_week, mean) |> sort(decreasing=T)

boxplot(tip_amount ~ pickup_doy, data=train)
tapply(train$tip_amount, train$pickup_doy, mean) |> sort(decreasing=T)

boxplot(tip_amount ~ pickup_wday, data=train)
tapply(train$tip_amount, train$pickup_wday, mean) |> sort(decreasing=T)

#elimino pickup_week e pickup_doy
train <- subset(train, select=-c(pickup_week, pickup_doy))


#variabili geografiche

#trascuro le variabili pair, pickup_NTACode e dropoff_NTACode in quanto presentano
#troppi livelli, la maggior parte dei quali è sottorappresentata
train <- subset(train, select=-c(pair, pickup_NTACode, dropoff_NTACode))

#mi concentro su pickup_BoroCode e dropoff_BoroCode
table(train$pickup_BoroCode) |> prop.table() |> round(3)*100
table(train$dropoff_BoroCode) |> prop.table() |> round(3)*100

#bronx e staten island sono sottorappresentati, decido di codificarli come other
levels_no <- c("Bronx", "Staten Island")

#pickup_BoroCode
train$pickup_BoroCode <- ifelse(train$pickup_BoroCode %in% levels_no, "Other", train$pickup_BoroCode)
train$pickup_BoroCode <- as.factor(train$pickup_BoroCode)
table(train$pickup_BoroCode)

#ricodifico i livelli
levels(train$pickup_BoroCode) <- c("Brooklyn", "Manhattan", "Queens", "Other")
table(train$pickup_BoroCode)


#dropoff_BoroCode
train$dropoff_BoroCode <- ifelse(train$dropoff_BoroCode %in% levels_no, "Other", train$dropoff_BoroCode)
train$dropoff_BoroCode <- as.factor(train$dropoff_BoroCode)
table(train$dropoff_BoroCode)

#ricodifico i livelli
levels(train$dropoff_BoroCode) <- c("Brooklyn", "Manhattan", "Queens", "Other")
table(train$dropoff_BoroCode)

#elimino anche latitudine e longitudine
train <- subset(train, select=-c(pickup_latitude, pickup_longitude,
                                 dropoff_latitude, dropoff_longitude))

#variabile passenger_count
#creo una dummy che discrimini tra un solo passeggero e due o più passeggeri
train$passengers <- 0
train$passengers[train$passenger_count!=1] <- 1
train$passengers <- as.factor(train$passengers)
levels(train$passengers) <- c("one", "two or more")
#head(train[, c("passenger_count", "passengers")])

#rimuovo vecchia variabile
train <- subset(train, select=-passenger_count)


#STIMA DEI MODELLI e CONFRONTO

#occorre prima "sistemare" il validation set creando le nuove variabili ed eliminando
#quelle trascurate in quest'analisi
valid <- dati[id_valid, ]
valid <- subset(valid, select=-c(pickup_week, pickup_doy, pickup_NTACode, 
                                 dropoff_NTACode, pair))

#converto trip_distance in trip_distance_km
valid$trip_distance_km <- round(valid$trip_distance * 1.60934, 2)
valid <- subset(valid, select=-trip_distance)

#ricodifico variabile passenger_count
valid$passengers <- 0
valid$passengers[valid$passenger_count!=1] <- 1
valid$passengers <- as.factor(valid$passengers)
levels(valid$passengers) <- c("one", "two or more")

valid <- subset(valid, select=-passenger_count)

#elimino anche variabili relative a latitudine e longitudine
valid <- subset(valid, select=-c(pickup_longitude, pickup_latitude,
                                 dropoff_longitude, dropoff_latitude))

#sistemo i livelli di pickup_BoroCode e dropoff_BoroCode
table(valid$pickup_BoroCode) |> prop.table() |> round(3)*100
table(valid$dropoff_BoroCode) |> prop.table() |> round(3)*100

#pickup_BoroCode
valid$pickup_BoroCode <- ifelse(valid$pickup_BoroCode %in% levels_no, "Other", valid$pickup_BoroCode)
valid$pickup_BoroCode <- as.factor(valid$pickup_BoroCode)
table(valid$pickup_BoroCode)

#ricodifico i livelli
levels(valid$pickup_BoroCode) <- c("Brooklyn", "Manhattan", "Queens", "Other")
table(valid$pickup_BoroCode)

#dropoff_BoroCode
valid$dropoff_BoroCode <- ifelse(valid$dropoff_BoroCode %in% levels_no, "Other", valid$dropoff_BoroCode)
valid$dropoff_BoroCode <- as.factor(valid$dropoff_BoroCode)
table(valid$dropoff_BoroCode)

#ricodifico i livelli
levels(valid$dropoff_BoroCode) <- c("Brooklyn", "Manhattan", "Queens", "Other")
table(valid$dropoff_BoroCode)


#MODELLI LINEARI

#modello lineare semplice con solo fare_amount
mod1 <- lm(tip_amount ~ fare_amount, data=train)
summary(mod1)
y_hat_1 <- predict(mod1, valid)
mae(valid$tip_amount, y_hat_1)
#0.6134

#modello lineare semplice con le sole variabili numeriche
mod2 <- lm(tip_amount ~ fare_amount + length_time + trip_distance_km, data=train)
summary(mod2)
#trip_distance_km non signufucativa
y_hat_2 <- predict(mod2, valid)
mae(valid$tip_amount, y_hat_2)
#0.6334

#modello lineare completo
mod3 <- lm(tip_amount ~ ., data=train)
summary(mod3)

y_hat_3 <- predict(mod3, valid)
mae(valid$tip_amount, y_hat_3)
#0.624063

dim(model.matrix(tip_amount~., data=train)[,-1])
#40 variabili

#RIDGE
library(glmnet)
X_shrinkage <- model.matrix(tip_amount ~ ., data=train)[,-1]
y_shrinkage <- train$tip_amount

#scelgo una griglia di valori possibili per lambda
lambda_grid <- exp(seq(-7, 7, length = 500))

m_ridge <- glmnet(X_shrinkage, y_shrinkage, alpha = 0, lambda=lambda_grid)

par(mfrow = c(1, 1))
plot(m_ridge, xvar = "lambda")

#scelgo lambda via cross validation
set.seed(123)
ridge_cv <- cv.glmnet(X_shrinkage, y_shrinkage, alpha = 0, lambda=lambda_grid)
par(mfrow = c(1, 1))
plot(ridge_cv)

ridge_cv$lambda.min
ridge_cv$lambda.1se

#previsione con lambda.1se
y_hat_ridge <- predict(ridge_cv, newx=model.matrix(tip_amount ~ ., data=valid)[,-1])
mae(valid$tip_amount, y_hat_ridge)
#0.6440

#equivalente a fare
#y_hat_ridge <- predict(m_ridge, newx=model.matrix(tip_amount ~ ., data=valid)[,-1],
#                       s=ridge_cv$lambda.1se)
#mae(valid$tip_amount, y_hat_ridge)

#previsione con lambda.min
y_hat_ridge <- predict(m_ridge, newx=model.matrix(tip_amount ~ ., data=valid)[,-1],
                        s=ridge_cv$lambda.min)
mae(valid$tip_amount, y_hat_ridge)
#0.6058

cbind(coef(mod3), coef(m_ridge, s=ridge_cv$lambda.min)) |> round(3)
#lambda ottimale è piccolo, coefficienti stimati sono molto simili a quelli ottenuti
#con la regressione semplice

#LASSO
#utilizzo stessa griglia di prima
m_lasso <- glmnet(X_shrinkage, y_shrinkage, alpha = 1, lambda = lambda_grid)
plot(m_lasso, xvar = "lambda")

set.seed(123)
lasso_cv <- cv.glmnet(X_shrinkage, y_shrinkage, alpha=1, lambda = lambda_grid)
plot(lasso_cv)

lasso_cv$lambda.min
lasso_cv$lambda.1se

#previsione con lambda.1se
y_hat_lasso <- predict(lasso_cv, newx = model.matrix(tip_amount ~ ., data=valid)[,-1])
mae(valid$tip_amount, y_hat_lasso)
#0.6078

#equivalente a
#y_hat_lasso <- predict(m_lasso, newx = model.matrix(tip_amount ~ ., data=valid)[,-1],
#                       s=lasso_cv$lambda.1se)
#mae(valid$tip_amount, y_hat_lasso)
#0.6079 


#previsione con lambda.min
y_hat_lasso2 <- predict(m_lasso, newx = model.matrix(tip_amount ~ ., data=valid)[,-1],
                        s=lasso_cv$lambda.min)
mae(valid$tip_amount, y_hat_lasso2)
#0.6056

coef(m_lasso, s=lasso_cv$lambda.min) #shrinkage "leggero", pochi coefficienti sono nulli
coef(m_lasso, s=lasso_cv$lambda.1se) #in questo caso shrinkage è maggiore, più coef nulli



#ELASTIC NET (con alpha=0.5)
m_en <- glmnet(X_shrinkage, y_shrinkage, alpha=0.5, lambda = lambda_grid)
plot(m_en, xvar = "lambda")

set.seed(123)
en_cv <- cv.glmnet(X_shrinkage, y_shrinkage, alpha=0.5, lambda = lambda_grid)
plot(en_cv)

en_cv$lambda.1se
en_cv$lambda.min

#previsione con lambda.1se
y_hat_en <- predict(en_cv, newx = model.matrix(tip_amount ~ ., data=valid)[,-1])
mae(valid$tip_amount, y_hat_en)
#0.6318

#previsione con lambda.min
y_hat_en2 <- predict(m_en, newx = model.matrix(tip_amount ~ ., data=valid)[,-1],
                     s=en_cv$lambda.min)
mae(valid$tip_amount, y_hat_en2)
#0.6057



#STIMA DEL LASSO SU TRAINING E VALIDATION SET
train <- dati[id_train,]
valid <- dati[id_valid,]
test <- dati[id_test,]

#considero come nuovo training set l'insieme di train e valid
id_new_train <- setdiff(1:nrow(dati), id_test)

#length(id_new_train) + length(id_test)
#head(id_new_train, 20)
#head(id_test, 20)

train <- dati[id_new_train, ]

#a questo punto rifaccio tutto il pre processing sul nuovo train
#creo variabile trip_distance_km (per facilità di interpretazione)
train$trip_distance_km <- round(train$trip_distance * 1.60934, 2)
train <- subset(train, select=-trip_distance)

#considero variabili length_time e trip_distance_km
summary(train$length_time)
summary(train$trip_distance_km)

#osservazioni con length_time e trip_distance_km nulle
ind.0 <- which(train$trip_distance_km==0 | train$length_time==0)
#le elimino
train <- train[-ind.0,]

#considero ora la variabile length_time
summary(train$length_time)

#osservazioni sopra i 10800 secondi (=3 ore) e sotto i 10 secondi sono outliers
ind <- which(train$length_time<10 | train$length_time>10800)
train[ind, c("length_time", "trip_distance_km", "fare_amount")]
train <- train[-ind,]

#considero ora la variabile fare_amount
summary(train$fare_amount)

ind <- which(train$fare_amount<2.5)
#elimino osservazioni con fare_amount inferiore a 2.5 dollari
train[ind, c("length_time", "trip_distance_km", "fare_amount")]
train <- train[,-ind]


#a questo punto decido di utilizzare il concetto di distanza di mahalanobis 
#per verificare presenza di outliers multivariati, per poi eventualmente escluderli
#dal training set

#creo un train ridotto, contenente le sole variabili numeriche (ANCHE tip_amount)
train_n <- subset(train, select=c(length_time, trip_distance_km, fare_amount, tip_amount))

#implemento a mano il calcolo della distanza di mahalanobis (al quadrato)
#calcolo vettore delle medie e lo metto in un vettore colonna
xbar <- apply(train_n, 2, mean)
xbar <- matrix(xbar, nrow=4, ncol=1)
xbar

#calcolo matrice varianza e relativa inversa
S <- var(train_n)
invS <- solve(S)


d <- apply(train_n, 1, dist_m, xbar, invS)

#soglia (quantile 0.95 di una chi quadro con p gradi di libertà)
q.95 = qchisq(0.95, df=4)

ind <- which(d>q.95)
#plot(d, pch=19, cex=0.5)
#abline(h=q.95, col="red", lty=2)
#secondo questo criterio 11965 elementi sono outliers

n <- nrow(train_n)
n*0.05
#il numero di outliers che ci si aspetterebbe (assumendo normalità) è 9095

#elimino tutti questi outliers
train <- train[-ind,]

tip_perc <- (train$tip_amount/train$fare_amount)*100
ind <- which(tip_perc>60 | tip_perc<10)
train <- train[-ind,]

#altre variabili
#variabili di "tempo"
#elimino pickup_week e pickup_doy
train <- subset(train, select=-c(pickup_week, pickup_doy))

#variabili geografiche
#trascuro le variabili pair, pickup_NTACode e dropoff_NTACode in quanto presentano
#troppi livelli, la maggior parte dei quali è sottorappresentata
train <- subset(train, select=-c(pair, pickup_NTACode, dropoff_NTACode))

#mi concentro su pickup_BoroCode e dropoff_BoroCode
table(train$pickup_BoroCode) |> prop.table() |> round(3)*100
table(train$dropoff_BoroCode) |> prop.table() |> round(3)*100

#bronx e staten island sono sottorappresentati, decido di codificarli come other
levels_no <- c("Bronx", "Staten Island")

#pickup_BoroCode
train$pickup_BoroCode <- ifelse(train$pickup_BoroCode %in% levels_no, "Other", train$pickup_BoroCode)
train$pickup_BoroCode <- as.factor(train$pickup_BoroCode)
table(train$pickup_BoroCode)

#ricodifico i livelli
levels(train$pickup_BoroCode) <- c("Brooklyn", "Manhattan", "Queens", "Other")
table(train$pickup_BoroCode)

#dropoff_BoroCode
train$dropoff_BoroCode <- ifelse(train$dropoff_BoroCode %in% levels_no, "Other", train$dropoff_BoroCode)
train$dropoff_BoroCode <- as.factor(train$dropoff_BoroCode)
table(train$dropoff_BoroCode)

#ricodifico i livelli
levels(train$dropoff_BoroCode) <- c("Brooklyn", "Manhattan", "Queens", "Other")
table(train$dropoff_BoroCode)

#elimino le variabili relative a latitudine e longitudine
train <- subset(train, select=-c(pickup_latitude, pickup_longitude,
                                 dropoff_latitude, dropoff_longitude))


#variabile passenger_count
#creo una dummy che discrimini tra un solo passeggero e due o più passeggeri
train$passengers <- 0
train$passengers[train$passenger_count!=1] <- 1
train$passengers <- as.factor(train$passengers)
levels(train$passengers) <- c("one", "two or more")
#head(train[, c("passenger_count", "passengers")])

#rimuovo vecchia variabile
train <- subset(train, select=-passenger_count)


#considero ora il mio TEST set e sistemo le variabili
test <- subset(test, select=-c(pickup_week, pickup_doy, pickup_NTACode, 
                                 dropoff_NTACode, pair))

#converto trip_distance in trip_distance_km
test$trip_distance_km <- round(test$trip_distance * 1.60934, 2)
test <- subset(test, select=-trip_distance)

#ricodifico variabile passenger_count
test$passengers <- 0
test$passengers[test$passenger_count!=1] <- 1
test$passengers <- as.factor(test$passengers)
levels(test$passengers) <- c("one", "two or more")

test <- subset(test, select=-passenger_count)

#elimino anche variabili relative a latitudine e longitudine
test <- subset(test, select=-c(pickup_longitude, pickup_latitude,
                                 dropoff_longitude, dropoff_latitude))

#sistemo i livelli di pickup_BoroCode e dropoff_BoroCode
table(test$pickup_BoroCode) |> prop.table() |> round(3)*100
table(test$dropoff_BoroCode) |> prop.table() |> round(3)*100

#pickup_BoroCode
test$pickup_BoroCode <- ifelse(test$pickup_BoroCode %in% levels_no, "Other", test$pickup_BoroCode)
test$pickup_BoroCode <- as.factor(test$pickup_BoroCode)
table(test$pickup_BoroCode)

#ricodifico i livelli
levels(test$pickup_BoroCode) <- c("Brooklyn", "Manhattan", "Queens", "Other")
table(test$pickup_BoroCode)

#dropoff_BoroCode
test$dropoff_BoroCode <- ifelse(test$dropoff_BoroCode %in% levels_no, "Other", test$dropoff_BoroCode)
test$dropoff_BoroCode <- as.factor(test$dropoff_BoroCode)
table(test$dropoff_BoroCode)

#ricodifico i livelli
levels(test$dropoff_BoroCode) <- c("Brooklyn", "Manhattan", "Queens", "Other")
table(test$dropoff_BoroCode)



#STIMA DEL LASSO
library(glmnet)
X_shrinkage <- model.matrix(tip_amount ~ ., data=train)[,-1]
y_shrinkage <- train$tip_amount

m_lasso <- glmnet(X_shrinkage, y_shrinkage, alpha = 1, lambda = lambda_grid)
plot(m_lasso, xvar = "lambda")

set.seed(123)
lasso_cv <- cv.glmnet(X_shrinkage, y_shrinkage, alpha=1, lambda = lambda_grid)
plot(lasso_cv)

lasso_cv$lambda.min
lasso_cv$lambda.1se

#previsione con lambda.1se
y_hat_lasso <- predict(lasso_cv, newx = model.matrix(tip_amount ~ ., data=test)[,-1])
mae(test$tip_amount, y_hat_lasso)
#0.6051

#previsione con lambda.min
y_hat_lasso2 <- predict(m_lasso, newx = model.matrix(tip_amount ~ ., data=test)[,-1],
                        s=lasso_cv$lambda.min)
mae(test$tip_amount, y_hat_lasso2)
#0.6030




#STIMA DEL MODELLO SU TUTTO L'INSIEME DI DATI A DISPOSIZIONE E PREVISIONI FINALI
#a questo punto considero come training l'intero insieme di partenza


train <- dati


#a questo punto rifaccio tutto il pre processing sul train finale
#creo variabile trip_distance_km (per facilità di interpretazione)
train$trip_distance_km <- round(train$trip_distance * 1.60934, 2)
train <- subset(train, select=-trip_distance)

#considero variabili length_time e trip_distance_km
summary(train$length_time)
summary(train$trip_distance_km)

#osservazioni nulle
ind.0 <- which(train$trip_distance_km==0 | train$length_time==0)
#le elimino
train <- train[-ind.0,]

#considero ora la variabile length_time
summary(train$length_time)
#osservazioni sopra i 10800 secondi (=3 ore) e sotto i 10 secondi sono outliers

ind <- which(train$length_time<10 | train$length_time>10800)
train[ind, c("length_time", "trip_distance_km", "fare_amount")]
train <- train[-ind,]

#considero ora la variabile fare_amount
summary(train$fare_amount)

ind <- which(train$fare_amount<2.5)
#osservazione con tariffa 0.01 è un outlier/errore
train[ind, c("length_time", "trip_distance_km", "fare_amount")]
train <- train[,-ind]

#a questo punto decido di utilizzare il concetto di distanza di mahalanobis 
#per verificare presenza di outliers multivariati, per poi eventualmente escluderli
#dal training set

#creo un train ridotto, contenente le sole variabili numeriche (ANCHE tip_amount)
train_n <- subset(train, select=c(length_time, trip_distance_km, fare_amount, tip_amount))

#implemento a mano il calcolo della distanza di mahalanobis (al quadrato)
#calcolo vettore delle medie e lo metto in un vettore colonna
xbar <- apply(train_n, 2, mean)
xbar <- matrix(xbar, nrow=4, ncol=1)
xbar

#calcolo matrice varianza e relativa inversa
S <- var(train_n)
invS <- solve(S)


d <- apply(train_n, 1, dist_m, xbar, invS)

#soglia (quantile 0.95 di una chi quadro con p gradi di libertà)
q.95 = qchisq(0.95, df=4)

ind <- which(d>q.95)
#plot(d, pch=19, cex=0.5)
#abline(h=q.95, col="red", lty=2)
#secondo questo criterio 15897 elementi sono outliers

n <- nrow(train_n)
n*0.05
#il numero di outliers che ci si aspetterebbe (assumendo normalità) è 12126

#elimino tutti questi outliers
train <- train[-ind,]

#elimino osservazioni con percentuale mancia troppo bassa o troppo alta
tip_perc <- (train$tip_amount/train$fare_amount)*100
ind <- which(tip_perc>60 | tip_perc<10)
train <- train[-ind,]

#altre variabili
#variabili di "tempo"
#elimino pickup_week e pickup_doy
train <- subset(train, select=-c(pickup_week, pickup_doy))

#variabili geografiche
#trascuro le variabili pair, pickup_NTACode e dropoff_NTACode in quanto presentano
#troppi livelli, la maggior parte dei quali è sottorappresentata
train <- subset(train, select=-c(pair, pickup_NTACode, dropoff_NTACode))

#mi concentro su pickup_BoroCode e dropoff_BoroCode
table(train$pickup_BoroCode) |> prop.table() |> round(3)*100
table(train$dropoff_BoroCode) |> prop.table() |> round(3)*100

#bronx e staten island sono sottorappresentati, decido di codificarli come other
levels_no <- c("Bronx", "Staten Island")

#pickup_BoroCode
train$pickup_BoroCode <- ifelse(train$pickup_BoroCode %in% levels_no, "Other", train$pickup_BoroCode)
train$pickup_BoroCode <- as.factor(train$pickup_BoroCode)
table(train$pickup_BoroCode)

#ricodifico i livelli
levels(train$pickup_BoroCode) <- c("Brooklyn", "Manhattan", "Queens", "Other")
table(train$pickup_BoroCode)

#dropoff_BoroCode
train$dropoff_BoroCode <- ifelse(train$dropoff_BoroCode %in% levels_no, "Other", train$dropoff_BoroCode)
train$dropoff_BoroCode <- as.factor(train$dropoff_BoroCode)
table(train$dropoff_BoroCode)

#ricodifico i livelli
levels(train$dropoff_BoroCode) <- c("Brooklyn", "Manhattan", "Queens", "Other")
table(train$dropoff_BoroCode)


#in questa consegna non effettuo il controllo relativo a latitudine e longitudine
#ed elimino direttamente le variabili
train <- subset(train, select=-c(pickup_latitude, pickup_longitude,
                                 dropoff_latitude, dropoff_longitude))


#variabile passenger_count
#creo una dummy che discrimini tra un solo passeggero e due o più passeggeri
train$passengers <- 0
train$passengers[train$passenger_count!=1] <- 1
train$passengers <- as.factor(train$passengers)
levels(train$passengers) <- c("one", "two or more")
#head(train[, c("passenger_count", "passengers")])

#rimuovo vecchia variabile
train <- subset(train, select=-passenger_count)


#considero ora il TEST SET FINALE e sistemo le variabili
test <- read.csv("test.csv", stringsAsFactors = T)

#elimino variabili id e pickup_month e trasformo in factor le variabili "di tempo"
test <- subset(test, select=-c(ID, pickup_month))

test$pickup_hour <- as.factor(test$pickup_hour)
test$pickup_week <- as.factor(test$pickup_week)
test$pickup_doy <- as.factor(test$pickup_doy)
test$pickup_wday <- as.factor(test$pickup_wday)

test <- subset(test, select=-c(pickup_week, pickup_doy, pickup_NTACode, 
                               dropoff_NTACode, pair))

#converto trip_distance in trip_distance_km
test$trip_distance_km <- round(test$trip_distance * 1.60934, 2)
test <- subset(test, select=-trip_distance)

#ricodifico variabile passenger_count
test$passengers <- 0
test$passengers[test$passenger_count!=1] <- 1
test$passengers <- as.factor(test$passengers)
levels(test$passengers) <- c("one", "two or more")

test <- subset(test, select=-passenger_count)

#elimino anche variabili relative a latitudine e longitudine
test <- subset(test, select=-c(pickup_longitude, pickup_latitude,
                               dropoff_longitude, dropoff_latitude))

#sistemo i livelli di pickup_BoroCode e dropoff_BoroCode
table(test$pickup_BoroCode) |> prop.table() |> round(3)*100
table(test$dropoff_BoroCode) |> prop.table() |> round(3)*100

#pickup_BoroCode
test$pickup_BoroCode <- ifelse(test$pickup_BoroCode %in% levels_no, "Other", test$pickup_BoroCode)
test$pickup_BoroCode <- as.factor(test$pickup_BoroCode)
table(test$pickup_BoroCode)

#ricodifico i livelli
levels(test$pickup_BoroCode) <- c("Brooklyn", "Manhattan", "Queens", "Other")
table(test$pickup_BoroCode)

#dropoff_BoroCode
test$dropoff_BoroCode <- ifelse(test$dropoff_BoroCode %in% levels_no, "Other", test$dropoff_BoroCode)
test$dropoff_BoroCode <- as.factor(test$dropoff_BoroCode)
table(test$dropoff_BoroCode)

#ricodifico i livelli
levels(test$dropoff_BoroCode) <- c("Brooklyn", "Manhattan", "Queens", "Other")
table(test$dropoff_BoroCode)

#stima del LASSO SUL TRAINING COMPLETO
library(glmnet)
X_shrinkage <- model.matrix(tip_amount ~ ., data=train)[,-1]
y_shrinkage <- train$tip_amount

m_lasso <- glmnet(X_shrinkage, y_shrinkage, alpha = 1, lambda = lambda_grid)
plot(m_lasso, xvar = "lambda")

set.seed(123)
lasso_cv <- cv.glmnet(X_shrinkage, y_shrinkage, alpha=1, lambda = lambda_grid)
plot(lasso_cv)

lasso_cv$lambda.min
lasso_cv$lambda.1se

y_hat_lasso <- predict(lasso_cv, newx = model.matrix( ~ ., data=test)[,-1])

#previsioni con lambda.min
y_hat_lasso2 <- predict(m_lasso, newx = model.matrix( ~ ., data=test)[,-1],
                        s=lasso_cv$lambda.min)
#mean(y_hat_lasso2)

prova2 <- data.frame(
  ID=1:nrow(test),
  prediction=y_hat_lasso2[,1]
)

write.csv(prova2, "submission_Giudici.csv", row.names = FALSE)
