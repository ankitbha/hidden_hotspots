#/////////////////////////////////////////////////////////////
#### 2021-03-18 Data May2018-May2020, ktgt tr fits v0.7.2 ####
#/////////////////////////////////////////////////////////////

# training-test set split: rough 80%-20% with training data up to 2019-10-31
#                          23:00 and test data starting on 2019-11-01 00:00.
#                          cf. emails with Shiva 16-18 Mar 2021

# kt = Kaiterra sensors
# gt = government fixed monitors, CPCB/DPCC/IMD

rm(list=ls())
paf <- '~'
setwd(paf)

library(splines) # for B-spline basis
library(MASS) # for mvrnorm
library(TMB)
library(viridisLite) # better color palette than rainbow
library(rgdal) # for lat/lon conversion to utm
library(OpenStreetMap) # for map downloading and plotting

# compile('STHM/STHM.cpp') # run only once per system
dyn.load(dynlib('STHM/STHM')) # v0.7.2 # load for every new R session


colgradient <- function(n,alpha=1){ # based on heat.colors
  # require(viridis)
  if ((n <- as.integer(n[1L])) > 0) {
    return(viridis(n=n,begin=0,end=1,alpha=alpha,direction=1,option='C'))
  } else {
    return(character())
  }
}
colgrad <- function(x,n=100,alpha=1,lbub=NULL){
  colvec <- colgradient(n,alpha)
  if (is.null(lbub)){lbub <- range(x)}
  x.ind <- trunc((x-lbub[1])/(lbub[2]-lbub[1])*(n-1)+1)
  return(colvec[x.ind])
}
# plot(1:100,col=colgradient(100),pch=19,cex=2) # test colgradient
# plot(1:100,col=colgrad(rnorm(100)),pch=19,cex=2) # test colgrad

# colvec <- c('#ff0f0f','#1b0fff','#00b822','#ffe81a','#ff1abe',
#             '#00c3d1','#cc7400','#80f7ff','#ff950a','#9c00eb')


### // load padded csv ----
loc <- read.csv(file='locations.csv',header=T)
trdat <- read.csv(file='pm25_training.csv',header=T) # training data only
trdat$ts <- as.POSIXct(strptime(trdat$ts,
                                format='%Y-%m-%d %H:%M',
                                tz="Asia/Kolkata"))
str(loc) # 28 kt sensors, 32 gt monitors = 60 loc
str(trdat)


### // basics of design ----
nT <- dim(trdat)[1]
nS <- dim(trdat)[2]-1 # ts as 1st col, take kt and gt together
n <- nS*nT

indlist.s <- vector(mode='list',length=nS) # indices when looping over space
for (i in 1:nS){ # loop over locations
  indlist.s[[i]] <- 1:nT + nT*(i-1)
}
str(indlist.s,1)

indlist.t <- vector(mode='list',length=nT) # indices when looping over time
for (t in 1:nT){ # loop over time points
  indlist.t[[t]] <- (0:(nS-1))*nT + t
}
str(indlist.t,1)


### // project lat/lon as UTM, dist mat ----
loc.latlon <- SpatialPoints(loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))

proj.string <- "+proj=utm +zone=43 +ellps=WGS84 +north +units=km"
# ^ Delhi = UTM zone 43R
coord.utm <- spTransform(loc.latlon, CRS(proj.string))
# ^ re-project, ignore warnings due to deprecated PROJ4
loc$utmx <- coord.utm@coords[,1]
loc$utmy <- coord.utm@coords[,2]
str(loc)

distmat <- matrix(NA_real_,nS,nS) # Euclidean dist on plane after WGS84 proj
for (i in 1:nS){
  distmat[i,i] <- 0
  j <- 1
  while (j<i){
    distmat[i,j] <- sqrt((loc$utmx[i]-loc$utmx[j])^2 +
                           + (loc$utmy[i]-loc$utmy[j])^2)
    distmat[j,i] <- distmat[i,j] # symmetry
    j <- j+1
  }
}
distmat[1:5,1:5] # in km according to proj
summary(distmat[upper.tri(distmat,diag=T)])


### // set up quad B-spline bases for two nested seasonality resolutions ----
# res 1 = daily period, 24 hours within day
# res 2 = weekly period, 7 days within week

# nb bases = J, nb free bases = J-3
# degree=2: nb knots = N = J-1, nb intervals = N-1 = J-2

n.period1 <- ceiling(as.numeric(diff(range(trdat$ts))))
# ^ nb days within tw, 549*24 = nT
n.period2 <- ceiling(n.period1/7) # n.period1/7
# ^ nb weeks within tw, rounded upwards to then truncate excess days

J1 <- 5 # 6 # min=5
kn1 <- (0:(J1-2))/(J1-2) # equally-spaced fixed knots within year

J2 <- 5 # 6 # min=5
kn2 <- (0:(J2-2))/(J2-2) # equally-spaced fixed knots within year

sb.grid.res1 <- seq(0,1,length.out=24+1) # day=[0,1], hourly within day
sb.res1 <- bs(sb.grid.res1,degree=2,knots=kn1,Boundary.knots=c(0,1))
sb.res1 <- sb.res1[-(24+1),-dim(sb.res1)[2]] # extra col because all knots in kn
head(sb.res1)
tail(sb.res1) # last row must mirror 2nd row, so that last+1 row mirrors 1st row
Bmat1 <- rep(1,n.period1*nS)%x%sb.res1 # repeat daily cycle
dim(Bmat1) # c(n,J1)

sb.grid.res2 <- seq(0,1,length.out=7*24+1) # week=[0,1], daily within week
sb.res2 <- bs(sb.grid.res2,degree=2,knots=kn2,Boundary.knots=c(0,1))
sb.res2 <- sb.res2[-(7*24+1),-dim(sb.res2)[2]] # extra col because all knots in kn
head(sb.res2)
tail(sb.res2) # last row must mirror 2nd row, so that last+1 row mirrors 1st row
Bmat2 <- rep(1,n.period2*nS)%x%sb.res2 # repeat weekly cycle
# ^ excess days due to n.period2 rounded upwards, simply truncate last week
Bmat2 <- Bmat2[1:n,]
dim(Bmat2) # c(n,J2)


### // prepare data inputs ----
summary(paste0('pm25_',unique(loc$id))==dimnames(trdat)[[2]][-1])
# ^ column order in trdat identical to row order in loc

rawobs <- as.numeric(as.matrix(trdat[,-1]))
sum(rawobs==0,na.rm=T) # ok, only 5 actual zeros, set to NAs for safety
rawobs[rawobs==0] <- NA

yvec <- as.numeric(log(rawobs)) # log-transform PM2.5 response
# ^ stack by col: space = outer loop, time = inner loop

hist(yvec)
# ^ mostly symmetric although tiny left skewness

obsind <- !is.na(yvec)
table(obsind)/n # 28.4% NAs in tr data

loctype <- c(rep(1L,nT*sum(loc$source=='kt')),
             rep(2L,nT*sum(loc$source=='gt')))


### // fit1: both loc types, fix link to identity for both loc types ----
datalist <- list(
  'obsvec'=yvec,
  'obsind'=as.integer(obsind),
  'loctype'=loctype,
  'Zmat1'=as.matrix(rep(1,n)), # intercept only
  'Zmat2'=as.matrix(rep(1,n)), # intercept only
  'Bmat1'=Bmat1,
  'Bmat2'=Bmat2,
  'kn1'=kn1,
  'kn2'=kn2,
  'distmat'=distmat
)
parlist <- list(
  'beta1'=c(0), # intercept only
  'beta2'=c(0), # intercept only
  'alpha1'=rep(0,J1-3), # 3 constraints
  'alpha2'=rep(0,J2-3), # 3 constraints
  'gamma1'=c(1,0), # set for id link
  'gamma2'=c(1,0), # set for id link
  'log_sigmaeps1'=0,
  'log_sigmaeps2'=0,
  't_phi'=1, # help in right direction, positive auto-cor
  'log_gamma'=0,
  'log_sigmadelta'=0,
  'X'=rep(0,n)
)

system.time(obj1 <- MakeADFun(data=datalist,
                              parameters=parlist,
                              map=list('gamma1'=factor(rep(NA,2)),
                                       'gamma2'=factor(rep(NA,2))),
                              # ^ fix gamma=c(0,1) <=> id link
                              random=c('X'),
                              DLL="STHM",silent=T))
# ^ 90 s | MBP13 nS=60, nT=13176, fixed id link

system.time(obj1$fn())
# ^ 798 s = 14 min | MBP13 nS=60, nT=13176, fixed id link
system.time(obj1$gr())
# ^ 63 s | MBP13 nS=60, nT=13176, fixed id link

system.time(opt1 <- nlminb(start=obj1$par,obj=obj1$fn,gr=obj1$gr,
                           control=list(eval.max=500,iter.max=500)))
# ^ 5983 s = 1h40 | MBP13 nS=60, nT=13176, fixed id link
opt1$mess
# ^ false conv...

rep1 <- obj1$report()

unlist(rep1[c('beta1','sigmaeps1')])
unlist(rep1[c('beta2','sigmaeps2')])
# ^ only diff param between 2 loc types

unlist(rep1[c('alphavec1','alphavec2','phi','gamma','sigmadelta')])
# ^ common to 2 loc types

system.time(summ.rep1 <- summary(sdreport(obj1)))
# ^ 824 s = 14 min | MBP13 nS=60, nT=13176, fixed id link
summ.rep1[1:11,]
# ^ no NaN, all good when fixing gamma1 and gamma2

# save(list=c('loc','trdat','nT','nS','n','indlist.s','indlist.t',
#             'loc.latlon','proj.string','coord.utm','distmat',
#             'J1','kn1','Bmat1',
#             'J2','kn2','Bmat2',
#             'rawobs','yvec','obsind','loctype',
#             'datalist','parlist','obj1','opt1','rep1','summ.rep1'),
#      file='Fit1_ktgt.RData') # 1.3 GB...



### // look at fit1 ----
Xpred1 <- rep1$linpred - rep1$detfx

opt1$mess
# ^ not converged... cannot blame non-id between and gamma since fit1 has a
#   fixed identity link.

opt1$par
# ^ some confounding between beta's and gamma's, but other param very close

unlist(rep1[c('beta1','sigmaeps1')])

unlist(rep1[c('beta2','sigmaeps2')])
# ^ up to non-id between beta and gamma, estimating linear link (fit2) leads to
#   very similar intercept and meas err sd for both loc types as compared to
#   fixed identity link (fit1)

c(rep1$beta1,rep1$beta2) 
# ^ different intercept between loc: only minor for fit2

c(rep1$sigmaeps1,rep1$sigmaeps2)
# ^ clear difference in meas err sd between loc

# => only notable differences between loc types is in meas err sd

unlist(rep1[c('phi','gamma','sigmadelta')])


### plot seas pattern fit1 and fit2, both res
period1 <- 24 # hourly within day, data has hourly res
period2 <- 7*24 # daily within week, data has hourly res
weekdays(trdat$ts[1]) # training data starts on a Tuesday, shift for display

firstweek.ind <- 1:period2 + 24*6
weekdays(trdat$ts[firstweek.ind][1]) # ok, first Monday in tw

plot(1:period2,rep1$seas1[firstweek.ind],type='l',lty=2,
     main='fit1 daily and weekly seasonality',
     col='red',xaxt='n',ylim=c(-0.3,0.3))
abline(v=(0:7)*24+1,lty=3) # day separation within week
axis(side=1,at=(0:7)*24+1,labels=c('Mon','Tue','Wed','Thu','Fri','Sat','Sun','Mon'))
abline(v=c(7,13,19),lty=3,col='grey80') # hour separation within first days 
axis(side=1,at=c(7,13,19),labels=c('06:00','12:00','18:00'),cex.axis=0.6,padj=-2)
lines(1:period2,rep1$seas2[firstweek.ind],col='red',lty=2)
lines(1:period2,rep1$seas1[firstweek.ind]+rep1$seas2[firstweek.ind],col='red',lwd=2)

which.max(rep1$seas1[firstweek.ind])
# ^ daily max at 04:00
which.min(rep1$seas1[firstweek.ind])
# ^ daily low at 15:00

range(rep1$seas2)
# ^ weekly seas really just flat
range(rep1$seas1)
# ^ daily seas has ~0.28 amplitude on identity link scale (= lin pred scale)

plot(x=loc$utmx,y=loc$utmy)
diff(range(loc$utmx)) # E-W span = 33 km
diff(range(loc$utmy)) # N-S span = 37 km
max(distmat) # max Euclidean dist between all loc = 39 km
3*rep1$gamma # effective range for expo cov ~= 23 km
# ^ spat dep range is large relative to spread of locations => fairly smooth GRF

xrange <- 1:nT # 1st loc
# xrange <- (nT+1):(2*nT) # 2nd loc
# xrange <- 4000:4500 # arbitrary zoom
xlimi <- range(xrange) # c(2000,3000)

plot((1:n)[xrange],yvec[xrange],pch=19,cex=0.3,col='grey80',xlim=xlimi)
abline(v=(0:(nS-1))*nT+1) # separate loc
lines((1:n)[xrange],rep1$fitted[xrange],type='l',col='blue')
points((1:n)[xrange],yvec[xrange],pch=19,cex=0.3,col='grey80') # more visible
# ^ close to overfit?


par(mfrow=c(2,1))
plot(rep(1:nT,nS)[loctype==1],yvec[loctype==1],ylim=ylimi,
     main='kt',pch=21,cex=0.5,
     col=paste0('#696969',10),bg=paste0('#696969',30))
# ^ overlay times series for all loc type 1
lines(1:nT,rep1$fitted[indlist.s[[1]]],col='blue') # one loc as ref for fit
plot(rep(1:nT,nS)[loctype==2],yvec[loctype==2],ylim=ylimi,
     main='gt',pch=21,cex=0.5,
     col=paste0('#696969',10),bg=paste0('#696969',30))
# ^ overlay time series of all loc type 2
lines(1:nT,rep1$fitted[indlist.s[[nS1+1]]],col='blue') # one loc as ref
par(mfrow=c(1,1))
# ^ higher meas err in gt loc clearly visible

summary(rep1$fitted[loctype==1])
summary(rep1$fitted[loctype==2])

exp(max(yvec[loctype==2],na.rm=T)) # gt ceiling at 1000
exp(max(yvec[loctype==1],na.rm=T)) # kt goes beyond 1200

# Q-Q plots of raw residuals
qqnorm(yvec-rep1$fitted)
qqline(yvec-rep1$fitted,col='blue')
# ^ heavy tails, raw resid on log scale highly leptokurtic, although symmetric

plot(rep1$fitted,yvec) # on log scale
abline(0,1,col='red')

plot(exp(rep1$fitted),rawobs) # on exp/ori scale
abline(0,1,col='red')
# ^ mostly underpred peaks, especially weird values hitting apparent ceiling
#   around 750 and 1000

cor(rep1$fitted,yvec,use='complete.obs')
cor(exp(rep1$fitted),rawobs,use='complete.obs')
# ^ pretty good agreement nonetheless, cor ~= 0.985




#///////////////////////////////////////////////////////////////////////////////
#### 2021-03-18 Data May2018-May2020, kt/gt, 1h res, save padded tr/te sets ####
#///////////////////////////////////////////////////////////////////////////////

# training-test set split: rough 80%-20% with training data up to 2019-10-31
#                          23:00 and test data starting on 2019-11-01 00:00.
#                          cf. emails with Shiva 16-18 Mar 2021

# kt = Kaiterra sensors
# gt = government fixed monitors, CPCB/DPCC/IMD

rm(list=ls())
paf <- '~'
setwd(paf)


### // read raw data from Shiva's csv ----
kt.loc <- read.csv('kaiterra_locations.csv',header=T)
kt.sens <- read.csv(paste0('kaiterra_fieldeggid_1H_20180501_20201101.csv'),
                    header=T)
gt.loc <- read.csv('govdata_locations.csv',header=T)
gt.sens <- read.csv(paste0('govdata_1H_20180501_20201101.csv'),header=T)

str(kt.loc) # 29 unique sensors/locations
table(kt.loc$udid_short) # all unique

str(kt.sens)
# ^ nb levels for timestamp != nb obs
summary(as.numeric(table(kt.sens$timestamp_round)))
# ^ all timestamps appear exactly 28 times, 1 for each sensor
dim(kt.sens)[1]/28 # 21960 time points

str(gt.sens)
summary(as.numeric(table(gt.sens$timestamp_round)))
# ^ same, 32 sensors stacked in a single col, and same ts
dim(gt.sens)[1]/32 # 21960 time points


### // write kt/gt padded data to csv, training set ----
kt.sens$ts <- as.POSIXct(strptime(kt.sens$timestamp_round,
                                  format='%Y-%m-%d %H:%M',
                                  tz="Asia/Kolkata"))

ts.endtr <- as.POSIXct(strptime('2019-10-31 23:00',
                                format='%Y-%m-%d %H:%M',
                                tz="Asia/Kolkata")) # end tr set = Oct 31 2019
ind.tr.kt <- kt.sens$ts <= ts.endtr
table(ind.tr.kt) # indicator for training set, kt data

kt.keptloc <- levels(factor(kt.sens$field_egg_id))
kt.keptloc
# ^ 28 loc in kt, hexadecimal alphabetical order

tw.tr <- seq(min(kt.sens$ts[ind.tr.kt]),
             max(kt.sens$ts[ind.tr.kt]),
             by='1 hour')

sens.tr <- data.frame('ts'=tw.tr)

for (j in 1:length(kt.keptloc)){
  ind.tmp <- ind.tr.kt & kt.sens$field_egg_id==kt.keptloc[j]
  sens.tr[[paste0('pm25_',kt.keptloc[j])]] <- kt.sens$pm25[ind.tmp]
}
str(sens.tr,1) # ok

gt.sens$ts <- as.POSIXct(strptime(gt.sens$timestamp_round,
                                  format='%Y-%m-%d %H:%M',
                                  tz="Asia/Kolkata"))
ind.tr.gt <- gt.sens$ts <= ts.endtr
table(ind.tr.gt) # indicator for training set
gt.keptloc <- levels(factor(gt.sens$monitor_id))
gt.keptloc
# ^ 33 loc in gt, alphabetical order

for (j in 1:length(gt.keptloc)){
  ind.tmp <- ind.tr.gt & gt.sens$monitor_id==gt.keptloc[j]
  tmp <- gt.sens$pm25[ind.tmp]
  tmp[which(tmp<=0)] <- NA # replace non-pos by NA, safer
  sens.tr[[paste0('pm25_',gt.keptloc[j])]] <- tmp
}
str(sens.tr,1) # kt and gt all in 1 df

sum(sens.tr[-1]<0,na.rm=T) # ok, no negative values

sum(is.na(sens.tr))/prod(dim(sens.tr[-1]))
# about 28% missing values for kt and gt combined

# write.table(sens.tr,file='pm25_training.csv',sep=',',row.names=F,col.names=T)
# # ^ csv file of 13176 obs (hourly temp res) pm25 padded with NAs, 61 cols:
# #   - col 1: character string for time stamp
# #   - col 2-29: numerical for pm25, one for each of 28 kt loc
# #   - col 30-61: numerical for pm25, one for each of 32 gt loc


### // write kt/gt padded data to csv, test set ----
ts.endte <- as.POSIXct(strptime('2020-04-30 23:00',
                                format='%Y-%m-%d %H:%M',
                                tz="Asia/Kolkata"))
# ^ end total time window = May 1 2020

ind.te.kt <- (kt.sens$ts > ts.endtr) & (kt.sens$ts <= ts.endte)
table(ind.te.kt) # indicator for test set

tw.te <- seq(min(kt.sens$ts[ind.te.kt]),
             max(kt.sens$ts[ind.te.kt]),
             by='1 hour')
str(tw.te)

sens.te <- data.frame('ts'=tw.te)

for (j in 1:length(kt.keptloc)){
  ind.tmp <- ind.te.kt & kt.sens$field_egg_id==kt.keptloc[j]
  sens.te[[paste0('pm25_',kt.keptloc[j])]] <- kt.sens$pm25[ind.tmp]
}
str(sens.te,1)

ind.te.gt <- (gt.sens$ts > ts.endtr) & (gt.sens$ts <= ts.endte)
table(ind.te.gt) # indicator for test set

for (j in 1:length(gt.keptloc)){
  ind.tmp <- ind.te.gt & gt.sens$monitor_id==gt.keptloc[j]
  tmp <- gt.sens$pm25[ind.tmp]
  tmp[which(tmp<=0)] <- NA # replace non-pos by NA, safer
  sens.te[[paste0('pm25_',gt.keptloc[j])]] <- tmp
}
str(sens.te,1)

sum(sens.tr[-1]<0,na.rm=T) # ok no negative values

sum(is.na(sens.te))/prod(dim(sens.te[-1]))
# about 42% missing values for kt and gt combined

# write.table(sens.te,file='pm25_test.csv',sep=',',row.names=F,col.names=T)
# # ^ csv file of 4368 obs pm25 padded with NAs, 61 cols:
# #   - col 1: character string for time stamp
# #   - col 2-29: numerical for pm25, one for each of 28 kt loc
# #   - col 30-61: numerical for pm25, one for each of 32 gt loc


### // write kt/gt locations to csv ----
kt.loc$udid_short%in%kt.keptloc
# ^ 1 sensor/loc not in data

df.loc <- data.frame('source'=c(rep('kt',length(kt.keptloc)),
                                rep('gt',length(gt.keptloc))),
                     'id'=c(kt.keptloc,gt.keptloc))
df.loc

lonvec.kt <- double(length(kt.keptloc))
latvec.kt <- lonvec.kt
for (j in 1:length(kt.keptloc)){
  lonvec.kt[j] <- kt.loc$longitude[kt.loc$udid_short==kt.keptloc[j]]
  latvec.kt[j] <- kt.loc$latitude[kt.loc$udid_short==kt.keptloc[j]]
}

lonvec.gt <- double(length(gt.keptloc))
latvec.gt <- lonvec.gt
for (j in 1:length(gt.keptloc)){
  lonvec.gt[j] <- gt.loc$Longitude[gt.loc$Monitor.ID==gt.keptloc[j]]
  latvec.gt[j] <- gt.loc$Latitude[gt.loc$Monitor.ID==gt.keptloc[j]]
}

df.loc$lon <- c(lonvec.kt,lonvec.gt)
df.loc$lat <- c(latvec.kt,latvec.gt)

df.loc

# write.table(df.loc,file='locations.csv',sep=',',row.names=F,col.names=T)
# # ^ csv file of 60 rows for loc (28 kt and 32 gt) and 4 cols:
# #   - source: char str "kt" or "gt"
# #   - id: unique id for each loc
# #   - lon and lat
