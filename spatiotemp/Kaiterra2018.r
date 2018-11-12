#//////////////////////////////////////////////////////////////////////////
#### 2018-11-11 CPCB csv 11May-10June, fit v0.5 covariates+seasonality ####
#//////////////////////////////////////////////////////////////////////////

# covariates+seasonality = cs

rm(list=ls())
paf2drop <- '/Users/WAWA/Desktop/Dropbox'
paf <- paste0(paf2drop,'/PostDoc/AirPollution/epod-nyu-delhi-pollution/spatiotemp')
setwd(paf)

library(TMB)
library(rgdal) # for lat/lon conversion to utm
library(OpenStreetMap) # for map downloading and plotting
library(splines) # for B-spline basis
library(akima) # for 2d interp

colgreenred <- function(n,alpha=1){ # based on heat.colors
  if ((n <- as.integer(n[1L])) > 0) {
    return(rainbow(n, s=1, v=1, start=0, end=3.5/6, alpha=alpha)[n:1]) # end=2/6
  } else {
    return(character())
  }
}
# plot(1:100,col=colgreenred(100),pch=19) # test color gradient


### create and load function from cpp template
# compile("TGHM.cpp")
dyn.load(dynlib("TGHM"))


### load csv data
cpcb.loc <- read.table('CPCB_11May-10June_loc.csv',sep=',',header=T)
cpcb.sens <- read.table('CPCB_11May-10June_pm25.csv',sep=',',header=T)
cpcb.weat <- read.table('Kaiterra_11May-10June_weather.csv',sep=',',header=T)
cpcb.coord <- read.table('CPCB_11May-10June_coord.csv',sep=',',header=T)
# ^ same weather data as for Kaiterra data analysis

str(cpcb.loc)
str(cpcb.coord)

cpcb.sens$ts <- as.POSIXct(strptime(cpcb.sens$ts,
                                    format='%Y-%m-%d %H:%M',
                                    tz="Asia/Kolkata")) # date/time ISO standard
# cpcb.sens$ts <- format(cpcb.sens$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat, no seconds
str(cpcb.sens)

# sum(na.omit(unlist(cpcb.sens))==0) # ok now
# sum(na.omit(unlist(cpcb.sens))<0) # ok now


cpcb.weat$ts <- as.POSIXct(strptime(cpcb.weat$ts,
                                    format='%Y-%m-%d %H:%M',
                                    tz="Asia/Kolkata")) # date/time ISO standard
# cpcb.sens$ts <- format(cpcb.sens$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat, no seconds
str(cpcb.weat)
# ^ hourly data only, no spatial info, for whole Delhi

nS <- dim(cpcb.loc)[1] # 28 locations
nT <- dim(cpcb.sens)[1] # 2881 time points
n <- nS*nT # 80'668, not accounting for NAs
# sum(is.na(cpcb.sens))/(nS*nT) # 12% NAs

nS.full <- dim(cpcb.coord)[1] # 59 total nb locations
nS.mesh <- nS.full-nS # 31 extra locations
n.full <- nT*nS.full # 169'979, incl pred locations and not accounting for NAs


### create quadratic B-spline basis, assuming day is [0,1]
sb.dailygrid <- seq(0,1,length.out=96)
# ^ 24*4=96 obs per day

# cpcb.sens$ts[1:26]
# sb.dailygrid[1:26]
# ^ 2nd knot = 0.25 => first 24 obs in first sub-interval, ok

kn <- c(0,0.25,0.5,0.75,1) # fixed knots
# nb bases = J=6
# degree=2: nb knots = N = J-1, nb intervals = N-1 = J-2

sb <- bs(sb.dailygrid,degree=2,knots=kn,Boundary.knots=c(0,1))
sb <- sb[,-dim(sb)[2]] # extra col because I specify all knots

Bmat <- rep(1,30)%x%sb # 30 days + 15 min in tw
Bmat <- rbind(Bmat,Bmat[1,]) # additional 15 min for midnight on 10 June
# dim(Bmat) # ok only time, without space yet, dim(cpcb.sens)[1] = 2881

Bmat <- rep(1,nS.full)%x%Bmat # replicate for 123 locations
dim(Bmat) # ok n.full=169'979
# ^ clearly inefficient and RAM-consuming...
# ^ TODO for later: replicate within C++ template


### create map.delhi for mapping
range(cpcb.loc$lat)
range(cpcb.loc$lon)

corners.delhi <- list('topleft'=c(28.46, 77.02), # lat/lon
                      'botright'=c(28.76, 77.32)) # lat/lon
map.delhi <- openmap(upperLeft=corners.delhi[[1]],lowerRight=corners.delhi[[2]],
                     zoom=NULL,type='stamen-toner') # type='osm'

df.latlon <- SpatialPoints(cpcb.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
df.latlon.sp <- spTransform(df.latlon,osm())

plot(map.delhi,removeMargin=F)
points(df.latlon.sp,pch=19,col='blue')
axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
title(main="CPCB/DPCC/IMD monitors Mar-Oct 2018, tw 2018-05-11 - 2018-06-10",
      xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')



### prep response vec and design for full spatial grid for X pred
coord.full <- as.matrix(cpcb.coord[,-3]) # extra locations at end, easier

proj.string <- "+proj=utm +zone=43 +ellps=WGS84 +north +units=km"
# ^ Delhi = UTM zone 43R
df.coord.full <- SpatialPoints(coord.full,proj4string=CRS(proj.string))
coord.full.osm <- spTransform(df.coord.full,map.delhi$tiles[[1]]$projection)

y.vec <- unlist(cpcb.sens[,-1]) # stack, grouped by location
logy.vec <- log(y.vec) # log response, NAs in correct places
logy.vec.full <- c(logy.vec,rep(NA,nS.mesh*nT)) # add NAs for extra locations

obsind.full <- as.integer(!is.na(logy.vec.full)) # 1=available, 0=missing value


distmat.full <- matrix(NA_real_,nS.full,nS.full) # Euclidean dist
for (i in 1:nS.full){
  distmat.full[i,i] <- 0
  j <- 1
  while (j<i){
    distmat.full[i,j] <- sqrt((coord.full[i,1]-coord.full[j,1])^2 + 
                                + (coord.full[i,2]-coord.full[j,2])^2)
    distmat.full[j,i] <- distmat.full[i,j] # symmetry
    j <- j+1
  }
}
distmat.full[1:5,1:5] # in km because utm coord in km from proj


### create zmat, correct dim by replicating covariate
str(cpcb.weat) # only 2881 rows, time only
zmat <- as.matrix(cpcb.weat[,-1])

zmat <- rep(1,nS.full)%x%zmat # replicate for 54 locations
dim(zmat) # ok n.full=169'979

# note: no intercept in zmat => intercept in seasonal component Bmat


### fit ML v0.5, intercept only for now, sampled+extra locations
datalist.full <- list()
datalist.full$log_y <- logy.vec.full
datalist.full$obsind <- obsind.full
datalist.full$zmat <- zmat
datalist.full$Bmat <- Bmat 
datalist.full$kn <- kn
datalist.full$distmat <- distmat.full
datalist.full$interceptonly <- 0L # covariates + seasonality

parlist.full <- list()
parlist.full$beta <- c(0,0,0) # dim = p = dim(zmat)[2]
parlist.full$alpha <- c(0.1,0.1,0,-0.1) # spline coeff, dim = J-1 = dim(Bmat)[2]-1
parlist.full$log_sigmaepsilon <- 0 # log(sigmaepsilon)
parlist.full$t_phi <- 1 # log((1+phi)/(1-phi)) # (exp(x)-1)/(exp(x)+1)
parlist.full$log_gamma <- 0 # log(gamma)
parlist.full$log_sigmadelta <- 0 # log(sigmadelta)
parlist.full$X <- rep(0,n.full) # logy.vec

system.time(obj.full <- MakeADFun(data=datalist.full,parameters=parlist.full,
                                  random=c('X'),DLL="TGHM",silent=T))
# ^ 21s for nS.mesh=31 with interceptonly=0

system.time(print(obj.full$fn()))
# ^ 204s for nS.mesh=31 with interceptonly=0

system.time(print(obj.full$gr()))
# ^ 20s for nS.mesh=31 with interceptonly=0


system.time(opt.full <- nlminb(start=obj.full$par,obj=obj.full$fn,gr=obj.full$gr,
                               control=list(eval.max=500,iter.max=500)))
# ^ 2954s nS.full=59, nT=2881, interceptonly=0
opt.full$mess # ok

system.time(rep.full <- sdreport(obj.full))
# ^ 313s nS.full=59, nT=2881, with interceptonly=0

summ.rep.full <- summary(rep.full)
summ.rep.full[(11+n.full+1):dim(summ.rep.full)[1],]
# ^ se available!

# save.image('cpcb_11May-10June_TempFit_cs_nS.mesh.31.RData')
# load('cpcb_11May-10June_TempFit_cs_nS.mesh.31.RData')


### compute PM2.5 predictions, interceptonly model
X.pred.full <- t(matrix(summ.rep.full[dimnames(summ.rep.full)[[1]]=='X',1],
                        nT,nS.full))
# X.se.full <- t(matrix(summ.rep.full[dimnames(summ.rep.full)[[1]]=='X',2],
#                       nT,nS.full))
# ^ original layout of data: one row per location, time points as cols
range(X.pred.full)
# ^ roughly [-7.8,+3] <= much wider than on Kaiterra data

detfx <- as.numeric(zmat%*%summ.rep.full[1:3,1])
season <- Bmat%*%summ.rep.full[(14+n.full+1):(19+n.full+1),1]

fixed.fx.full <- t(matrix(detfx+season,nT,nS.full)) # covariates + seasonality
# ^ fixed effects, constant through time
range(fixed.fx.full) # roughly [4.1,4.5], narrower than [3.86,4.52] on kt

pred.pm.full <- exp(fixed.fx.full+X.pred.full)
range(y.vec,na.rm=T)
range(pred.pm.full) # fairly close but much smaller max => spike smoothed out?
# ^ identical range, overfit? also sigma-measurement error arbitrarily close to
#   zero, so X basically matches observed y....


### plot PM2.5 predictions on top of Delhi map
lbub <- c(0,1408) # bounds for color gradient, scale of pred.pm.full
mai.def <- c(1.02, 0.82, 0.82, 0.42)
mar.def <- c(5.1, 4.1, 4.1, 2.1)
alpha.colgrad <- 0.7
legend_image <- as.raster(matrix(colgreenred(100,alpha=alpha.colgrad)[100:1],
                                 ncol=1))

loc.lonlat <- SpatialPoints(cpcb.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
loc.osm <- spTransform(loc.lonlat,map.delhi$tiles[[1]]$projection) # re-project

ts2print <- format(cpcb.sens$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat, no seconds

# pdf('Kaiterra_11May-10June_STHMio_MapPredPM25.pdf',width=8,height=8,onefile=T)
# for (j in 1:nT){
  # layout(matrix(1:2,ncol=2),width=c(8,1),height=c(1,1)) # split plot region
  # par(mai=rep(1,4),mar=c(6,5,4,2))
  # plot(map.delhi,removeMargin=F)
  # intsurf <- interp(x=coord.full.osm@coords[,1],y=coord.full.osm@coords[,2],
  #                   z=pred.pm.full[,j],nx=200,ny=200,linear=T)
  # image(intsurf$x,intsurf$y,intsurf$z,col=colgreenred(100,alpha=alpha.colgrad),
  #       zlim=lbub,add=T)
  # points(loc.osm,pch=19,col='blue')
  # axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
  # axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
  # title(main=paste0("CPCB/DPCC/IMD 11 May - 10 June, cs, predicted PM2.5"),
  #       xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')
  # title(sub=ts2print[j],cex.sub=0.8,adj=1) # time stamp at bottomright
  # # legend as color bar
  # par(mar=c(5.1,1,4.1,1))
  # plot(c(0,2),c(0,1),type='n',axes=F,xlab ='',ylab='')
  # rasterImage(legend_image,xleft=0,ybottom=0,xright=2,ytop=1,angle=0)
  # text(x=1,y=seq(0.02,0.98,l=5),adj=0.5,cex=0.7,
  #      labels=round(seq(lbub[1],lbub[2],l=5),2))
  # par(mai=mai.def,mar=mar.def) # back to default margins
  # layout(1) # default layout
# }
# dev.off()
# # ^ file way too heavy!


### separate png for video
lbub <- c(0,1408) # bounds for color gradient, scale of pred.pm.full
mai.def <- c(1.02, 0.82, 0.82, 0.42)
mar.def <- c(5.1, 4.1, 4.1, 2.1)
alpha.colgrad <- 0.7
legend_image <- as.raster(matrix(colgreenred(100,alpha=alpha.colgrad)[100:1],
                                 ncol=1))

for (j in 1:nT){
  png(paste0('Outputs/VideoSeparatePNG/CPCB_11May-10June_STHMcs_MapPredPM25_',
             sprintf(j,fmt='%04d'),'.png'),
      width=7,height=7,res=200,units='in')
  layout(matrix(1:2,ncol=2),width=c(8,1),height=c(1,1)) # split plot region
  # par(mar=c(5.1,4.1,4.1,1))
  par(mai=rep(1,4),mar=c(6,5,4,2))
  plot(map.delhi,removeMargin=F)
  intsurf <- interp(x=coord.full.osm@coords[,1],y=coord.full.osm@coords[,2],
                    z=pred.pm.full[,j],nx=200,ny=200,linear=T)
  image(intsurf$x,intsurf$y,intsurf$z,col=colgreenred(100,alpha=alpha.colgrad),
        zlim=lbub,add=T)
  points(loc.osm,pch=19,col='blue')
  axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
  axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
  title(main=paste0("CPCB/DPCC/IMD 11 May - 10 June, covariates+seasonality, predicted PM2.5"),
        xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')
  title(sub=ts2print[j],cex.sub=0.8,adj=1) # time stamp at bottomright
  # legend as color bar
  par(mar=c(5.1,1,4.1,1))
  plot(c(0,2),c(0,1),type='n',axes=F,xlab ='',ylab='')
  rasterImage(legend_image,xleft=0,ybottom=0,xright=2,ytop=1,angle=0)
  text(x=1,y=seq(0.02,0.98,l=5),adj=0.5,cex=0.7,
       labels=round(seq(lbub[1],lbub[2],l=5),2))
  par(mai=mai.def,mar=mar.def) # back to default margins
  layout(1) # default layout
  dev.off()
}

# ffmpeg -framerate 24 -i CPCB_11May-10June_STHMcs_MapPredPM25_%04d.png
#   ../CPCB_11May-10June_STHMcs_MapPredPM25.mp4

# 24 fps => 1 movie second = 6 real time hours


### plot of daily seasonal periodic effect
alphavec <- summ.rep.full[(14+n.full+1):(19+n.full+1),1] # dim J=6

kn <- c(0.00,0.25,0.50,0.75,1.00)

xgrid <- seq(0,1,0.01)

sb.grid <- bs(xgrid,degree=2,knots=kn,Boundary.knots=c(0,1))
sb.grid <- sb.grid[,-dim(sb.grid)[2]] # extra col because I specify all knots
sb.grid.rep <- rep(1,3)%x%sb.grid # repeat for visuals

dailyseason <- as.numeric(sb.grid%*%alphavec)
dailyseason.rep <- as.numeric(sb.grid.rep%*%alphavec)

pdf('CPCB_11May-10June_SeasonalEffect.pdf',width=8,height=7)
plot(c(seq(-1,0,0.01),seq(0,1,0.01),seq(1,2,0.01)),xlim=c(-0.3,1.3),
     dailyseason.rep,type='l',xaxt='n',xlab='Time (hours)',col='grey',lty=2,
     ylab='Predicted log(PM2.5) concentration',
     main='CPCB/DPCC/IMD B-splines estimated daily seasonal effect')
abline(h=seq(3.9,4.5,0.1),lty=3,col='lightgrey')
abline(v=seq(-0.25,1.25,0.25),lty=3,col='lightgrey')
lines(xgrid,dailyseason)
axis(side=1,at=seq(-0.25,1.25,0.25),
     labels=c('18:00','00:00','06:00','12:00','18:00','00:00','06:00'))
dev.off()








#//////////////////////////////////////////////////////////////////////
#### 2018-11-11 CPCB csv 11May-10June, extra loc inlamesh save csv ####
#//////////////////////////////////////////////////////////////////////

rm(list=ls())
paf2drop <- '/Users/WAWA/Desktop/Dropbox'
paf <- paste0(paf2drop,'/PostDoc/AirPollution/epod-nyu-delhi-pollution/spatiotemp')
setwd(paf)

library(INLA) # for mesh
library(rgdal) # for lat/lon conversion to utm
library(OpenStreetMap) # for map downloading and plotting

cpcb.loc <- read.table('CPCB_11May-10June_loc.csv',sep=',',header=T)
cpcb.sens <- read.table('CPCB_11May-10June_pm25.csv',sep=',',header=T)
cpcb.weat <- read.table('Kaiterra_11May-10June_weather.csv',sep=',',header=T)
# ^ same weather data since for all Delhi at same ts

str(cpcb.loc) # 28 locations

cpcb.sens$ts <- as.POSIXct(strptime(cpcb.sens$ts,
                                    format='%Y-%m-%d %H:%M',
                                    tz="Asia/Kolkata")) # date/time ISO standard
# cpcb.sens$ts <- format(cpcb.sens$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat, no seconds
str(cpcb.sens)

cpcb.weat$ts <- as.POSIXct(strptime(cpcb.weat$ts,
                                    format='%Y-%m-%d %H:%M',
                                    tz="Asia/Kolkata")) # date/time ISO standard
# cpcb.sens$ts <- format(cpcb.sens$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat, no seconds
str(cpcb.weat)
# ^ hourly data only, no spatial info, for whole Delhi

nS <- dim(cpcb.loc)[1] # 28 locations
nT <- dim(cpcb.sens)[1] # 2881 time points
n <- nS*nT # 80'668, not accounting for NAs

sum(is.na(cpcb.sens))/(nS*nT) # 12% NAs, not bad



### use INLA to create grid of locations by Delaunay triangulation
coord <- cpcb.loc[,c('utmx','utmy')]
apply(coord,2,range) # determine borders of domain

mar.mesh <- 1.0 # beyond range of obs locations
coord.border <- data.frame('utmx'=c(min(coord$utmx)-mar.mesh,
                                    rep(max(coord$utmx)+mar.mesh,2),
                                    rep(min(coord$utmx)-mar.mesh,2)),
                           'utmy'=c(rep(min(coord$utmy)-mar.mesh,2),
                                    rep(max(coord$utmy)+mar.mesh,2),
                                    min(coord$utmy)-mar.mesh))
# ^ 4 corners, last=first to close domain, mar.mesh beyond observed range

system.time(inlamesh <- inla.mesh.2d(loc=coord, # coordinates in UTM
                                     loc.domain=coord.border,
                                     offset=1, #                  | offset=1
                                     max.edge=10,#                | max.edge=5
                                     min.angle=5,  #              | min.angle=20
                                     # max.n=1000, # overrides max.edge
                                     cutoff=0,
                                     plot.delay=NULL))
inlamesh$n
# ^ with nT=2881 and nS=18, 123 is too much for my MBP RAM- and CPU-wise

plot(inlamesh)
lines(coord.border,lwd=3,col='blue')
points(coord$utmx,coord$utmy,pch=20,cex=1.5,col=2)

plot(inlamesh$loc[,1],inlamesh$loc[,2],pch=8,cex=1,col='deeppink')
lines(coord.border,lwd=3,col='blue') # well-spread?

coord.mesh <- data.frame('utmx'=inlamesh$loc[,1],'utmy'=inlamesh$loc[,2])
# ^ to be used for predictions and mapping

which(coord.mesh[,1]%in%coord[,1])
which(coord.mesh[,2]%in%coord[,2])
# ^ original points are included in the mesh locations, arbitrary position

coord.mesh <- coord.mesh[-which(coord.mesh[,1]%in%coord[,1]),]
str(coord.mesh)
points(coord.mesh[,1],coord.mesh[,2],pch=8,cex=1,col='limegreen')
# ^ 105 extra locations too much for my MBP

nS.mesh <- dim(coord.mesh)[1] # 105 extra locations
nS.full <- nS+nS.mesh # 59 total nb locations
n.full <- nT*nS.full # 169'979, incl pred locations and not accounting for NAs

coord.df <- data.frame(rbind(coord,coord.mesh))
coord.df$observed <- c(rep(1L,nS),rep(0L,dim(coord.mesh)[1]))

write.table(coord.df,file='CPCB_11May-10June_coord.csv',sep=',',
            row.names=F,col.names=T)






#//////////////////////////////////////////////////////////////////////////////
#### 2018-11-11 CPCB updated data Mar-Sep 2018 csv from Shiva, save tw csv ####
#//////////////////////////////////////////////////////////////////////////////

rm(list=ls())
paf2drop <- '/Users/WAWA/Desktop/Dropbox'
paf <- paste0(paf2drop,'/PostDoc/AirPollution/epod-nyu-delhi-pollution/spatiotemp')
setwd(paf)

pafdata <- '/Delhi Pollution/07_Data_Secondary/CPCB_Oct2018'


### import data from Shiva's csv, reformat timestamp, check nb sensors/locations
cpcb.full <- read.table(paste0(paf2drop,pafdata,'/govdata_15min_all.csv'),
                        sep=',',header=T)
str(cpcb.full) # huge

cpcb.full$ts <- as.POSIXct(strptime(cpcb.full$timestamp_round,
                                    format='%Y-%m-%d %H:%M:%S',
                                    tz="Asia/Kolkata")) # date/time ISO standard
# kt.full$ts <- format(kt.full$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat no seconds
# ^ convert to correct class with IST time zone
range(cpcb.full$ts) # overview of full time period = Mar 1 - Oct 31

names.loc <- as.character(unique(cpcb.full$location))
nb.loc <- length(names.loc) # 33 locations overall


### visualize data availability with tw
tw.bd <- as.POSIXct(c('2018-05-11 00:00:00',
                      '2018-06-10 00:00:00'),tz="Asia/Kolkata")

tw.vizbd <- range(cpcb.full$ts)

# by location (= unique field_egg_id)
plot(x=tw.vizbd,y=c(1,nb.loc),type='n',
     xlab='Total time period',ylab='Location',yaxt='n',
     main='CPCB/DPCC/IMD Mar-Oct 2018, up and down time by location')
axis(side=2,at=1:nb.loc,labels=names.loc,las=1)
for (j in 1:nb.loc){
  ind <- cpcb.full$location==names.loc[j]
  ind.na <- is.na(cpcb.full$pm25[ind])
  arrows(x0=min(cpcb.full$ts[ind]),x1=max(cpcb.full$ts[ind]),y0=j,y1=j,
         col='grey',angle=90,code=3,length=0.05,lty=1)
  points(x=cpcb.full$ts[ind][!ind.na],y=rep(j,sum(!ind.na)),pch=19,cex=0.1)
}
abline(v=tw.bd,lty=2)
# ^ directly exclude five locations

discarded.loc <- c("Sri Aurobindo Marg, Delhi - DPCC",
                   "Pusa, Delhi - DPCC",
                   "Mundaka, Delhi - DPCC",
                   "East Arjun Nagar, Delhi - CPCB",
                   "Burari Crossing, New Delhi - IMD")
kept.loc <- names.loc[!names.loc%in%discarded.loc] # 28 left



### create df lat/lon and map both kept and discarded locations
cpcb.loc <- data.frame('loc'=names.loc,stringsAsFactors=F)
table(round(cpcb.full$longitude,4)) # looks ok
table(round(cpcb.full$latitude,4)) # looks ok

for (j in 1:nb.loc){
  cpcb.loc$lon[j] <- cpcb.full$longitude[cpcb.full$location==names.loc[j]][1]
  cpcb.loc$lat[j] <- cpcb.full$latitude[cpcb.full$location==names.loc[j]][1]
}


library(sp)
library(rgdal)
library(OpenStreetMap)

range(cpcb.full$latitude)
range(cpcb.full$longitude)
corners.delhi <- list('topleft'=c(28.47, 77.03), # lat/lon
                      'botright'=c(28.76, 77.32)) # lat/lon
map.delhi <- openmap(upperLeft=corners.delhi[[1]],lowerRight=corners.delhi[[2]],
                     zoom=NULL,type='stamen-toner') # type='osm'

df.latlon <- SpatialPoints(cpcb.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
df.latlon.sp <- spTransform(df.latlon,osm())

plot(map.delhi,removeMargin=F)
points(df.latlon.sp,pch=19,col=ifelse(names.loc%in%discarded.loc,'red','blue'))
text(names.loc,x=df.latlon.sp@coords[1:j,1],y=df.latlon.sp@coords[1:j,2],
     col=ifelse(names.loc%in%discarded.loc,'red','blue'),cex=0.5
     # ,pos=c(4,2,3,2,4,4,2,4,4,1,1,2,1,4,4,4,1,4,4,2,2,4)
)
axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
title(main="CPCB/DPCC/IMD monitors Mar-Oct 2018, tw 2018-05-11 - 2018-06-10",
      xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')
legend('topleft',c('Discarded','Kept'),pch=c(19,19),bg='white',
       col=c('red','blue'))
# ^ ok, discarded monitors do not affect convex hull much, all quite compact


### 28 locations left: extract tw from cpcb.full, save data to txt (no RData)
cpcb <- subset(cpcb.full,subset=(cpcb.full$ts >= tw.bd[1] &
                                   cpcb.full$ts <= tw.bd[2] &
                                   cpcb.full$location%in%kept.loc))
cpcb <- cpcb[,c('location','ts','pm25')]
str(cpcb) # 2881*28 = 80668 obs

range(cpcb$ts)
tw.bd # identical by construction

tw <- seq(from=tw.bd[1],to=tw.bd[2],by='15 min') # common time stamp
tw <- format(tw,"%Y-%m-%d %H:%M",usetz=T) # reformat without seconds
length(tw) # 2881 values incl bounds
str(tw) # correct format, no seconds

cpcb.shortloc <- c("Anand_Vihar","Ashok_Vihar","Aya_Nagar","CRRI_Mathura_Road",
                   "DTU","Dr._Karni_Singh_Shooting_Range","Dwarka-Sector_8",
                   "IGI_Airport_(T3)","IHBAS,_Dilshad_Garden","ITO",
                   "Jahangirpuri","Jawaharlal_Nehru_Stadium","Lodhi_Road",
                   "Major_Dhyan_Chand_National_Stadium","Mandir_Marg",
                   "NSIT_Dwarka","Nehru_Nagar","Okhla_Phase-2","Patparganj",
                   "Punjabi_Bagh","Pusa","R_K_Puram","Rohini","Shadipur",
                   "Sirifort","Sonia_Vihar","Vivek_Vihar","Wazirpur")

cpcb.sens <- data.frame('ts'=tw)
for (j in 1:length(kept.loc)){
  ind <- cpcb$location==kept.loc[j]
  tmp <- cpcb$pm25[ind]
  # sum(tmp<=0,na.rm=T)
  tmp[which(tmp<=0)] <- NA # replace non-pos by NA, safest
  cpcb.sens[[paste0('pm25_',cpcb.shortloc[j])]] <- tmp
}
str(cpcb.sens,1)
head(cpcb.sens)
tail(cpcb.sens) # looks good

write.table(cpcb.sens,file='CPCB_11May-10June_pm25.csv',sep=',',
            row.names=F,col.names=T)
# ^ csv file of 2881 obs pm25, 29 cols:
#   - col 1: character string for time stamp
#   - col 2-29: numerical for pm25, one for each location (short names)


### create df of 18 locations with lat/lon and UTM coord
df.latlon <- SpatialPoints(cpcb.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
proj.string <- "+proj=utm +zone=43 +ellps=WGS84 +north +units=km"
# ^ Delhi = UTM zone 43R
coord.utm <- spTransform(df.latlon, CRS(proj.string)) # re-project
cpcb.loc$utmx <- coord.utm@coords[,1]
cpcb.loc$utmy <- coord.utm@coords[,2]

str(cpcb.loc)

cpcb.loc <- cpcb.loc[names.loc%in%kept.loc,] # keep only kept loc

cpcb.loc$loc <- cpcb.shortloc
str(cpcb.loc) # much cleaner

write.table(cpcb.loc,file='CPCB_11May-10June_loc.csv',sep=',',
            row.names=F,col.names=T)
# ^ csv file for 28 rows = 28 locations:
#   - col 1 = character string for location (= short name)
#   - col 2 = numerical for longitude
#   - col 3 = numerical for latitude
#   - col 4 = numerical for projected UTM X
#   - col 5 = numerical for projected UTM Y









#////////////////////////////////////////////////////////////////////////////////
#### 2018-11-11 Kaiterra csv 11May-10June, v0.5 cs, leave-one-out CV spatial ####
#////////////////////////////////////////////////////////////////////////////////

rm(list=ls())
paf2drop <- '/Users/WAWA/Desktop/Dropbox'
paf <- paste0(paf2drop,'/PostDoc/AirPollution/epod-nyu-delhi-pollution/spatiotemp')
setwd(paf)

library(TMB)

### create and load function from cpp template
# compile("TGHM.cpp")
dyn.load(dynlib("TGHM"))

### load envir from whole fit to Kaiterra 11May-10June data
load('kt_11May-10June_TempFit_cs_nS.mesh.36.RData')


### Spatial pred, loop over all sensors, record pred resid (raw diff) for each as a ts
res.spcv.raw <- matrix(NA_real_,nS,nT) # raw resid for spatial leave-one-out CV
res.spcv.rel <- res.spcv.raw # divide by obs and *100, relative but no abs val

wallclock <- proc.time()[3]
for (j in 1:nS){
  excl.sens <- j # out of nS
  which.excl <- (1:nT)+(excl.sens-1)*nT # space=outer loop, time=inner loop
  # ^ excluded values in stacked vector
  # ^ valid for original data also, because extra loc stacked after
  
  logy.vec.full.cv <- logy.vec.full
  logy.vec.full.cv[which.excl] <- NA # as if missing values
  obsind.full.cv <- as.integer(!is.na(logy.vec.full.cv))
  # ^ 1=available, 0=missing value
  
  logy.vec.cv <- logy.vec
  logy.vec.cv[which.excl] <- NA # as if missing values
  
  
  # ML fit to sub-sample
  datalist.full.cv <- list()
  datalist.full.cv$log_y <- logy.vec.full.cv
  datalist.full.cv$obsind <- obsind.full.cv
  datalist.full.cv$zmat <- zmat
  datalist.full.cv$Bmat <- Bmat 
  datalist.full.cv$kn <- kn
  datalist.full.cv$distmat <- distmat.full
  datalist.full.cv$interceptonly <- 0L # covariates + seasonality
  
  obj.full.cv <- MakeADFun(data=datalist.full.cv,parameters=parlist.full,
                           random=c('X'),DLL="TGHM",silent=T)
  # ^ 18s for nS.mesh=36 with interceptonly=0 on full data
  
  opt.full.cv <- nlminb(start=obj.full.cv$par,obj=obj.full.cv$fn,gr=obj.full.cv$gr,
                        control=list(eval.max=500,iter.max=500))
  # ^ 1292s nS.full=54, nT=2881, interceptonly=0 on full data
  
  rep.full.cv <- sdreport(obj.full.cv)
  # ^ 237s nS.full=54, nT=2881, with interceptonly=0 on full data
  
  summ.rep.full.cv <- summary(rep.full.cv)
  
  # prediction error of excl.sens
  X.pred.full.cv <- t(matrix(summ.rep.full[dimnames(summ.rep.full.cv)[[1]]=='X',1],
                             nT,nS.full))
  
  detfx.cv <- as.numeric(zmat%*%summ.rep.full.cv[1:3,1])
  season.cv <- Bmat%*%summ.rep.full.cv[(14+n.full+1):(19+n.full+1),1]
  fixed.fx.full.cv <- t(matrix(detfx.cv+season.cv,nT,nS.full)) # cs
  pred.pm.full.cv <- exp(fixed.fx.full.cv+X.pred.full.cv)
  
  # plot(pred.pm.full.cv[which.excl],y.vec[which.excl],xlim=c(0,600),ylim=c(0,600))
  # abline(0,1,col='red')
  
  res.spcv.raw[j,] <- y.vec[which.excl]-pred.pm.full.cv[excl.sens,] # ts, length=nT
  res.spcv.rel[j,] <- 100*(y.vec[which.excl]-pred.pm.full.cv[excl.sens,])/
    y.vec[which.excl]
  message('loc ',j,' out of ',nS)
}
elapsed.time <- proc.time()[3]-wallclock
print(elapsed.time)
# ^ 28030s = 7h47 for nS.mesh=36, leave-one-out

# par(mfrow=c(2,1))
# for (j in 1:nS){
#   plot(kt.sens[,1],res.spcv.raw[j,],type='l',
#        main=paste0('Spatial pred raw resid, sensor ',names.sens[j]))
#   plot(kt.sens[,1],res.spcv.rel[j,],type='l',
#        main=paste0('Spatial pred rel resid, sensor ',names.sens[j]))
# }
# par(mfrow=c(1,1))

# save.image('kt_11May-10June_cs_nS.mesh.36_SptialLOOCV.RData')











#//////////////////////////////////////////////////////////////////////////
#### 2018-11-11 Kaiterra csv 11May-10June, v0.5 cs, 9-fold CV temporal ####
#//////////////////////////////////////////////////////////////////////////

rm(list=ls())
paf2drop <- '/Users/WAWA/Desktop/Dropbox'
paf <- paste0(paf2drop,'/PostDoc/AirPollution/epod-nyu-delhi-pollution/spatiotemp')
setwd(paf)

library(TMB)

### create and load function from cpp template
# compile("TGHM.cpp")
dyn.load(dynlib("TGHM"))

### load envir from whole fit to Kaiterra 11May-10June data
load('kt_11May-10June_TempFit_cs_nS.mesh.36.RData')


### Temporal pred: 9-fold CV, predict blocks of 320 obs (last=321) for all loc
nT/9 # 9-fold CV: 8 first blocks with 320 obs and last with 321 obs
8*320 + 321 # ok

bs <- c(rep(320,8),321) # block sizes, sum(bs)=nT

mape.temp.abs <- double(9) # 9 folds, absolute error
mape.temp.rel <- mape.temp.abs # relative error

wallclock <- proc.time()[3]
for (j in 1:9){ # loop over 9 folds
  btw <- (1:bs[j])+(j-1)*bs[1] # block time window, last one has diff length

  which.excl <- as.numeric(sapply((1:nS-1)*nT,function(x){x+btw}))
  # ^ excluded values in full stacked vector, btw for each loc
  # ^ valid for original data also, because extra loc stacked after
  
  logy.vec.full.cv <- logy.vec.full
  logy.vec.full.cv[which.excl] <- NA # as if missing values
  
  obsind.full.cv <- as.integer(!is.na(logy.vec.full.cv))
  # ^ 1=available, 0=missing value
  # plot(t(matrix(obsind.full.cv,nT,nS))[2,]) # check zeros in correct places
  
  logy.vec.cv <- logy.vec
  logy.vec.cv[which.excl] <- NA # as if missing values
  

  # ML fit to sub-sample
  datalist.full.cv <- list()
  datalist.full.cv$log_y <- logy.vec.full.cv
  datalist.full.cv$obsind <- obsind.full.cv
  datalist.full.cv$zmat <- zmat
  datalist.full.cv$Bmat <- Bmat 
  datalist.full.cv$kn <- kn
  datalist.full.cv$distmat <- distmat.full
  datalist.full.cv$interceptonly <- 0L # covariates + seasonality
  
  obj.full.cv <- MakeADFun(data=datalist.full.cv,parameters=parlist.full,
                           random=c('X'),DLL="TGHM",silent=T)
  # ^ 18s for nS.mesh=36 with interceptonly=0 on full data
  
  opt.full.cv <- nlminb(start=obj.full.cv$par,obj=obj.full.cv$fn,gr=obj.full.cv$gr,
                        control=list(eval.max=500,iter.max=500))
  # ^ 1292s nS.full=54, nT=2881, interceptonly=0 on full data
  
  rep.full.cv <- sdreport(obj.full.cv)
  # ^ 237s nS.full=54, nT=2881, with interceptonly=0 on full data
  
  summ.rep.full.cv <- summary(rep.full.cv)
  
  # prediction error of excl.sens
  X.pred.full.cv <- t(matrix(summ.rep.full[dimnames(summ.rep.full.cv)[[1]]=='X',1],
                             nT,nS.full))
  
  detfx.cv <- as.numeric(zmat%*%summ.rep.full.cv[1:3,1])
  season.cv <- Bmat%*%summ.rep.full.cv[(14+n.full+1):(19+n.full+1),1]
  fixed.fx.full.cv <- t(matrix(detfx.cv+season.cv,nT,nS.full)) # cs
  pred.pm.full.cv <- exp(fixed.fx.full.cv+X.pred.full.cv)
  
  # plot(pred.pm.full.cv[which.excl],y.vec[which.excl],xlim=c(0,600),ylim=c(0,600))
  # abline(0,1,col='red')
  mape.temp.abs[j] <- mean(abs(y.vec[which.excl]-pred.pm.full.cv[which.excl]),na.rm=T)
  mape.temp.rel[j] <- mean(abs(y.vec[which.excl]-pred.pm.full.cv[which.excl])/
                             y.vec[which.excl],na.rm=T)
  message('fold ',j,' out of 9')
}
elapsed.time <- proc.time()[3]-wallclock
print(elapsed.time)
# ^ 14514s = 4h for nS.mesh=36, 9-fold

mape.temp.abs
mean(mape.temp.abs) # off by 37.65 mu g/m3 on average

mape.temp.rel
mean(mape.temp.rel) # 62% relative error on average...
# ^ improved over 70%-72% on PP2 with only intercept as det fx, although not
#   same data so not really comparable.

# save.image('kt_11May-10June_cs_nS.mesh.36_Temporal9foldCV.RData')










#//////////////////////////////////////////////////////////////////////////////
#### 2018-11-10 Kaiterra csv 11May-10June, fit v0.5 covariates+seasonality ####
#//////////////////////////////////////////////////////////////////////////////

# covariates+seasonality = cs

rm(list=ls())
paf2drop <- '/Users/WAWA/Desktop/Dropbox'
paf <- paste0(paf2drop,'/PostDoc/AirPollution/epod-nyu-delhi-pollution/spatiotemp')
setwd(paf)

library(TMB)
library(rgdal) # for lat/lon conversion to utm
library(OpenStreetMap) # for map downloading and plotting
library(splines) # for B-spline basis
library(akima) # for 2d interp

colgreenred <- function(n,alpha=1){ # based on heat.colors
  if ((n <- as.integer(n[1L])) > 0) {
    return(rainbow(n, s=1, v=1, start=0, end=3.5/6, alpha=alpha)[n:1]) # end=2/6
  } else {
    return(character())
  }
}
# plot(1:100,col=colgreenred(100),pch=19) # test color gradient


### create and load function from cpp template
# compile("TGHM.cpp")
dyn.load(dynlib("TGHM"))


### load csv data
kt.loc <- read.table('Kaiterra_11May-10June_loc.csv',sep=',',header=T)
kt.sens <- read.table('Kaiterra_11May-10June_pm25.csv',sep=',',header=T)
kt.weat <- read.table('Kaiterra_11May-10June_weather.csv',sep=',',header=T)
kt.coord <- read.table('Kaiterra_11May-10June_coord.csv',sep=',',header=T)

str(kt.loc)
str(kt.coord)

kt.sens$ts <- as.POSIXct(strptime(kt.sens$ts,
                                  format='%Y-%m-%d %H:%M',
                                  tz="Asia/Kolkata")) # date/time ISO standard
# kt.sens$ts <- format(kt.sens$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat, no seconds
str(kt.sens)

kt.weat$ts <- as.POSIXct(strptime(kt.weat$ts,
                                  format='%Y-%m-%d %H:%M',
                                  tz="Asia/Kolkata")) # date/time ISO standard
# kt.sens$ts <- format(kt.sens$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat, no seconds
str(kt.weat)
# ^ hourly data only, no spatial info, for whole Delhi

nS <- dim(kt.loc)[1] # 18 locations
nT <- dim(kt.sens)[1] # 2881 time points
n <- nS*nT # 51'858, not accounting for NAs
# sum(is.na(kt.sens))/(nS*nT) # 36% NAs...

nS.full <- dim(kt.coord)[1] # 123 total nb locations
nS.mesh <- nS.full-nS # 105 extra locations
n.full <- nT*nS.full # 354'363, incl pred locations and not accounting for NAs


### create quadratic B-spline basis, assuming day is [0,1]
sb.dailygrid <- seq(0,1,length.out=96)
# ^ 24*4=96 obs per day

# kt.sens$ts[1:26]
# sb.dailygrid[1:26]
# ^ 2nd knot = 0.25 => first 24 obs in first sub-interval, ok

kn <- c(0,0.25,0.5,0.75,1) # fixed knots
# nb bases = J=6
# degree=2: nb knots = N = J-1, nb intervals = N-1 = J-2

sb <- bs(sb.dailygrid,degree=2,knots=kn,Boundary.knots=c(0,1))
sb <- sb[,-dim(sb)[2]] # extra col because I specify all knots

Bmat <- rep(1,30)%x%sb # 30 days + 15 min in tw
Bmat <- rbind(Bmat,Bmat[1,]) # additional 15 min for midnight on 10 June
# dim(Bmat) # ok only time, without space yet, dim(kt.sens)[1] = 2881

Bmat <- rep(1,nS.full)%x%Bmat # replicate for 123 locations
dim(Bmat) # ok n.full=354'363
# ^ clearly inefficient and RAM-consuming...
# ^ TODO for later: replicate within C++ template


### create map.delhi for mapping
range(kt.loc$lat)
range(kt.loc$lon)

corners.delhi <- list('topleft'=c(28.41, 77.04), # lat/lon
                      'botright'=c(28.64, 77.38)) # lat/lon
map.delhi <- openmap(upperLeft=corners.delhi[[1]],lowerRight=corners.delhi[[2]],
                     zoom=NULL,type='stamen-toner') # type='osm'

df.latlon <- SpatialPoints(kt.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
df.latlon.sp <- spTransform(df.latlon,osm())

plot(map.delhi,removeMargin=F)
points(df.latlon.sp,pch=19,col='blue')
axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
title(main="Kaiterra sensors Mar-Sep 2018, tw 2018-05-11 - 2018-06-10",
      xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')



### prep response vec and design for full spatial grid for X pred
coord.full <- as.matrix(kt.coord[,-3]) # extra locations at end, easier

proj.string <- "+proj=utm +zone=43 +ellps=WGS84 +north +units=km"
# ^ Delhi = UTM zone 43R
df.coord.full <- SpatialPoints(coord.full,proj4string=CRS(proj.string))
coord.full.osm <- spTransform(df.coord.full,map.delhi$tiles[[1]]$projection)

y.vec <- unlist(kt.sens[,-1]) # stack, grouped by location
logy.vec <- log(y.vec) # log response, NAs in correct places
logy.vec.full <- c(logy.vec,rep(NA,nS.mesh*nT)) # add NAs for extra locations

obsind.full <- as.integer(!is.na(logy.vec.full)) # 1=available, 0=missing value


distmat.full <- matrix(NA_real_,nS.full,nS.full) # Euclidean dist
for (i in 1:nS.full){
  distmat.full[i,i] <- 0
  j <- 1
  while (j<i){
    distmat.full[i,j] <- sqrt((coord.full[i,1]-coord.full[j,1])^2 + 
                                + (coord.full[i,2]-coord.full[j,2])^2)
    distmat.full[j,i] <- distmat.full[i,j] # symmetry
    j <- j+1
  }
}
distmat.full[1:5,1:5] # in km because utm coord in km from proj


### create zmat, correct dim by replicating covariate
str(kt.weat) # only 2881 rows, time only
zmat <- as.matrix(kt.weat[,-1])

zmat <- rep(1,nS.full)%x%zmat # replicate for 54 locations
dim(zmat) # ok n.full=155'574

# note: no intercept in zmat => intercept in seasonal component Bmat


### fit ML v0.5, intercept only for now, sampled+extra locations
datalist.full <- list()
datalist.full$log_y <- logy.vec.full
datalist.full$obsind <- obsind.full
datalist.full$zmat <- zmat
datalist.full$Bmat <- Bmat 
datalist.full$kn <- kn
datalist.full$distmat <- distmat.full
datalist.full$interceptonly <- 0L # covariates + seasonality

parlist.full <- list()
parlist.full$beta <- c(0,0,0) # dim = p = dim(zmat)[2]
parlist.full$alpha <- c(0.1,0.1,0,-0.1) # spline coeff, dim = J-1 = dim(Bmat)[2]-1
parlist.full$log_sigmaepsilon <- 0 # log(sigmaepsilon)
parlist.full$t_phi <- 1 # log((1+phi)/(1-phi)) # (exp(x)-1)/(exp(x)+1)
parlist.full$log_gamma <- 0 # log(gamma)
parlist.full$log_sigmadelta <- 0 # log(sigmadelta)
parlist.full$X <- rep(0,n.full) # logy.vec

system.time(obj.full <- MakeADFun(data=datalist.full,parameters=parlist.full,
                                  # map=list('beta'=factor(c(1,rep(factor(NA),4)))),
                                  random=c('X'),DLL="TGHM",silent=T))
# ^ 18s for nS.mesh=36 with interceptonly=1
# ^ 18s for nS.mesh=36 with interceptonly=0

system.time(print(obj.full$fn()))
# ^  90s for nS.mesh=36 with interceptonly=1
# ^ 103s for nS.mesh=36 with interceptonly=0

system.time(print(obj.full$gr()))
# ^ 15s for nS.mesh=36 with interceptonly=1
# ^ 16s for nS.mesh=36 with interceptonly=0


system.time(opt.full <- nlminb(start=obj.full$par,obj=obj.full$fn,gr=obj.full$gr,
                               control=list(eval.max=500,iter.max=500)))
# ^ 475s nS.full=54, nT=2881, interceptonly=1
# ^ 1292s nS.full=54, nT=2881, interceptonly=0
opt.full$mess # ok

system.time(rep.full <- sdreport(obj.full))
# ^ 217s nS.full=54, nT=2881, with interceptonly=1
# ^ 237s nS.full=54, nT=2881, with interceptonly=0

summ.rep.full <- summary(rep.full)
summ.rep.full[(11+n.full+1):dim(summ.rep.full)[1],]
# ^ se available!

# save.image('kt_11May-10June_TempFit_cs_nS.mesh.36.RData')


### compute PM2.5 predictions, interceptonly model
X.pred.full <- t(matrix(summ.rep.full[dimnames(summ.rep.full)[[1]]=='X',1],
                        nT,nS.full))
# X.se.full <- t(matrix(summ.rep.full[dimnames(summ.rep.full)[[1]]=='X',2],
#                       nT,nS.full))
# ^ original layout of data: one row per location, time points as cols
range(X.pred.full)
# ^ roughly [-2,+2.5]

detfx <- as.numeric(zmat%*%summ.rep.full[1:3,1])
season <- Bmat%*%summ.rep.full[(14+n.full+1):(19+n.full+1),1]

fixed.fx.full <- t(matrix(detfx+season,nT,nS.full)) # covariates + seasonality
# ^ fixed effects, constant through time
range(fixed.fx.full) # roughly [3.86,4.52]

pred.pm.full <- exp(fixed.fx.full+X.pred.full)
range(y.vec,na.rm=T)
range(pred.pm.full) # fairly close but much smaller max => spike smoothed out?


### plot PM2.5 predictions on top of Delhi map
lbub <- c(8,821) # bounds for color gradient, scale of pred.pm.full
mai.def <- c(1.02, 0.82, 0.82, 0.42)
mar.def <- c(5.1, 4.1, 4.1, 2.1)
alpha.colgrad <- 0.7
legend_image <- as.raster(matrix(colgreenred(100,alpha=alpha.colgrad)[100:1],
                                 ncol=1))

loc.lonlat <- SpatialPoints(kt.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
loc.osm <- spTransform(loc.lonlat,map.delhi$tiles[[1]]$projection) # re-project

ts2print <- format(kt.sens$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat, no seconds

# pdf('Kaiterra_11May-10June_STHMio_MapPredPM25.pdf',width=8,height=8,onefile=T)
# for (j in 1:nT){
#   layout(matrix(1:2,ncol=2),width=c(8,1),height=c(1,1)) # split plot region
#   par(mai=rep(1,4),mar=c(6,5,4,2))
#   plot(map.delhi,removeMargin=F)
#   intsurf <- interp(x=coord.full.osm@coords[,1],y=coord.full.osm@coords[,2],
#                     z=pred.pm.full[,j],nx=200,ny=200,linear=T)
#   image(intsurf$x,intsurf$y,intsurf$z,col=colgreenred(100,alpha=alpha.colgrad),
#         zlim=lbub,add=T)
#   points(loc.osm,pch=19,col='blue')
#   axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
#   axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
#   title(main=paste0("Kaiterra 11 May - 10 June, intercept only, predicted PM2.5"),
#         xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')
#   title(sub=ts2print[j],cex.sub=0.8,adj=1) # time stamp at bottomright
#   # legend as color bar
#   par(mar=c(5.1,1,4.1,1))
#   plot(c(0,2),c(0,1),type='n',axes=F,xlab ='',ylab='')
#   rasterImage(legend_image,xleft=0,ybottom=0,xright=2,ytop=1,angle=0)
#   text(x=1,y=seq(0.02,0.98,l=5),adj=0.5,cex=0.7,
#        labels=round(seq(lbub[1],lbub[2],l=5),2))
#   par(mai=mai.def,mar=mar.def) # back to default margins
#   layout(1) # default layout
# }
# dev.off()
# # ^ file way too heavy!


### separate png for video
lbub <- c(8,821) # bounds for color gradient, scale of pred.pm.full
mai.def <- c(1.02, 0.82, 0.82, 0.42)
mar.def <- c(5.1, 4.1, 4.1, 2.1)
alpha.colgrad <- 0.7
legend_image <- as.raster(matrix(colgreenred(100,alpha=alpha.colgrad)[100:1],
                                 ncol=1))

for (j in 1:nT){
  png(paste0('Outputs/VideoSeparatePNG/Kaiterra_11May-10June_STHMcs_MapPredPM25_',
             sprintf(j,fmt='%04d'),'.png'),
      width=8,height=6.5,res=200,units='in')
  layout(matrix(1:2,ncol=2),width=c(8,1),height=c(1,1)) # split plot region
  # par(mar=c(5.1,4.1,4.1,1))
  par(mai=rep(1,4),mar=c(6,5,4,2))
  plot(map.delhi,removeMargin=F)
  intsurf <- interp(x=coord.full.osm@coords[,1],y=coord.full.osm@coords[,2],
                    z=pred.pm.full[,j],nx=200,ny=200,linear=T)
  image(intsurf$x,intsurf$y,intsurf$z,col=colgreenred(100,alpha=alpha.colgrad),
        zlim=lbub,add=T)
  points(loc.osm,pch=19,col='blue')
  axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
  axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
  title(main=paste0("Kaiterra 11 May - 10 June, covariates+seasonality, predicted PM2.5"),
        xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')
  title(sub=ts2print[j],cex.sub=0.8,adj=1) # time stamp at bottomright
  # legend as color bar
  par(mar=c(5.1,1,4.1,1))
  plot(c(0,2),c(0,1),type='n',axes=F,xlab ='',ylab='')
  rasterImage(legend_image,xleft=0,ybottom=0,xright=2,ytop=1,angle=0)
  text(x=1,y=seq(0.02,0.98,l=5),adj=0.5,cex=0.7,
       labels=round(seq(lbub[1],lbub[2],l=5),2))
  par(mai=mai.def,mar=mar.def) # back to default margins
  layout(1) # default layout
  dev.off()
}

# ffmpeg -framerate 24 -i Kaiterra_11May-10June_STHMcs_MapPredPM25_%04d.png
#   ../Kaiterra_11May-10June_STHMcs_MapPredPM25.mp4

# 24 fps => 1 movie second = 6 real time hours


### plot of daily seasonal periodic effect
alphavec <- summ.rep.full[(14+n.full+1):(19+n.full+1),1] # dim J=6

kn <- c(0.00,0.25,0.50,0.75,1.00)

xgrid <- seq(0,1,0.01)

sb.grid <- bs(xgrid,degree=2,knots=kn,Boundary.knots=c(0,1))
sb.grid <- sb.grid[,-dim(sb.grid)[2]] # extra col because I specify all knots
sb.grid.rep <- rep(1,3)%x%sb.grid # repeat for visuals

dailyseason <- as.numeric(sb.grid%*%alphavec)
dailyseason.rep <- as.numeric(sb.grid.rep%*%alphavec)

pdf('Kaiterra_11May-10June_SeasonalEffect.pdf',width=8,height=7)
plot(c(seq(-1,0,0.01),seq(0,1,0.01),seq(1,2,0.01)),xlim=c(-0.3,1.3),
     dailyseason.rep,type='l',xaxt='n',xlab='Time (hours)',col='grey',lty=2,
     ylab='Predicted log(PM2.5) concentration',
     main='B-splines estimated daily seasonal effect')
abline(h=seq(3.9,4.5,0.1),lty=3,col='lightgrey')
abline(v=seq(-0.25,1.25,0.25),lty=3,col='lightgrey')
lines(xgrid,dailyseason)
axis(side=1,at=seq(-0.25,1.25,0.25),
     labels=c('18:00','00:00','06:00','12:00','18:00','00:00','06:00'))
dev.off()




#//////////////////////////////////////////////////////////////////////
#### 2018-11-10 Kaiterra csv 11May-10June, fit v0.5 intercept only ####
#//////////////////////////////////////////////////////////////////////

# intercept only = io

rm(list=ls())
paf2drop <- '/Users/WAWA/Desktop/Dropbox'
paf <- paste0(paf2drop,'/PostDoc/AirPollution/epod-nyu-delhi-pollution/spatiotemp')
setwd(paf)

library(TMB)
library(rgdal) # for lat/lon conversion to utm
library(OpenStreetMap) # for map downloading and plotting
library(splines) # for B-spline basis
library(akima) # for 2d interp

colgreenred <- function(n,alpha=1){ # based on heat.colors
  if ((n <- as.integer(n[1L])) > 0) {
    return(rainbow(n, s=1, v=1, start=0, end=3.5/6, alpha=alpha)[n:1]) # end=2/6
  } else {
    return(character())
  }
}
# plot(1:100,col=colgreenred(100),pch=19) # test color gradient


### create and load function from cpp template
# compile("TGHM.cpp")
dyn.load(dynlib("TGHM"))


### load csv data
kt.loc <- read.table('Kaiterra_11May-10June_loc.csv',sep=',',header=T)
kt.sens <- read.table('Kaiterra_11May-10June_pm25.csv',sep=',',header=T)
kt.weat <- read.table('Kaiterra_11May-10June_weather.csv',sep=',',header=T)
kt.coord <- read.table('Kaiterra_11May-10June_coord.csv',sep=',',header=T)

str(kt.loc)
str(kt.coord)

kt.sens$ts <- as.POSIXct(strptime(kt.sens$ts,
                                  format='%Y-%m-%d %H:%M',
                                  tz="Asia/Kolkata")) # date/time ISO standard
# kt.sens$ts <- format(kt.sens$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat, no seconds
str(kt.sens)

kt.weat$ts <- as.POSIXct(strptime(kt.weat$ts,
                                  format='%Y-%m-%d %H:%M',
                                  tz="Asia/Kolkata")) # date/time ISO standard
# kt.sens$ts <- format(kt.sens$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat, no seconds
str(kt.weat)
# ^ hourly data only, no spatial info, for whole Delhi

nS <- dim(kt.loc)[1] # 18 locations
nT <- dim(kt.sens)[1] # 2881 time points
n <- nS*nT # 51'858, not accounting for NAs
# sum(is.na(kt.sens))/(nS*nT) # 36% NAs...

nS.full <- dim(kt.coord)[1] # 123 total nb locations
nS.mesh <- nS.full-nS # 105 extra locations
n.full <- nT*nS.full # 354'363, incl pred locations and not accounting for NAs



### create map.delhi for mapping
range(kt.loc$lat)
range(kt.loc$lon)

corners.delhi <- list('topleft'=c(28.41, 77.04), # lat/lon
                      'botright'=c(28.64, 77.38)) # lat/lon
map.delhi <- openmap(upperLeft=corners.delhi[[1]],lowerRight=corners.delhi[[2]],
                     zoom=NULL,type='stamen-toner') # type='osm'

df.latlon <- SpatialPoints(kt.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
df.latlon.sp <- spTransform(df.latlon,osm())

plot(map.delhi,removeMargin=F)
points(df.latlon.sp,pch=19,col='blue')
axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
title(main="Kaiterra sensors Mar-Sep 2018, tw 2018-05-11 - 2018-06-10",
      xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')



### prep response vec and design for full spatial grid for X pred
coord.full <- as.matrix(kt.coord[,-3]) # extra locations at end, easier

proj.string <- "+proj=utm +zone=43 +ellps=WGS84 +north +units=km"
# ^ Delhi = UTM zone 43R
df.coord.full <- SpatialPoints(coord.full,proj4string=CRS(proj.string))
coord.full.osm <- spTransform(df.coord.full,map.delhi$tiles[[1]]$projection)

y.vec <- unlist(kt.sens[,-1]) # stack, grouped by location
logy.vec <- log(y.vec) # log response, NAs in correct places
logy.vec.full <- c(logy.vec,rep(NA,nS.mesh*nT)) # add NAs for extra locations

obsind.full <- as.integer(!is.na(logy.vec.full)) # 1=available, 0=missing value


distmat.full <- matrix(NA_real_,nS.full,nS.full) # Euclidean dist
for (i in 1:nS.full){
  distmat.full[i,i] <- 0
  j <- 1
  while (j<i){
    distmat.full[i,j] <- sqrt((coord.full[i,1]-coord.full[j,1])^2 + 
                                + (coord.full[i,2]-coord.full[j,2])^2)
    distmat.full[j,i] <- distmat.full[i,j] # symmetry
    j <- j+1
  }
}
distmat.full[1:5,1:5] # in km because utm coord in km from proj



### fit ML v0.5, intercept only for now, sampled+extra locations
datalist.full <- list()
datalist.full$log_y <- logy.vec.full
datalist.full$obsind <- obsind.full
datalist.full$zmat <- matrix(rep(1,4),2,2) # ignored if interceptonly=1
datalist.full$Bmat <- matrix(rep(1,4),2,2) # ignored if interceptonly=1
datalist.full$kn <- c(0,0.25,0.5,0.75,1) # ignored if interceptonly=1
datalist.full$distmat <- distmat.full
datalist.full$interceptonly <- 1L

parlist.full <- list()
parlist.full$beta <- c(5,0,0,0,0) # beta, only 1st entry if interceptonly=1
parlist.full$alpha <- c(0.1,0.2,0.3,0.4) # B-spline coeff, ignored if interceptonly=1
parlist.full$log_sigmaepsilon <- 0 # log(sigmaepsilon)
parlist.full$t_phi <- 1 # log((1+phi)/(1-phi)) # (exp(x)-1)/(exp(x)+1)
parlist.full$log_gamma <- 0 # log(gamma)
parlist.full$log_sigmadelta <- 0 # log(sigmadelta)
parlist.full$X <- rep(0,n.full) # logy.vec

system.time(obj.full <- MakeADFun(data=datalist.full,parameters=parlist.full,
                                  # map=list('beta'=factor(c(1,rep(factor(NA),4)))),
                                  random=c('X'),DLL="TGHM",silent=T)) # 45s
# map not needed in v0.5 with interceptonly=1
# ^ 130s for nS.mesh=123 and MBP starts swapping... rsession uses 3GB RAM
# ^ 18s for nS.mesh=36

system.time(print(obj.full$fn()))
# ^ 90s for nS.mesh=36

system.time(print(obj.full$gr()))
# ^ 15s for nS.mesh=36

system.time(opt.full <- nlminb(start=obj.full$par,obj=obj.full$fn,gr=obj.full$gr,
                               control=list(eval.max=500,iter.max=500)))
# ^ 475s nS.full=54, nT=2881
opt.full$mess # ok

system.time(rep.full <- sdreport(obj.full))
# ^ 217s nS.full=54, nT=2881
summ.rep.full <- summary(rep.full)
summ.rep.full[(13+n.full+1):dim(summ.rep.full)[1],]
# ^ se not feasible...

# save.image('kt_11May-10June_TempFit_interceptonly_nS.mesh.36.RData')


### compute PM2.5 predictions, interceptonly model
X.pred.full <- t(matrix(summ.rep.full[dimnames(summ.rep.full)[[1]]=='X',1],
                        nT,nS.full))
# X.se.full <- t(matrix(summ.rep.full[dimnames(summ.rep.full)[[1]]=='X',2],
#                       nT,nS.full))
# ^ original layout of data: one row per location, time points as cols
range(X.pred.full)
# ^ roughly [-2,+2.5]

fixed.fx.full <- t(matrix(summ.rep.full[1,1],nT,nS.full)) # intercept only
# ^ fixed effects, constant through time
range(fixed.fx.full) # constant

pred.pm.full <- exp(fixed.fx.full+X.pred.full)
range(y.vec,na.rm=T)
range(pred.pm.full) # fairly close but much smaller max => spike smoothed out?


### plot PM2.5 predictions on top of Delhi map
lbub <- c(8,816) # bounds for color gradient, scale of pred.pm.full
mai.def <- c(1.02, 0.82, 0.82, 0.42)
mar.def <- c(5.1, 4.1, 4.1, 2.1)
alpha.colgrad <- 0.7
legend_image <- as.raster(matrix(colgreenred(100,alpha=alpha.colgrad)[100:1],
                                 ncol=1))

loc.lonlat <- SpatialPoints(kt.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
loc.osm <- spTransform(loc.lonlat,map.delhi$tiles[[1]]$projection) # re-project

ts2print <- format(kt.sens$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat, no seconds

# pdf('Kaiterra_11May-10June_STHMio_MapPredPM25.pdf',width=8,height=8,onefile=T)
# for (j in 1:nT){
#   layout(matrix(1:2,ncol=2),width=c(8,1),height=c(1,1)) # split plot region
#   par(mai=rep(1,4),mar=c(6,5,4,2))
#   plot(map.delhi,removeMargin=F)
#   intsurf <- interp(x=coord.full.osm@coords[,1],y=coord.full.osm@coords[,2],
#                     z=pred.pm.full[,j],nx=200,ny=200,linear=T)
#   image(intsurf$x,intsurf$y,intsurf$z,col=colgreenred(100,alpha=alpha.colgrad),
#         zlim=lbub,add=T)
#   points(loc.osm,pch=19,col='blue')
#   axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
#   axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
#   title(main=paste0("Kaiterra 11 May - 10 June, intercept only, predicted PM2.5"),
#         xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')
#   title(sub=ts2print[j],cex.sub=0.8,adj=1) # time stamp at bottomright
#   # legend as color bar
#   par(mar=c(5.1,1,4.1,1))
#   plot(c(0,2),c(0,1),type='n',axes=F,xlab ='',ylab='')
#   rasterImage(legend_image,xleft=0,ybottom=0,xright=2,ytop=1,angle=0)
#   text(x=1,y=seq(0.02,0.98,l=5),adj=0.5,cex=0.7,
#        labels=round(seq(lbub[1],lbub[2],l=5),2))
#   par(mai=mai.def,mar=mar.def) # back to default margins
#   layout(1) # default layout
# }
# dev.off()
# ^ file way too heavy!


### separate png for video
lbub <- c(8,816) # bounds for color gradient, scale of pred.pm.full
mai.def <- c(1.02, 0.82, 0.82, 0.42)
mar.def <- c(5.1, 4.1, 4.1, 2.1)
alpha.colgrad <- 0.7
legend_image <- as.raster(matrix(colgreenred(100,alpha=alpha.colgrad)[100:1],
                                 ncol=1))

for (j in 1:nT){
  png(paste0('Outputs/VideoSeparatePNG/Kaiterra_11May-10June_STHMio_MapPredPM25_',
             sprintf(j,fmt='%04d'),'.png'),
      width=8,height=6.5,res=200,units='in')
  layout(matrix(1:2,ncol=2),width=c(8,1),height=c(1,1)) # split plot region
  # par(mar=c(5.1,4.1,4.1,1))
  par(mai=rep(1,4),mar=c(6,5,4,2))
  plot(map.delhi,removeMargin=F)
  intsurf <- interp(x=coord.full.osm@coords[,1],y=coord.full.osm@coords[,2],
                    z=pred.pm.full[,j],nx=200,ny=200,linear=T)
  image(intsurf$x,intsurf$y,intsurf$z,col=colgreenred(100,alpha=alpha.colgrad),
        zlim=lbub,add=T)
  points(loc.osm,pch=19,col='blue')
  axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
  axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
  title(main=paste0("Kaiterra 11 May - 10 June, intercept only, predicted PM2.5"),
        xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')
  title(sub=ts2print[j],cex.sub=0.8,adj=1) # time stamp at bottomright
  # legend as color bar
  par(mar=c(5.1,1,4.1,1))
  plot(c(0,2),c(0,1),type='n',axes=F,xlab ='',ylab='')
  rasterImage(legend_image,xleft=0,ybottom=0,xright=2,ytop=1,angle=0)
  text(x=1,y=seq(0.02,0.98,l=5),adj=0.5,cex=0.7,
       labels=round(seq(lbub[1],lbub[2],l=5),2))
  par(mai=mai.def,mar=mar.def) # back to default margins
  layout(1) # default layout
  dev.off()
}

# ffmpeg -framerate 16 -i Kaiterra_11May-10June_STHMio_MapPredPM25_%04d.png
#   ../Kaiterra_11May-10June_STHMio_MapPredPM25.mp4
# ^ -framerate 8 makes a very long video...








#//////////////////////////////////////////////////////////////////////////
#### 2018-11-09 Kaiterra csv 11May-10June, extra loc inlamesh save csv ####
#//////////////////////////////////////////////////////////////////////////

rm(list=ls())
paf2drop <- '/Users/WAWA/Desktop/Dropbox'
paf <- paste0(paf2drop,'/PostDoc/AirPollution/epod-nyu-delhi-pollution/spatiotemp')
setwd(paf)

library(INLA) # for mesh
library(rgdal) # for lat/lon conversion to utm
library(OpenStreetMap) # for map downloading and plotting

kt.loc <- read.table('Kaiterra_11May-10June_loc.csv',sep=',',header=T)
kt.sens <- read.table('Kaiterra_11May-10June_pm25.csv',sep=',',header=T)
kt.weat <- read.table('Kaiterra_11May-10June_weather.csv',sep=',',header=T)

str(kt.loc)

kt.sens$ts <- as.POSIXct(strptime(kt.sens$ts,
                                  format='%Y-%m-%d %H:%M',
                                  tz="Asia/Kolkata")) # date/time ISO standard
# kt.sens$ts <- format(kt.sens$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat, no seconds
str(kt.sens)

kt.weat$ts <- as.POSIXct(strptime(kt.weat$ts,
                                  format='%Y-%m-%d %H:%M',
                                  tz="Asia/Kolkata")) # date/time ISO standard
# kt.sens$ts <- format(kt.sens$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat, no seconds
str(kt.weat)
# ^ hourly data only, no spatial info, for whole Delhi

nS <- dim(kt.loc)[1] # 18 locations
nT <- dim(kt.sens)[1] # 2991 time points
n <- nS*nT # 51'858, not accounting for NAs

sum(is.na(kt.sens))/(nS*nT) # 36% NAs...



### use INLA to create grid of locations by Delaunay triangulation
coord <- kt.loc[,c('utmx','utmy')]
apply(coord,2,range) # determine borders of domain

mar.mesh <- 1.0 # beyond range of obs locations
coord.border <- data.frame('utmx'=c(min(coord$utmx)-mar.mesh,
                                    rep(max(coord$utmx)+mar.mesh,2),
                                    rep(min(coord$utmx)-mar.mesh,2)),
                           'utmy'=c(rep(min(coord$utmy)-mar.mesh,2),
                                    rep(max(coord$utmy)+mar.mesh,2),
                                    min(coord$utmy)-mar.mesh))
# ^ 4 corners, last=first to close domain, mar.mesh beyond observed range

system.time(inlamesh <- inla.mesh.2d(loc=coord, # coordinates in UTM
                                     loc.domain=coord.border,
                                     offset=1, #                  | offset=1
                                     max.edge=10,#                | max.edge=5
                                     min.angle=5,  #              | min.angle=20
                                     # max.n=1000, # overrides max.edge
                                     cutoff=0,
                                     plot.delay=NULL))
inlamesh$n
# ^ with nT=2881 and nS=18, 123 is too mcuh for my MBP RAM- and CPU-wise

plot(inlamesh)
lines(coord.border,lwd=3,col='blue')
points(coord$utmx,coord$utmy,pch=20,cex=1.5,col=2)

plot(inlamesh$loc[,1],inlamesh$loc[,2],pch=8,cex=1,col='deeppink')
lines(coord.border,lwd=3,col='blue') # well-spread?

coord.mesh <- data.frame('utmx'=inlamesh$loc[,1],'utmy'=inlamesh$loc[,2])
# ^ to be used for predictions and mapping

which(coord.mesh[,1]%in%coord[,1])
which(coord.mesh[,2]%in%coord[,2])
# ^ original points are included in the mesh locations, arbitrary position

coord.mesh <- coord.mesh[-which(coord.mesh[,1]%in%coord[,1]),]
str(coord.mesh)
points(coord.mesh[,1],coord.mesh[,2],pch=8,cex=1,col='limegreen')
# ^ 105 extra locations too much for my MBP

nS.mesh <- dim(coord.mesh)[1] # 105 extra locations
nS.full <- nS+nS.mesh # 123 total nb locations
n.full <- nT*nS.full # 354'363, incl pred locations and not accounting for NAs

coord.df <- data.frame(rbind(coord,coord.mesh))
coord.df$observed <- c(rep(1L,nS),rep(0L,dim(coord.mesh)[1]))


write.table(coord.df,file='Kaiterra_11May-10June_coord.csv',sep=',',
            row.names=F,col.names=T)







#/////////////////////////////////////////////////////////////////////////
#### 2018-11-09 Kaiterra 11May-10June, write weather data json to csv ####
#/////////////////////////////////////////////////////////////////////////

rm(list=ls())
paf2drop <- '/Users/WAWA/Desktop/Dropbox'
paf <- paste0(paf2drop,'/PostDoc/AirPollution/epod-nyu-delhi-pollution/spatiotemp')
setwd(paf)

kt.loc <- read.table('Kaiterra_11May-10June_loc.csv',sep=',',header=T)
kt.sens <- read.table('Kaiterra_11May-10June_pm25.csv',sep=',',header=T)

str(kt.loc)

kt.sens$ts <- as.POSIXct(strptime(kt.sens$ts,
                                  format='%Y-%m-%d %H:%M',
                                  tz="Asia/Kolkata")) # date/time ISO standard
# kt.sens$ts <- format(kt.sens$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat, no seconds
str(kt.sens)


### weather data from Ulzee
pafjsondata <- paste0(paf2drop,'/PostDoc/AirPollution/Data/KaiterraSensors2018/open_weather_newdelhi.json')

# library(rjson)
# 
# 
# jsondta <- fromJSON(file=pafjsondata,simplify=T)
# str(unlist(jsondta),1) # annoying to deal with

library(jsonlite)

wdta <- jsonlite::fromJSON(txt=pafjsondata)
str(wdta,1) # better, dataframe

table(wdta$city_id) # only 1 city id

str(wdta$main) # df for temp, pressure and humidity, 5649 obs
sum(is.na(wdta$main)) # no missing values
str(wdta$wind) # df for wind speed and degree
str(wdta$clouds) # df for some measure of cloudiness?
str(unlist(wdta$weather),1) # long list for mist, fog, haze
str(wdta$dt) # integer vector, some time stamp
str(wdta$dt_iso) # integer vector, time stamp UTC
str(wdta$rain)

### Summary weather data: 
#  - 1 obs per hour
#  - no spatial resolution, single obs for the whole city of Delhi
#  - temp, humidity, wind, etc.


### extract tw of 11May-10June, map to ts in sensor data
wdta$dt_iso[2205:2210]
# ^ careful: June 1st at 7am missing! => last obs carried forward

wdta$ts <- as.POSIXct(strptime(wdta$dt_iso,
                               format='%Y-%m-%d %H:%M:%S',
                               tz="UTC")) # date/time ISO standard
wdta$ts <- format(wdta$ts,format='%Y-%m-%d %H:%M:%S',
                             tz="Asia/Kolkata",usetz=T)
wdta$ts <- as.POSIXct(wdta$ts,format='%Y-%m-%d %H:%M:%S',
                      tz="Asia/Kolkata")
str(wdta$ts)
attr(wdta$ts,'tz')
# ^ tz important because hourly data and need to compare to kt.sens$ts
wdta$ts[1:3]

range(kt.sens$ts)

wdta.tw <- which(wdta$ts >= min(kt.sens$ts)-60*30 & wdta$ts <= max(kt.sens$ts))
length(wdta.tw) # 720 obs within tw, but one obs missing on June 1st

summary(wdta$ts[wdta.tw]) # annoying: hourly and shifted by 30 minutes...

wdta$ts[wdta.tw][1:5]

mapped.wdta.tw <- c(rep(wdta.tw[1],2),
                    rep(wdta.tw[2:517],each=4),
                    rep(wdta.tw[518],8), # missing June 1 at 12:30 => last obs
                    rep(wdta.tw[-(1:518)],each=4))
mapped.wdta.tw <- mapped.wdta.tw[-length(mapped.wdta.tw)] # one too many
length(mapped.wdta.tw)

cbind(as.character(kt.sens$ts[1:10]),
      as.character(wdta$ts[mapped.wdta.tw][1:10]))
cbind(as.character(kt.sens$ts[2872:2881]),
      as.character(wdta$ts[mapped.wdta.tw][2872:2881]))
# ^ now corresponds now to kt.sens$ts



### create csv of weather covariates

weather.df <- data.frame('ts'=wdta$ts[mapped.wdta.tw])
weather.df$temperature <- wdta$main$temp[mapped.wdta.tw]
weather.df$temperature <- weather.df$temperature-273.15 # convert to celsius
weather.df$humidity <- wdta$main$humidity[mapped.wdta.tw]
weather.df$windspeed <- wdta$wind$speed[mapped.wdta.tw]
# ^ stick to those for now

str(weather.df)
head(weather.df) # looks good

write.table(weather.df,file='Kaiterra_11May-10June_weather.csv',sep=',',
            row.names=F,col.names=T)




#///////////////////////////////////////////////////////////////////////////////
#### 2018-11-03 Kaiterra updated data Apr-Sep 2018 csv from Shiva, save txt ####
#///////////////////////////////////////////////////////////////////////////////

rm(list=ls())
paf2drop <- '/Users/WAWA/Desktop/Dropbox'
paf <- paste0(paf2drop,'/PostDoc/AirPollution/epod-nyu-delhi-pollution/spatiotemp')
setwd(paf)

pafdata <- '/Delhi Pollution/08_Data_Analysis/Output/csvFiles'


### import data from Shiva's csv, reformat timestamp, check nb sensors/locations
kt.full <- read.table(paste0(paf2drop,pafdata,'/kaiterra_panel_15min_2018_Sep_28.csv'),
                      sep=',',header=T)
str(kt.full)

kt.full$ts <- as.POSIXct(strptime(kt.full$timestamp_round,
                                  format='%Y-%m-%d %H:%M:%S',
                                  tz="Asia/Kolkata")) # date/time ISO standard
# kt.full$ts <- format(kt.full$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat no seconds
# ^ convert to correct class with IST time zone
range(kt.full$ts) # overview of full time period


table(kt.full$field_egg_id) # all same length because padded with NAs
length(unique(kt.full$field_egg_id))
names.loc <- as.character(unique(kt.full$field_egg_id))
nb.loc <- length(names.loc) # 22 locations
# ^ field_egg_id as unique identifier for location-sensor pairs, but Shiva
#   already merged disjoint sensor windows, so field_egg_id uniquely identifies
#   the 22 locations now.


### visualize available data points per sensor, highlight candidate tw
range(kt.full$ts)
tw.bd <- as.POSIXct(c('2018-05-11 00:00:00',
                      '2018-06-10 00:00:00'),tz="Asia/Kolkata")
# ^ tw suggested by Shiva

diff(tw.bd) # exactly 30 days
as.numeric(diff(tw.bd))*24*4
# ^ 30 days = 2880 intervals = 2881 values not accounting for NAs


pdf('Kaiterra_15min_FieldEggIdUptime.pdf',width=14,height=7)
# by location (= unique field_egg_id)
plot(x=tw.vizbd,y=c(1,nb.loc),type='n',
     xlab='Total time period',ylab='Location (field egg id)',yaxt='n',
     main='Kaiterra Apr-Jul 2018, up and down time by location')
axis(side=2,at=1:nb.loc,labels=names.loc,las=1)
for (j in 1:nb.loc){
  ind <- kt.full$field_egg_id==names.loc[j]
  ind.na <- is.na(kt.full$pm25[ind])
  arrows(x0=min(kt.full$ts[ind]),x1=max(kt.full$ts[ind]),y0=j,y1=j,
         col='grey',angle=90,code=3,length=0.05,lty=1)
  points(x=kt.full$ts[ind][!ind.na],y=rep(j,sum(!ind.na)),pch=19,cex=0.1)
}
abline(v=tw.bd,lty=2)
dev.off()

# ^ Visual inspection 11 May - 10 Jun up/down time per location:
#   drop 4 locations: DF07, E8E4, 8E2A, 113E
#   => 18 locations left


### create df lat/lon and map
discarded.loc <- c('DF07', 'E8E4', '8E2A', '113E')
kept.loc <- names.loc[!names.loc%in%discarded.loc] # 18 left

kt.loc <- data.frame('loc'=names.loc,stringsAsFactors=F)
table(round(kt.full$longitude,4)) # looks ok
table(round(kt.full$latitude,4)) # looks ok

for (j in 1:nb.loc){
  kt.loc$lon[j] <- kt.full$longitude[kt.full$field_egg_id==names.loc[j]][1]
  kt.loc$lat[j] <- kt.full$latitude[kt.full$field_egg_id==names.loc[j]][1]
}


### plot all locations on map of Delhi
library(sp)
library(rgdal)
library(OpenStreetMap)

range(kt.full$latitude)
range(kt.full$longitude)
corners.delhi <- list('topleft'=c(28.41, 77.04), # lat/lon
                      'botright'=c(28.64, 77.38)) # lat/lon
map.delhi <- openmap(upperLeft=corners.delhi[[1]],lowerRight=corners.delhi[[2]],
                     zoom=NULL,type='stamen-toner') # type='osm'

df.latlon <- SpatialPoints(kt.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
df.latlon.sp <- spTransform(df.latlon,osm())


pdf('Kaiterra_15min_Map_DiscardedSensors.pdf',width=9,height=7)
plot(map.delhi,removeMargin=F)
points(df.latlon.sp,pch=19,col=ifelse(names.loc%in%discarded.loc,'red','blue'))
text(names.loc,x=df.latlon.sp@coords[1:j,1],y=df.latlon.sp@coords[1:j,2],
     col=ifelse(names.loc%in%discarded.loc,'red','blue'),cex=0.5,
     pos=c(4,2,3,2,4,4,2,4,4,1,1,2,1,4,4,4,1,4,4,2,2,4))
axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
title(main="Kaiterra sensors Mar-Sep 2018, tw 2018-05-11 - 2018-06-10",
      xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')
legend('bottomright',c('Discarded','Kept'),pch=c(19,19),bg='white',
       col=c('red','blue'))
dev.off()


### 18 locations left: extract tw from kt.full, save data to txt (no RData)
kt <- subset(kt.full,subset=(kt.full$ts >= tw.bd[1] &
                               kt.full$ts <= tw.bd[2] &
                               kt.full$field_egg_id%in%kept.loc))
kt <- kt[,c('field_egg_id','ts','pm25')]
str(kt) # 2881*18 = 51858 obs

range(kt$ts)
tw.bd # identical by construction

tw <- seq(from=tw.bd[1],to=tw.bd[2],by='15 min') # common time stamp
tw <- format(tw,"%Y-%m-%d %H:%M",usetz=T) # reformat without seconds
length(tw) # 2881 values incl bounds
str(tw) # correct format, no seconds

kt.sens <- data.frame('ts'=tw)
for (j in 1:length(kept.loc)){
  ind <- kt$field_egg_id==kept.loc[j]
  kt.sens[[paste0('pm25_',kept.loc[j])]] <- kt$pm25[ind]
}
str(kt.sens)
head(kt.sens)
tail(kt.sens) # looks good

write.table(kt.sens,file='Kaiterra_11May-10June_pm25.csv',sep=',',
            row.names=F,col.names=T)
# ^ csv file of 2881 obs pm25, 19 cols:
#   - col 1: character string for time stamp
#   - col 2-19: numerical for pm25, one for each field_egg_id



### create df of 18 locations with lat/lon and UTM coord
df.latlon <- SpatialPoints(kt.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
proj.string <- "+proj=utm +zone=43 +ellps=WGS84 +north +units=km"
# ^ Delhi = UTM zone 43R
coord.utm <- spTransform(df.latlon, CRS(proj.string)) # re-project
kt.loc$utmx <- coord.utm@coords[,1]
kt.loc$utmy <- coord.utm@coords[,2]

str(kt.loc)

kt.loc <- kt.loc[names.loc%in%kept.loc,] # keep only kept loc

write.table(kt.loc,file='Kaiterra_11May-10June_loc.csv',sep=',',
            row.names=F,col.names=T)
# ^ csv file for 18 rows = 18 locations:
#  - col 1 = character string for field_egg_id (= unique location identifier)
#  - col 2 = numerical for longitude
#  - col 3 = numerical for latitude
#  - col 4 = numerical for projected UTM X
#  - col 5 = numerical for projected UTM Y


# ### save useful objects in envir
# save(list=c('kt.loc','kt.sens','tw.bd','names.sens',
#             'corners.delhi','map.delhi'),
#      file='kt_GoodTW_Padded.RData')






#////////////////////////////////////////////////////////////////////
#### 2018-10-23 Kaiterra updated data Apr-Sep 2018, find good tw ####
#////////////////////////////////////////////////////////////////////

rm(list=ls())
paf2drop <- '/Users/WAWA/Desktop/Dropbox'
paf <- paste0(paf2drop,'/PostDoc/AirPollution/epod-nyu-delhi-pollution/spatiotemp')
setwd(paf)

pafdta <- '/Delhi Pollution/08_Data_Analysis/dtaFiles/Aug_2018_Kaiterra/'

library(haven) # load Stata .dta via read_dta()

### load full data and explore
kt.full <- data.frame(na.omit(read_dta(
  paste0(paf2drop,pafdta,'pilot2_2018_panel_15min_28_Sep_2018.dta'),
  encoding='latin1')))
kt.full$ts <- as.POSIXct(strptime(kt.full$timestamp_round,
                                  format='%Y-%m-%d %H:%M:%S',
                                  tz="Asia/Kolkata")) # date/time ISO standard
kt.full$ts <- format(kt.full$ts,"%Y-%m-%d %H:%M",usetz=T) # reformat no seconds
# ^ convert to correct class with IST time zone

range(kt.full$ts) # overview of full time period

str(kt.full,1)

table(kt.full$loc_s_id)
length(unique(kt.full$loc_s_id))
# ^ loc_s_id as unique identifier for location-sensor pairs

# names.loc <- sprintf('%02g',unique(kt.full$loc_id))
names.loc <- as.character(unique(kt.full$loc_id))
names.sens <- unique(kt.full$s_id_short)
nb.loc <- length(names.loc) # 22 locations
nb.sens <- length(names.sens) # 23 sensors now overall


### visualize available data points per sensor
range(kt.full$ts)
tw.vizbd <- as.POSIXct(c('2018-03-01 00:00:00',
                         '2018-09-26 00:00:00'),tz="Asia/Kolkata")

tw.bd <- as.POSIXct(c('2018-05-11 00:00:00',
                      '2018-06-10 00:00:00'),tz="Asia/Kolkata")
# ^ tw suggested by Shiva

pdf('Kaiterra_15min_SensorsUptime.pdf',width=14,height=7)
par(mfrow=c(1,2))
# by sensor
plot(x=tw.vizbd,y=c(1,nb.sens),type='n', # x=range(kt.full$ts)
     xlab='Total time period',ylab='Sensor',yaxt='n',
     main='Kaiterra Mar-Sep 2018, up and down time by sensor')
axis(side=2,at=1:nb.sens,labels=names.sens,las=1)
for (j in 1:nb.sens){
  ind <- kt.full$s_id_short==names.sens[j]
  ind.na <- c(T,diff(kt.full$ts[ind])!=15) # contiguous = time lag 15
  arrows(x0=min(kt.full$ts[ind]),x1=max(kt.full$ts[ind]),y0=j,y1=j,
         col='grey',angle=90,code=3,length=0.05,lty=1)
  points(x=kt.full$ts[ind],y=rep(j,sum(ind)),pch=19,cex=0.1)
  points(x=kt.full$ts[ind][ind.na],y=rep(j,sum(ind.na)),pch="|",col='red')
  message(names.sens[j],' | ',min(kt.full$ts[ind]),' - ',max(kt.full$ts[ind]))
}
abline(v=tw.bd,lty=2)
legend('topleft',c('downtime','uptime','down -> up'),
       pch=c(NA,19,NA),pt.cex=c(1,0.4,1),lty=c(1,NA,1),col=c('grey',1,'red'))

# by location
plot(x=tw.vizbd,y=c(1,nb.loc),type='n',
     xlab='Total time period',ylab='Location',yaxt='n',
     main='Kaiterra Apr-Jul 2018, up and down time by location')
axis(side=2,at=1:nb.loc,labels=names.loc,las=1)
for (j in 1:nb.loc){
  ind <- kt.full$loc_id==names.loc[j]
  ind.na <- c(T,diff(kt.full$ts[ind])!=15) # contiguous = time lag 15
  arrows(x0=min(kt.full$ts[ind]),x1=max(kt.full$ts[ind]),y0=j,y1=j,
         col='grey',angle=90,code=3,length=0.05,lty=1)
  points(x=kt.full$ts[ind],y=rep(j,sum(ind)),pch=19,cex=0.1)
  points(x=kt.full$ts[ind][ind.na],y=rep(j,sum(ind.na)),pch="|",col='red')
  message(names.loc[j],' | ',min(kt.full$ts[ind]),' - ',max(kt.full$ts[ind]))
}
abline(v=tw.bd,lty=2)
legend('topleft',c('downtime','uptime','down -> up'),
       pch=c(NA,19,NA),pt.cex=c(1,0.4,1),lty=c(1,NA,1),col=c('grey',1,'red'))

# ^ candidate tw: mid-May and mid-July
par(mfrow=c(1,1))
dev.off()


### tw suggested by Shiva in email 2018-10-23
# Removed ('01_FBCB', '07_59F2', '09_8220', '16_407D', '17_CEDB') out of 25 sensors
# 2018-05-11, 30 days: 24442/38420 (valid / total)

tw.bd <- as.POSIXct(c('2018-05-11 00:00:00',
                      '2018-06-10 00:00:00'),tz="Asia/Kolkata")

diff(tw.bd) # exactly 30 days
as.numeric(diff(tw.bd))*24*4
# ^ 30 days = 2880 values not accounting for NAs

discarded.sloc <- c('01_FBCB','07_59F2','09_8220','16_407D','17_CEDB','22_3384')
discarded.sens <- names.sens%in%c('FBCB','59F2','8220','407D','CEDB','3384')
discarded.loc <- names.loc%in%c('1','7','9','16','17','22')

# show tw and 4 discarded sensors
pdf('Kaiterra_15min_TW_DiscardedSensors.pdf',width=7,height=7)
plot(x=tw.bd,y=c(1,nb.sens),type='n', # x=tw.vizbd
     xlab='Restricted time period',ylab='Sensor',yaxt='n',
     main='Kaiterra Mar-Sep 2018, suggested tw and discarded sensors')
axis(side=2,at=1:nb.sens,labels=NA,las=1)
mtext(side=2,at=1:nb.sens,names.sens,las=1,line=0.7,
      col=ifelse(discarded.sens,'red','blue'))
# ^ discared sensors in red
for (j in 1:nb.sens){
  ind <- kt.full$s_id_short==names.sens[j]
  ind.na <- c(T,diff(kt.full$ts[ind])!=15) # contiguous = time lag 15
  arrows(x0=min(kt.full$ts[ind]),x1=max(kt.full$ts[ind]),y0=j,y1=j,
         col='grey',angle=90,code=3,length=0.05,lty=1)
  points(x=kt.full$ts[ind],y=rep(j,sum(ind)),pch=19,cex=0.1)
  points(x=kt.full$ts[ind][ind.na],y=rep(j,sum(ind.na)),pch="|",col='red')
}
abline(v=tw.bd,lty=2)
dev.off()



### create df of locations with lat/lon
kt.loc <- data.frame('loc'=names.loc,stringsAsFactors=F)

table(round(kt.full$longitude,4)) # looks ok
table(round(kt.full$latitude,4)) # looks ok

for (j in 1:nb.loc){
  kt.loc$lon[j] <- kt.full$longitude[kt.full$loc_id==names.loc[j]][1]
  kt.loc$lat[j] <- kt.full$latitude[kt.full$loc_id==names.loc[j]][1]
}


### plot all locations on map of Delhi
library(sp)
library(rgdal)
library(OpenStreetMap)

range(kt.full$latitude)
range(kt.full$longitude)
corners.delhi <- list('topleft'=c(28.41, 77.04), # lat/lon
                      'botright'=c(28.64, 77.38)) # lat/lon
map.delhi <- openmap(upperLeft=corners.delhi[[1]],lowerRight=corners.delhi[[2]],
                     zoom=NULL,type='stamen-toner') # type='osm'

df.latlon <- SpatialPoints(kt.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
df.latlon.sp <- spTransform(df.latlon,osm())


pdf('Kaiterra_15min_Map_DiscardedSensors.pdf',width=9,height=7)
plot(map.delhi,removeMargin=F)
points(df.latlon.sp,pch=19,col=ifelse(discarded.loc,'red','blue')) # our sensors
text(names.loc,x=df.latlon.sp@coords[1:j,1],y=df.latlon.sp@coords[1:j,2],
     col=ifelse(discarded.loc,'red','blue'),cex=0.5,
     pos=c(4,2,3,2,4,4,2,4,4,1,1,2,1,4,4,4,1,4,4,2,2,4))
axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
title(main="Kaiterra sensors Mar-Sep 2018, tw 2018-05-11 - 2018-06-10",
      xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')
legend('bottomright',c('Discarded','Kept'),pch=c(19,19),bg='white',
       col=ifelse(discarded.loc,'red','blue'))
dev.off()









#//////////////////////////////////////////////////////////////////////
#### 2018-08-09 Kaiterra sensors, fit STHM to July padded tw, v0.4 ####
#//////////////////////////////////////////////////////////////////////

# exact same model as in Ubicomp paper, just different data
# new area = 11 locations, 1729 time points

rm(list=ls())
paf2drop <- '/Users/WAWA/Desktop/Dropbox'
paf <- paste0(paf2drop,'/PostDoc/AirPollution/Data/KaiterraSensors2018')
setwd(paf)

library(TMB)
library(INLA) # for mesh
library(rgdal) # for lat/lon conversion to utm
library(akima) # for 2d interp
library(OpenStreetMap) # for map downloading and plotting

colgreenred <- function(n,alpha=1){ # based on heat.colors
  if ((n <- as.integer(n[1L])) > 0) {
    return(rainbow(n, s=1, v=1, start=0, end=3.5/6, alpha=alpha)[n:1]) # end=2/6
  } else {
    return(character())
  }
}
# plot(1:100,col=colgreenred(100),pch=19)
# plot(1:100,col=rainbow(100, s=1, v=1, start=0, end=3.5/6)[100:1],pch=19)
colgrad <- function(x,lb=NULL,ub=NULL,n=100,alpha=1){
  # n = resolution of colour gradient between green and red
  if (length(x)==1){
    if (is.null(lb) | is.null(ub)){
      stop('lb and ub must be specified if length(x)=1.')
    }
    if (x<lb | x>ub){stop('x beyond range of c(lb,ub).')}
    ind <- round((x-lb)/(ub-lb)*n)
    ind <- ifelse(ind==0,1,ind)
    colvec <- colgreenred(n,alpha=alpha)[ind]
  } else {
    if (is.null(lb) | is.null(ub)){ # if one bound missing, then use range(x)
      lb <- min(x)
      ub <- max(x)
    } else {
      if (any(x<lb) | any(x>ub)){stop('x beyond range of c(lb,ub).')}
    }
    colvec <- sapply(x,function(x){
      ind <- round((x-lb)/(ub-lb)*n)
      colgreenred(n,alpha=alpha)[ifelse(ind==0,1,ind)]
    })
  }
  # return(ifelse(colvec==0,1,colvec))
  return(colvec)
}



### create and load function from cpp template
# compile("TGHM.cpp")
dyn.load(dynlib("TGHM"))


### load RData of padded PP2 data with good tw
load(file='kt_GoodTW_Padded.RData')

kt.sens$ts <- format(kt.sens$ts,"%Y-%m-%d %H:%M",usetz=T)

str(kt.sens) # 22 sensors, padded with NAs
str(kt.loc)  # info about all locations 

nS <- dim(kt.loc)[1] # 11 locations now in new area
nT <- dim(kt.sens)[1] # 1729 time points
n <- nS*nT # 19'019, not accounting for NAs

sum(is.na(kt.sens))/(nS*nT) # 32% NAs...


### plot map of locations
loc.lonlat <- SpatialPoints(kt.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
loc.osm <- spTransform(loc.lonlat,map.delhi$tiles[[1]]$projection) # re-project

pdf('Kaiterra_15min_MapSensors.pdf',width=7,height=7)
plot(map.delhi,removeMargin=F)
points(loc.osm,pch=19,col='red') # Kaiterra's 22 sensors
text(names.sens,x=loc.osm@coords[,1],y=loc.osm@coords[,2],
     col='red',cex=0.8,pos=c(3,2,3,1,2,4,4,1,3,4,4))
# pos=c(4,2,3,2,4,4,2,4,4,1,1,2,1,4,4,4,1,4,4,2,2,4)
axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
title(main="Kaiterra 11 sensors left",
      xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')
dev.off()


### use INLA to create grid of locations by Delaunay triangulation
coord <- kt.loc[,c('utmx','utmy')]
apply(coord,2,range) # determine borders of domain

mar.mesh <- 1.0 # beyond range of obs locations
coord.border <- data.frame('utmx'=c(min(coord$utmx)-mar.mesh,
                                    rep(max(coord$utmx)+mar.mesh,2),
                                    rep(min(coord$utmx)-mar.mesh,2)),
                           'utmy'=c(rep(min(coord$utmy)-mar.mesh,2),
                                    rep(max(coord$utmy)+mar.mesh,2),
                                    min(coord$utmy)-mar.mesh))
# ^ 4 corners, last=first to close domain, mar.mesh beyond observed range

system.time(inlamesh <- inla.mesh.2d(loc=coord, # coordinates in UTM
                                     loc.domain=coord.border,
                                     offset=1, #                  | offset=1
                                     max.edge=2, #                | max.edge=5
                                     min.angle=20, #              | min.angle=20
                                     # max.n=1000, # overrides max.edge
                                     cutoff=0,
                                     plot.delay=NULL))
inlamesh$n # not too many locations if possible

plot(inlamesh)
lines(coord.border,lwd=3,col='blue')
points(coord$utmx,coord$utmy,pch=20,cex=1.5,col=2)

plot(inlamesh$loc[,1],inlamesh$loc[,2],pch=8,cex=1,col='deeppink')
lines(coord.border,lwd=3,col='blue') # well-spread?

coord.mesh <- data.frame('utmx'=inlamesh$loc[,1],'utmy'=inlamesh$loc[,2])
# ^ to be used for predictions and mapping

which(coord.mesh[,1]%in%coord[,1])
which(coord.mesh[,2]%in%coord[,2])
# ^ original points are included in the mesh locations, arbitrary position

coord.mesh <- coord.mesh[-which(coord.mesh[,1]%in%coord[,1]),]
str(coord.mesh)
points(coord.mesh[,1],coord.mesh[,2],pch=8,cex=1,col='limegreen')
# ^ only 100 locations left, only the extra ones

nS.mesh <- dim(coord.mesh)[1]
nS.full <- nS+nS.mesh
n.full <- nT*nS.full # 191'919, incl pred locations and not accounting for NAs
p <- 5 # length(beta), fixed design as in v0.3
q <- p+4


### prep response vec and design for full spatial grid for X pred
coord.full <- rbind(coord,coord.mesh) # extra locations at end, easier

proj.string <- "+proj=utm +zone=43 +ellps=WGS84 +north +units=km"
# ^ Delhi = UTM zone 43R
df.coord.full <- SpatialPoints(coord.full,proj4string=CRS(proj.string))
coord.full.osm <- spTransform(df.coord.full,map.delhi$tiles[[1]]$projection)


y.vec <- unlist(kt.sens[,-1]) # stack, grouped by location
logy.vec <- log(y.vec) # log response, NAs in correct places
logy.vec.full <- c(logy.vec,rep(NA,nS.mesh*nT)) # add NAs for extra locations

obsind.full <- as.integer(!is.na(logy.vec.full)) # 1=available, 0=missing value

# coord.full.s <- scale(coord.full) # scaled, mean=0 and sd=1
# zmat.full <- cbind(rep(1,nS.full),coord.full.s,coord.full.s^2) # fixed, v0.3
# dimnames(zmat.full)[[2]] <- c('intercept','utmx','utmy','utmx2','utmy2')
# zmat.full.rep <- matrix(rep(zmat.full,each=nT),n.full,p)
# str(zmat.full.rep)

distmat.full <- matrix(NA_real_,nS.full,nS.full) # Euclidean dist
for (i in 1:nS.full){
  distmat.full[i,i] <- 0
  j <- 1
  while (j<i){
    distmat.full[i,j] <- sqrt((coord.full[i,1]-coord.full[j,1])^2 + 
                                + (coord.full[i,2]-coord.full[j,2])^2)
    distmat.full[j,i] <- distmat.full[i,j] # symmetry
    j <- j+1
  }
}
distmat.full[1:5,1:5] # in km because utm coord in km from proj


### fit ML with intercept only v0.4, sampled+extra locations
datalist.full <- list()
datalist.full$log_y <- logy.vec.full
datalist.full$obsind <- obsind.full
datalist.full$zmat <- matrix(rep(1,4),2,2) # ignored if interceptonly=1
datalist.full$distmat <- distmat.full
datalist.full$interceptonly <- 1L

parlist.full <- list()
parlist.full$beta <- c(5,0,0,0,0) # beta, only 1st entry if interceptonly=1
parlist.full$log_sigmaepsilon <- 0 # log(sigmaepsilon)
parlist.full$t_phi <- 1 # log((1+phi)/(1-phi)) # (exp(x)-1)/(exp(x)+1)
parlist.full$log_gamma <- 0 # log(gamma)
parlist.full$log_sigmadelta <- 0 # log(sigmadelta)
parlist.full$X <- rep(0,n.full) # logy.vec

system.time(obj.full <- MakeADFun(data=datalist.full,parameters=parlist.full,
                                  # map=list('beta'=factor(c(1,rep(factor(NA),4)))),
                                  random=c('X'),DLL="TGHM",silent=T)) # 45s
# ^ 52s, map not needed in v0.4 with interceptonly=1

system.time(opt.full <- nlminb(start=obj.full$par,obj=obj.full$fn,gr=obj.full$gr,
                                control=list(eval.max=1000,iter.max=1000)))
# ^ 2546s nS.full=111, nT=1729
opt.full$mess # ok

# save.image('kt_fit_temp.RData')

system.time(rep.full <- sdreport(obj.full))
# ^ 605s nS.full=111, nT=1729
summ.rep.full <- summary(rep.full)
summ.rep.full[(q+n.full+1):dim(summ.rep.full)[1],]
# ^ se not feasible?

X.pred.full <- t(matrix(summ.rep.full[dimnames(summ.rep.full)[[1]]=='X',1],
                         nT,nS.full))
X.se.full <- t(matrix(summ.rep.full[dimnames(summ.rep.full)[[1]]=='X',2],
                       nT,nS.full))
# ^ original layout of data: one row per location, time points as cols
range(X.pred.full)
# ^ rather large range


### compute PM2.5 predictions
fixed.fx.full <- t(matrix(summ.rep.full[1,1],nT,nS.full)) # intercept only
# ^ fixed effects, constant through time
range(fixed.fx.full) # constant

pred.pm.full <- exp(fixed.fx.full+X.pred.full)
range(y.vec,na.rm=T)
range(pred.pm.full) # fairly close but much smaller max => spike smoothed out?


### plot PM2.5 predictions on top of Delhi map
lbub <- c(5,280) # bounds for color gradient, scale of pred.pm.full
mai.def <- c(1.02, 0.82, 0.82, 0.42)
mar.def <- c(5.1, 4.1, 4.1, 2.1)
alpha.colgrad <- 0.7
legend_image <- as.raster(matrix(colgreenred(100,alpha=alpha.colgrad)[100:1],
                                 ncol=1))


pdf('Outputs/kt_STHM_MapPredPM25.pdf',width=8,height=8,onefile=T)
for (j in 1:nT){
  layout(matrix(1:2,ncol=2),width=c(8,1),height=c(1,1)) # split plot region
  par(mai=rep(1,4),mar=c(6,5,4,2))
  plot(map.delhi,removeMargin=F)
  intsurf <- interp(x=coord.full.osm@coords[,1],y=coord.full.osm@coords[,2],
                    z=pred.pm.full[,j],nx=200,ny=200,linear=T)
  image(intsurf$x,intsurf$y,intsurf$z,col=colgreenred(100,alpha=alpha.colgrad),
        zlim=lbub,add=T)
  points(loc.osm,pch=19,col='blue')
  axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
  axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
  title(main=paste0("Kaiterra intercept only predicted PM2.5"),
        xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')
  title(sub=kt.sens$ts[j],cex.sub=0.8,adj=1) # time stamp at bottomright
  # legend as color bar
  par(mar=c(5.1,1,4.1,1))
  plot(c(0,2),c(0,1),type='n',axes=F,xlab ='',ylab='')
  rasterImage(legend_image,xleft=0,ybottom=0,xright=2,ytop=1,angle=0)
  text(x=1,y=seq(0.02,0.98,l=5),adj=0.5,cex=0.7,
       labels=round(seq(lbub[1],lbub[2],l=5),2))
  par(mai=mai.def,mar=mar.def) # back to default margins
  layout(1) # default layout
}
dev.off()


### separate png for video
lbub <- c(5,280) # bounds for color gradient, scale of pred.pm.full
mai.def <- c(1.02, 0.82, 0.82, 0.42)
mar.def <- c(5.1, 4.1, 4.1, 2.1)
alpha.colgrad <- 0.7
legend_image <- as.raster(matrix(colgreenred(100,alpha=alpha.colgrad)[100:1],
                                 ncol=1))

for (j in 1:nT){
  png(paste0('Outputs/VideoSeparatePNG/kt_STHM_MapPredPM25_',sprintf(j,fmt='%04d'),'.png')
      ,width=8,height=8,res=200,units='in')
  layout(matrix(1:2,ncol=2),width=c(8,1),height=c(1,1)) # split plot region
  # par(mar=c(5.1,4.1,4.1,1))
  par(mai=rep(1,4),mar=c(6,5,4,2))
  plot(map.delhi,removeMargin=F)
  intsurf <- interp(x=coord.full.osm@coords[,1],y=coord.full.osm@coords[,2],
                    z=pred.pm.full[,j],nx=200,ny=200,linear=T)
  image(intsurf$x,intsurf$y,intsurf$z,col=colgreenred(100,alpha=alpha.colgrad),
        zlim=lbub,add=T)
  points(loc.osm,pch=19,col='blue')
  axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
  axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
  title(main=paste0("Kaiterra intercept only predicted PM2.5"),
        xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')
  title(sub=kt.sens$ts[j],cex.sub=0.8,adj=1) # time stamp at bottomright
  # legend as color bar
  par(mar=c(5.1,1,4.1,1))
  plot(c(0,2),c(0,1),type='n',axes=F,xlab ='',ylab='')
  rasterImage(legend_image,xleft=0,ybottom=0,xright=2,ytop=1,angle=0)
  text(x=1,y=seq(0.02,0.98,l=5),adj=0.5,cex=0.7,
       labels=round(seq(lbub[1],lbub[2],l=5),2))
  par(mai=mai.def,mar=mar.def) # back to default margins
  layout(1) # default layout
  dev.off()
}
# ffmpeg -framerate 8 -i kt_STHM_MapPredPM25_%04d.png ../kt_STHM_MapPredPM25.mp4


# save.image('kt_STHM_MapPredPM25.RData')








#//////////////////////////////////////////////////////////////////////////////
#### 2018-08-09 Kaiterra sensors April-July 2018, find good tw, save RData ####
#//////////////////////////////////////////////////////////////////////////////

rm(list=ls())
paf2drop <- '/Users/WAWA/Desktop/Dropbox'
paf <- paste0(paf2drop,'/PostDoc/AirPollution/Data/KaiterraSensors2018')
setwd(paf)

library(haven) # load Stata .dta via read_dta()

### load data and explore
kt.full <- na.omit(read_dta('pilot2_2018_panel_15min__8_Aug_2018.dta',
                            encoding='latin1'))
kt.full$ts <- as.POSIXct(strptime(kt.full$timestamp_round,
                                  format='%Y-%m-%d %H:%M:%S',
                                  tz="Asia/Kolkata"))
# ^ convert to correct class with IST time zone

kt.full$ts <- format(kt.full$ts,"%Y-%m-%d %H:%M",usetz=T)
# ^ good format: midnight displayed as 00:00, and seconds dropped

summary(kt.full$ts) # overview of whole time period

table(kt.full$loc_id)
length(unique(kt.full$loc_id))
# ^ loc_id (1--22) as identifier for location
table(kt.full$loc_s_id)
length(unique(kt.full$loc_s_id))
# ^ loc_s_id as unique identifier for location-sensor pairs

length(unique(kt.full$location))
length(unique(kt.full$loc_id_short))
length(unique(kt.full$s_id_short))
# ^ 23 sensors, 22 locations

# names.loc <- sprintf('%02g',unique(kt.full$loc_id))
names.loc <- as.character(unique(kt.full$loc_id_short)) # unique(kt.full$loc_id)
names.sens <- unique(kt.full$s_id_short)
nb.loc <- length(names.loc) # 22 locations
nb.sens <- length(names.sens) # 23 sensors now overall

for (j in 1:nb.loc){
  message('location ',names.loc[j])
  print(unique(kt.full$s_id_short[kt.full$loc_id_short==names.loc[j]]))
}
# ^ 25 sensor-location pairs, some sensors reused elsewhere

### visualize available data points per sensor
range(kt.full$ts) # starts much earlier than previous versions of data

tw.vizbd <- as.POSIXct(c('2018-04-01 00:00:00',
                         '2018-08-01 00:00:00'),tz="Asia/Kolkata")

pdf('Kaiterra_15min_SensorsUptime.pdf',width=14,height=7)
par(mfrow=c(1,2))
# by sensor
plot(x=tw.vizbd,y=c(1,nb.sens),type='n', # x=range(kt.full$ts)
     xlab='Total time period',ylab='Sensor',yaxt='n',
     main='Kaiterra Apr-Jul 2018, up and down time by sensor')
axis(side=2,at=1:nb.sens,labels=names.sens,las=1)
for (j in 1:nb.sens){
  ind <- kt.full$s_id_short==names.sens[j]
  ind.na <- c(T,diff(kt.full$ts[ind])!=15) # contiguous = time lag 15
  arrows(x0=min(kt.full$ts[ind]),x1=max(kt.full$ts[ind]),y0=j,y1=j,
         col='grey',angle=90,code=3,length=0.05,lty=1)
  points(x=kt.full$ts[ind],y=rep(j,sum(ind)),pch=19,cex=0.1)
  points(x=kt.full$ts[ind][ind.na],y=rep(j,sum(ind.na)),pch="|",col='red')
  message(names.sens[j],' | ',min(kt.full$ts[ind]),' - ',max(kt.full$ts[ind]))
}
legend('topleft',c('downtime','uptime','down -> up'),
       pch=c(NA,19,NA),pt.cex=c(1,0.4,1),lty=c(1,NA,1),col=c('grey',1,'red'))
# by location
plot(x=tw.vizbd,y=c(1,nb.loc),type='n', # x=range(kt.full$ts)
     xlab='Total time period',ylab='Location',yaxt='n',
     main='Kaiterra Apr-Jul 2018, up and down time by location')
axis(side=2,at=1:nb.loc,labels=names.loc,las=1)
for (j in 1:nb.loc){
  ind <- kt.full$loc_id_short==names.loc[j]
  ind.na <- c(T,diff(kt.full$ts[ind])!=15) # contiguous = time lag 15
  arrows(x0=min(kt.full$ts[ind]),x1=max(kt.full$ts[ind]),y0=j,y1=j,
         col='grey',angle=90,code=3,length=0.05,lty=1)
  points(x=kt.full$ts[ind],y=rep(j,sum(ind)),pch=19,cex=0.1)
  points(x=kt.full$ts[ind][ind.na],y=rep(j,sum(ind.na)),pch="|",col='red')
  message(names.loc[j],' | ',min(kt.full$ts[ind]),' - ',max(kt.full$ts[ind]))
}
# ^ candidate tw, both with 13 sensors ~continuous: mid-May and mid-July
par(mfrow=c(1,1))
dev.off()

tw.bd <- as.POSIXct(c('2018-07-06 00:00:00',
                      '2018-07-24 00:00:00'),tz="Asia/Kolkata")

diff(tw.bd)
as.numeric(diff(tw.bd))*24*4
# ^ 18 days, 1728 values not accounting NAs, not bad

 
### create df of locations with lat/lon
kt.loc <- data.frame('loc'=names.loc,stringsAsFactors=F)

table(round(kt.full$longitude,4)) # looks ok
table(round(kt.full$latitude,4)) # looks ok

for (j in 1:nb.loc){
  kt.loc$lon[j] <- kt.full$longitude[kt.full$loc_id_short==names.loc[j]][1]
  kt.loc$lat[j] <- kt.full$latitude[kt.full$loc_id_short==names.loc[j]][1]
}


### plot all locations on map of Delhi
library(sp)
library(rgdal)
library(OpenStreetMap)

range(kt.full$latitude)
range(kt.full$longitude)
corners.delhi <- list('topleft'=c(28.64, 77.04), # lat/lon
                      'botright'=c(28.40, 77.38)) # lat/lon
map.delhi <- openmap(upperLeft=corners.delhi[[1]],lowerRight=corners.delhi[[2]],
                     zoom=NULL,type='stamen-toner') # type='osm'

df.latlon <- SpatialPoints(kt.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
df.latlon.sp <- spTransform(df.latlon,osm())

j <- 22
plot(map.delhi,removeMargin=F)
points(df.latlon.sp,pch=19,col='red') # our sensors
text(names.loc[1:j],x=df.latlon.sp@coords[1:j,1],y=df.latlon.sp@coords[1:j,2],
     col='red',cex=0.5,pos=c(4,2,3,2,4,4,2,4,4,1,1,2,1,4,4,4,1,4,4,2,2,4))
axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
title(main="Kaiterra sensors Apr-Jul 2018, 22 locations",
      xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')

# unique(kt.full$location[kt.full$loc_id=='13'])
# head(kt.full[kt.full$loc_id=='13',c('longitude','latitude')])
# unique(kt.full$location[kt.full$loc_id=='21'])
# head(kt.full[kt.full$loc_id=='21',c('longitude','latitude')])
# ^ loc numbering 1--22 doesn't Sameeksha's report...



### exclude locations, restrict area
plot(x=tw.vizbd,y=c(1,nb.loc),type='n', # x=range(kt.full$ts)
     xlab='Total time period',ylab='Location',yaxt='n',
     main='Kaiterra Apr-Jul 2018, up and down time by location')
axis(side=2,at=1:nb.loc,labels=names.loc,las=1)
for (j in 1:nb.loc){
  ind <- kt.full$loc_id_short==names.loc[j]
  ind.na <- c(T,diff(kt.full$ts[ind])!=15) # contiguous = time lag 15
  arrows(x0=min(kt.full$ts[ind]),x1=max(kt.full$ts[ind]),y0=j,y1=j,
         col='grey',angle=90,code=3,length=0.05,lty=1)
  points(x=kt.full$ts[ind],y=rep(j,sum(ind)),pch=19,cex=0.1)
  points(x=kt.full$ts[ind][ind.na],y=rep(j,sum(ind.na)),pch="|",col='red')
  message(names.loc[j],' | ',min(kt.full$ts[ind]),' - ',max(kt.full$ts[ind]))
}
abline(v=tw.bd,col='blue')
# ^ exclude 2E9C, DF07, E1F8, E47A, and E486

loc.excl <- c('2E9C','DF07','E1F8','E47A','E486')
col.excl <- ifelse(names.loc%in%loc.excl,'blue','red')

plot(map.delhi,removeMargin=F)
points(df.latlon.sp,pch=19,col=col.excl) # our sensors
text(names.loc,x=df.latlon.sp@coords[,1],y=df.latlon.sp@coords[,2],
     col=col.excl,cex=0.5,pos=c(4,2,3,2,4,4,2,4,4,1,1,2,1,4,4,4,1,4,4,2,2,4))
axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
title(main="Kaiterra sensors Apr-Jul 2018, 22 locations",
      xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')
abline(v=c(8588000,8601000)) # new area
abline(h=c(3312000,3322000)) # new area
# ^ exclude locations too far too, too many extra loc in Delaunay triangulation,
#   retained area is larger than Sameeksha's

unique(kt.full$longitude[kt.full$loc_id_short=='20CA']) # new area left
unique(kt.full$latitude[kt.full$loc_id_short=='20CA'])  # new area top
unique(kt.full$longitude[kt.full$loc_id_short=='113E']) # new area right
unique(kt.full$latitude[kt.full$loc_id_short=='91B8']) # new area bottom

corners.delhi <- list('topleft'=c(28.58, 77.16), # lat/lon
                      'botright'=c(28.50, 77.25)) # lat/lon
map.delhi <- openmap(upperLeft=corners.delhi[[1]],lowerRight=corners.delhi[[2]],
                     zoom=NULL,type='stamen-toner') # type='osm'

df.latlon <- SpatialPoints(kt.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
df.latlon.sp <- spTransform(df.latlon,osm())

plot(map.delhi,removeMargin=F)
points(df.latlon.sp,pch=19,col=col.excl) # our sensors
text(names.loc,x=df.latlon.sp@coords[,1],y=df.latlon.sp@coords[,2],
     col=col.excl,cex=0.7,pos=c(4,2,3,2,4,4,2,4,4,1,1,2,1,4,4,4,1,4,4,2,2,4))
axis(side=1,at=seq(map.delhi$bbox$p1[1],map.delhi$bbox$p2[1],length=5),line=1)
axis(side=2,at=seq(map.delhi$bbox$p1[2],map.delhi$bbox$p2[2],length=5),line=1)
title(main="Kaiterra sensors Apr-Jul 2018, new area, 11 locations left",
      xlab='Pseudo-Mercator easting (m)',ylab='Pseudo-Mercator northing (m)')


### create df of new area, 11 locations, 1729 time points = 18 days
kt <- subset(kt.full,subset=(kt.full$ts >= tw.bd[1] &
                               kt.full$ts <= tw.bd[2] &
                               kt.full$latitude <= corners.delhi[[1]][1] &
                               kt.full$latitude >= corners.delhi[[2]][1] &
                               kt.full$longitude >= corners.delhi[[1]][2] &
                               kt.full$longitude <= corners.delhi[[2]][2]))
str(kt)

names.loc <- as.character(unique(kt$loc_id_short))
names.sens <- unique(kt$s_id_short)
nb.loc <- length(names.loc) # 11 locations left
nb.sens <- length(names.sens) # 11 sensors

for (j in 1:nb.loc){
  message('location ',names.loc[j])
  print(unique(kt$s_id_short[kt$loc_id_short==names.loc[j]]))
}
# ^ 11 sensor-location pairs, 1-1 matching with loc now


### visualize available data points per location
range(kt$ts)
tw.bd # identical by construction

# pdf('Kaiterra_15min_SensorsUptime.pdf',width=14,height=7)
plot(x=range(kt$ts),y=c(1,nb.loc),type='n',
     xlab='Total time period',ylab='Location',yaxt='n',
     main='Kaiterra Apr-Jul 2018, up and down time by location, new area')
axis(side=2,at=1:nb.loc,labels=names.loc,las=1)
for (j in 1:nb.loc){
  ind <- kt$loc_id_short==names.loc[j]
  ind.na <- c(T,diff(kt$ts[ind])!=15) # contiguous = time lag 15
  arrows(x0=min(kt$ts[ind]),x1=max(kt$ts[ind]),y0=j,y1=j,
         col='grey',angle=90,code=3,length=0.05,lty=1)
  points(x=kt$ts[ind],y=rep(j,sum(ind)),pch=19,cex=0.1)
  points(x=kt$ts[ind][ind.na],y=rep(j,sum(ind.na)),pch="|",col='red')
  message(names.loc[j],' | ',min(kt$ts[ind]),' - ',max(kt$ts[ind]))
}
# legend('topleft',c('downtime','uptime','down -> up'),
#        pch=c(NA,19,NA),pt.cex=c(1,0.4,1),lty=c(1,NA,1),col=c('grey',1,'red'))
# ^ candidate tw, both with 13 sensors ~continuous: mid-May and mid-July
# dev.off()







### create df of locations with lat/lon and UTM coord
kt.loc <- data.frame('loc'=names.loc,stringsAsFactors=F)

table(round(kt$longitude,4)) # looks ok
table(round(kt$latitude,4)) # looks ok

for (j in 1:nb.loc){
  kt.loc$lon[j] <- kt$longitude[kt$loc_id_short==names.loc[j]][1]
  kt.loc$lat[j] <- kt$latitude[kt$loc_id_short==names.loc[j]][1]
}

df.latlon <- SpatialPoints(kt.loc[,c('lon','lat')],proj4string=CRS("+init=epsg:4326"))
proj.string <- "+proj=utm +zone=43 +ellps=WGS84 +north +units=km"
# ^ Delhi = UTM zone 43R
coord.utm <- spTransform(df.latlon, CRS(proj.string)) # re-project
kt.loc$utmx <- coord.utm@coords[,1]
kt.loc$utmy <- coord.utm@coords[,2]

str(kt.loc)


### create df of new area 11 sensors with NAs in correct places, one row per loc
tw <- seq(from=tw.bd[1],to=tw.bd[2],by='15 min') # common time stamp
length(tw) # 1729 values incl bounds
str(tw) # correct format

library(padr) # vignette('padr') # pad() to pad df with NAs based on time stamp

kt.sens <- data.frame('ts'=tw)
for (j in 1:nb.sens){
  ind <- kt$s_id_short==names.sens[j] &
    kt$ts>=tw.bd[1] & kt$ts<=tw.bd[2]
  df.tmp <- data.frame('ts'=kt$ts[ind],'pm25'=kt$pm25[ind])
  padded.df.tmp <- pad(df.tmp,interval='15 min',start=tw.bd[1],end=tw.bd[2])
  # table(diff(df.tmp$ts))
  # table(diff(padded.df.tmp$ts))
  kt.sens[[paste0('pm25_',names.sens[j])]] <- padded.df.tmp$pm25
}
str(kt.sens)
head(kt.sens)
tail(kt.sens) # looks good



### save useful objects in envir
save(list=c('kt.loc','kt.sens','tw.bd','names.sens',
            'corners.delhi','map.delhi'),
     file='kt_GoodTW_Padded.RData')



