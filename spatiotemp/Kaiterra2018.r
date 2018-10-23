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



