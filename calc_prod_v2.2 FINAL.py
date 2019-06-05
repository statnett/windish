print 'Initializing...'
import numpy as np
import logging
import logging.config
import time
import os
import glob
#import scipy.io
from scipy.io import loadmat
import datetime
import math
import matplotlib.dates
import yaml
import sys
import pandas as pd
#import argparse

version = '1.2'
'''Change log
version 1.1: Delivered to customer
version 1.2: Rectified error: When wind point is filtered out the total weight is adjusted.
version 2.0: Added iceloss calculations
version 2.1: Rectified error: In version 2.0, the iceloss variable was not set. Iceloss defined in line 115.
'''


# -- ## -- ### -- ## -- # Tillagte valg  # -- ## -- ### -- ## -- #

# Her kan vi oppgi hvordan modellen skal kjoeres
area = ['europe']     # [europe, nordic, nordic_icing, norway, solar, parker_norge, parker_norge_icing, parker_norge_i_drift, sweeden] kan brukes til flere eller ett område
solar_tracker = True # velger om vi kjører med tracker eller ikke [True, False]
startyear = 2016      # 1950-2016 (fra og med)
endyear = 2017        # 1950-2017 (frem til, uten) kanskje legge til +1

# Utskriftsalternativer
orginal_csv = True        # Utskrift ubehandlet[True, False]
BID = False           # Velger om BID-filer skal lages [True, False]
samlast = False       # Utsrift tilpasset samlast [True, False]

# Dette kan endres for spesielt interesserte
indir = 'data/'             # velger mappen hvor vaerdataen ligger (matlab-filer) [data, data_smooth]
outdir = 'exports/'         # velger hvor dataen skal lagres
config_dir = 'config/'      # velger hvor konfigurasjonsfilene ligger





# --- ## --- Kjeller Vindteknikk sine funksjoner --- ## --- #

# Timing
start_time = time.time()

# Setup logging
logging.config.fileConfig('./logging.conf',disable_existing_loggers=False)
logger = logging.getLogger(sys.argv[0])
logger.info('Running version %s' % version)

def calc_point(reg_file,startyear,endyear,indir,outdir,config_dir):
    '''
    Write in description of what this code does.
    '''
    # Read file with points in the region
    namelist, region = read_region(config_dir+reg_file)
    namelist.sort() # Sort namelist so that region with same name is added together
    region_list = np.unique([x['name'] for x in region.values()]) # List of unique points in region
    logger.debug(region_list)
    logger.debug(namelist)
    logger.info('Calculating production for region in config file: %s' % reg_file)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Setting directory for input files
    if 'solar' in reg_file:
        indir = indir + 'solar/'
        energy_type = 'solar'
        config_dir = ''.join([config_dir,'solar_panels/'])

    elif 'park' in reg_file:
        indir = indir + 'norge_park/'
        energy_type = 'wind'
        config_dir = ''.join([config_dir,'turbines/'])
        do_wind_f = False
    else:
        indir = indir + 'wind/'
        energy_type = 'wind'
        config_dir = ''.join([config_dir,'turbines/'])
        do_wind_f = True # Filtering of points

    logger.info('Energy type: %s.' % energy_type)
    logger.info('Reading files from directory: %s.' % indir )

    tmp_point_name = ''
    t1 = datetime.date(startyear, 01, 01)
    t2 = datetime.date(endyear, 01, 01)

    # Read all points in namelist and calculate the production based
    for key in namelist:
        logger.debug("Start calculating production for %s." % key)
        if 'park' in reg_file:
            name = ''.join(['*',key,'*'])
        else:
            name = ''.join(['*_',key,'_*'])
        logger.debug('name is %s' % name)
        file = ''.join(glob.iglob(os.path.join(indir, name)))

        
        if file:
            logger.info('Reading file %s ...' % file)
            #data = scipy.io.loadmat(file)
            data = loadmat(file)
            point = region[key]
            point_name =point['name']
            logger.debug('Writing point: %s ' % point_name)
            total_val = 0
                         
            
            if energy_type == 'solar':
                weight = point['weight']
                panel = point['panel']
                panelfile = ''.join([config_dir,panel,'.yml'])
                logger.info('Calculating production for  point %s with weight %.2f. The point belongs to %s.'% (key, weight, point_name))
                jdate, swdown, prod, lat, lon = load_picky_solar(data,t1,t2,panelfile)
                if tmp_point_name == '':
                    logger.info('First point of solar written')
                    outdata = np.zeros(len(jdate), 'a20,' + ','.join(len(region_list)*['f4']))
                    outdata.dtype.names = ['date'] + list(region_list)
                    outdata[:]=np.nan
                if point_name != tmp_point_name:
                    outdata[point_name] = np.around(prod*weight,decimals=4)
                    tmp_point_name = point_name    
                else: # Point already exists
                    outdata[point_name]= outdata[point_name] + np.around(prod*weight,decimals=4)
                
            elif energy_type == 'wind':
    
                turbine_list = point['turbine']
                turb_weight_list = point['turb_weight']
                weight = float(point['weight'])
                
                iceloss = 0.0
                if 'icing' in reg_file:
                    iceloss = float(point['iceloss'])
                    
                logger.info('Calculating production for point %s with weight %.2f. The point belongs to %s.'% (key, weight, point_name))
                logger.info('Turbine configurations: %s, Weight: %s.' % (turbine_list, turb_weight_list))

                for i in range(len(turbine_list)):
                    turbine = turbine_list[i]
                    turb_weight = float(turb_weight_list[i])
                    turbfile = ''.join([config_dir,turbine,'.yml'])
                    jdate, ws, prod, lat, lon = load_picky(file,data,turbfile,do_wind_f,t1,t2,config_dir) # Calculate production with the current turbfile

                    # Calculate iceloss if it is over 0%
                    if iceloss > 0.0:
                        iceFile = ''.join(glob.iglob(os.path.join('icing/', name)))
                        iceData = loadmat(iceFile)
                        odat = iceData['odat']
                        val = odat[0,0]
                        iceDate = val["jdate"].squeeze()
                        powercorrection = val["Powercorrection"].squeeze()
                        Mstd = val["Mstd"].squeeze()
                        powerloss = 1 - powercorrection

                        tstart = matplotlib.dates.date2num(t1) + 366
                        tend = matplotlib.dates.date2num(t2) + 366
                        ind = (iceDate >= tstart)*(iceDate <= tend)
                                    
                        if np.mean(powerloss) < iceloss:
                            I = np.where((Mstd >10) & (powerloss < 0.1))
                            powerloss[I]=0.1
                            
                        logger.debug('Mean of powerloss: %.4f, configured ice loss: %.4f'    % (np.mean(powerloss), iceloss))
                        scale = iceloss/np.mean(powerloss)
                        powerloss = powerloss*scale
                        powerloss[powerloss>1]=1
                        logger.info('Mean of scaled ice loss: %.4f'    % np.mean(powerloss))
                        logger.info('Mean of production without ice loss is %.2f' % np.mean(prod))
                        prod = prod*(1-powerloss[ind])
                        logger.info('Mean of production with ice loss is %.2f' % np.mean(prod))

                    if tmp_point_name == '':
                        # Initializing outdata
                        outdata = np.zeros(len(jdate), 'a20,' + ','.join(len(region_list)*['f4']))
                        outdata.dtype.names = ['date'] + list(region_list)
                        outdata[:]=np.nan
                    if point_name != tmp_point_name:
                        if sum(prod)==0.0:
                           logger.info('Production is not calculated for this point.')
                        else:
                           logger.info('Writing first point to %s.' % (point_name))
                           outdata[point_name] = np.around(prod*weight*turb_weight,decimals=4)
                           
                        tmp_point_name = point_name
                        logger.info('tmp_point_name = %s ' % tmp_point_name)        
                    else:
                        logger.info('Adding an additional point, %s, to %s, with weight %.2f.' % (key, point_name, float(weight)))
                        if sum(prod)==0.0:
                            logger.debug('Production is not calculated for this point.')
                        else:
                            outdata[point_name] = outdata[point_name] + np.around(prod*weight*turb_weight,decimals=4) # Adding prod*weight to the point
                                                           
                    logger.debug('Number of columns in outdata is %d ' % (outdata.ndim))
            else:
                logger.debug('There is no file to be read for this point.')

    # Convert jdate to string date format
    if tmp_point_name == '':
        logger.info('No file has been read, please check the config file.')
        exit()
        
    datevec = matplotlib.dates.num2date(jdate - 366)
    date_str = [str(datevec[i]) for i in range(len(datevec))]
    date_str = [word[:-12] for word in date_str]
    outdata['date']=date_str
    outfile_region = np.char.rstrip(reg_file,'.yml') 
    outfile = ''.join([outdir, str(outfile_region), '_prod_', str(startyear),'_', str(endyear), '.csv']) # navn på fil
    # Writing data to file
    if orginal_csv == True:
        with open(outfile, 'wb') as f:
            header_line = 'date,'+','.join(region_list) + '\n'
            f.write(bytes(header_line))
            np.savetxt(f, outdata,fmt='%s',newline=os.linesep,delimiter=',')
        f.close

    # print samlast and BID
    if (BID == True or samlast == True):
        df = pd.DataFrame(outdata)
        df['date'] =  pd.to_datetime(df['date'], format="%Y-%m-%d") # from string to date-format
        df.set_index('date', inplace=True) # date as index
        # new index starting 1.1 the first year, endig last full year the 31.12, every hour 
        ix = pd.DatetimeIndex(start=datetime.datetime(df.index.year[0], 1, 1), end=datetime.datetime(df.index.year[-2], 12, 31, 23), freq='H') 
        df = df.reindex(ix, method = 'nearest') # if values are missing, we use the nearest value 
        df = df[~((df.index.month == 2) & (df.index.day == 29))] # remove leap days
        area_names = df.columns # make a list of column names and loop over them
        for i in range(len(area_names)):
            df_column = pd.DataFrame(df[area_names[i]]) # make new df for each column
            if BID == True: # bid format
                df_bid = df_column
                df_bid['year'] = df_bid.index.year #new column with year
                df_bid['hour'] = list(range(1,8761,1))*len(set(df_bid['year'])) #hour is hour nr. that year: 1-->8760
                df_bid = df_bid.pivot(columns='year', values=str(area_names[i]), index='hour') # years as columns  and hour of the year as index
                outfile_bid = ''.join([outdir, 'BID_', str(area_names[i]), '_', str(outfile_region), '_prod_', str(startyear),'_', str(endyear), '.csv']) #name file
                df_bid.to_csv(str(outfile_bid), sep=';', header=False, index=False) #save file
            if samlast == True: #samlast format
                df_sam = df_column
                df_sam = df_sam[~((df_sam.index.month == 12) & (df_sam.index.day == 31))] # delete 31.12
                df_sam['hour'] = list(range(1,25,1))*(len(df_sam)/24) # hour that day 1-->24
                df_sam['day'] = list(np.repeat(list(range(1,len(set(df_sam.index.date))+1,1)),24)) # day nr in whole df
                # setter header med informasjon
                info = [("Number of year", 'Start year', 'Number of weeks','Start week','End week', 'Start day','Type data (Vind=1, Tilsig=2)','Type resolution (Week=1, Day=2, Hour=3)'),(len(set(df_sam.index.year)),df_sam.index.year[0],52,1,52,0,1,3),('Series with hour resolution','','','','','','','')]
                df_sam = df_sam.pivot(columns='hour', values=str(area_names[i]), index='day') # day nr in df as index and hour as columns
                
                df_header = pd.DataFrame(data=info,columns=list(df_sam.columns[:8])) # make a df that we append on top of df_sam
                df_samlast = pd.concat([df_header, df_sam], ignore_index=True).fillna('')
                outfile_samlast = ''.join([outdir, 'samlast_', str(area_names[i]), '_', str(outfile_region), '_prod_', str(startyear),'_', str(endyear), '.csv'])
                df_samlast.to_csv(str(outfile_samlast), sep=';', header=False, index=False) # save as csv without header or index


def load_picky_solar(mat,t1,t2,configfile):
    from sunposition import sunpos

    jdate = mat["jdate"].squeeze()
    lon = mat["wrflon"]
    lat = mat["wrflat"]
    hgt = mat['hgt']
    swdown = mat['data']
    lon = np.mean(lon)
    lat = np.mean(lat)
    hgt = np.mean(hgt)

    # Only use the data from t1 to t2
    tstart = matplotlib.dates.date2num(t1) + 366
    tend = matplotlib.dates.date2num(t2) + 366
    ind = (jdate >= tstart)*(jdate <= tend)
    jdate = jdate[ind]
    swdown = swdown[ind]
    swdown = np.mean(swdown,axis=1) #
    
    datevec = matplotlib.dates.num2date(jdate-366)
    day_of_year = np.zeros([len(jdate)])
    for i in range(len(jdate)): day_of_year[i] = datevec[i].timetuple().tm_yday
    
    # Find azimuth, zenith, declination, angle
    logger.info('Compute the coordinates of the sun as viewed at the given time and location. This may take some time...')
    coords = np.zeros((len(jdate),5))
    coords = sunpos(datevec,lat,lon,hgt)
    az   = coords[:,0]     #coords[...,0] = observed azimuth angle, measured eastward from north
    zen   = coords[:,1]     #coords[...,1] = observed zenith angle, measured down from vertical matlab.zen
    delta = coords[:,3]     #coords[...,3] = topocentric declination (delta?) Same as matlab.delta
    omega = coords[:,4]+360 #coords[...,4] = topocentric hour angle (omega?) Same as matlab.omega -360
    logger.debug('Computiation of coordinates of the sun is finished.')
    
    def cost(angle): return np.cos(np.radians(angle))
    def sint(angle): return np.sin(np.radians(angle))
    def tant(angle): return np.tan(np.radians(angle))

    #bbeam
    b00 = 1367
    grad = swdown
    eps0 = 1 + 0.033*cost((360*day_of_year)/365)

    # Extraterrestial radiation
    b = b00*eps0*cost(zen)
    sun_alt = 90-zen
    b[sun_alt<=0]=0

    # Clearness index
    kt = np.zeros(b.shape)
    kt[b>0] = grad[b>0]/b[b>0]
    kt[kt>1] = 0

    # Diffuse indices
    fd = 0.868 + 1.335*(kt) - 5.782*(kt**2)  + 3.721*(kt**3)
    fd[kt<=0.13]=0.952
    fd[kt>0.8] = 0.141
    drad = grad*fd

    # Beam radiation (Global radiation - diffuse radiation)                                        
    brad = grad - drad # beam radiation
    bbeam = brad/cost(zen)
    bbeam[bbeam<0]=0
    bbeam[bbeam>2000]=0

    efficiency = 0.1
    p_inst = 1
    area = p_inst*0.00875
    derate = 0.77
    
    #set panel orientation
    panel_matrix_az, panel_matrix_tlt, panel_matrix_weight,azimuth_median,tilt_median = read_solar_panel(configfile)

    if not azimuth_median:
        azimuth_median = 0
    if not tilt_median:
        tilt_median = lat

    logger.info('Panel is directed with azimuth at median %d and tilt at median %d' % (azimuth_median, tilt_median))
        
    panel_matrix_az = azimuth_median + panel_matrix_az
    panel_matrix_tlt = tilt_median + panel_matrix_tlt
        
    prod = np.zeros(len(bbeam))
    # solar tracker is considered
    # TIPS: https://www.e-education.psu.edu/eme810/node/576 
    # https://www.e-education.psu.edu/eme810/node/485
    if solar_tracker == False: # use the standard model if False
        for i in range(3):
            for j in range(3):
                panel_az = panel_matrix_az[i,j]
                panel_slp = panel_matrix_tlt[i,j]
                weight = panel_matrix_weight[i,j]
                 
                #angel between beam and panel
                costh_s = sint(delta)*sint(lat)*cost(panel_slp) - np.sign(lat)*sint(delta)*cost(lat)* sint(panel_slp)*cost(panel_az) + cost(delta)*cost(omega)*cost(lat)*cost(panel_slp) + np.sign(lat)*cost(delta)*cost(omega)*sint(lat)*sint(panel_slp)* cost(panel_az) + cost(delta)*sint(omega)*sint(panel_az)*sint(panel_slp)
                bbeam_panel = bbeam*np.maximum(0,costh_s)
    
                drad_panel = drad*(1+cost(panel_slp))/2
                rad_panel = bbeam_panel + drad_panel
                
                prod = prod + weight*rad_panel*efficiency*area*derate
                
    elif solar_tracker == True:
        costh_s_tracker = 1 # strålen treffer 90 grader på panelet hele tiden, ergo er cos av denne vinkelen lik 1
        bbeam_panel = bbeam*costh_s_tracker  # bbeam sørger for at dette blir null når solen er nede
        drad_panel = drad * (1+cost(zen)) / 2 # setter zentih angle som tilt
        rad_panel = bbeam_panel + drad_panel
        prod = rad_panel*efficiency*area*derate
        
    return jdate, swdown, prod, lat, lon
        
        
def load_picky(fname,mat,turbfile, do_wind_f,t1,t2,config_dir):
    ''' This function reads the data from .mat file
    1. Read variables, jdate, FF, lmask, hgt, lon and lat for a point.
    2. Interpolate FF to the given height for both reference data and 'site' data.
    3. Find syntesized long term data set
    4. Filter out points in the data set that has low mean wind speed, high mean wind speed or are not onshore/offshore (depending on offshore/onshore point)
    5. Calculate production (prod), production with wake loss (prod_wake), production prod_loss, p_mat_use =
    
    MAREN FREDBO 2016.12.14
    INPUT:
    file:       .mat-file to read
    turb_file:  turbfile to read
    OUTPUT:

    '''
    height, vel_cl, pmat, loss, w_loss, scale_w_loss = read_turbine(turbfile)
    levels = np.array([30.,80.,100., 150., 180.])
    lev1, lev2 = get_levels(levels,height) # Find levels to interpolate between
    #logger.info('Reading file %s ...' % fname)
    jdate = mat["jdate"].squeeze()
    FF1 = mat['FF'+(str(int(lev1)))]
    FF1_r = mat['FF'+(str(int(lev1))) + '_r'] # Long (reference) term data
    rdate = mat["jdate_r"].squeeze()
    hgt = mat['hgt'].squeeze()
    lmask = mat['lmask'].squeeze()
    lon = mat["wrflon"]
    lat = mat["wrflat"]
    logger.debug('Number of grid points to be read is %d ' % hgt.size)

    # Interpolate FF to the given height
    if lev1 == lev2:
        # No need for interpolation, FF1 and FF_r can be used directly.
        logger.debug('No interpolation needed. FF at level %d is used directly.' % lev1)
        site = FF1
        ref = FF1_r
    else:
        # Interpolate to find wind speed at the given height
        FF2 = mat['FF'+(str(int(lev2)))]
        FF2_r = mat['FF'+(str(int(lev2))) + '_r']
        sz = FF1.shape
        sz2 = FF1_r.shape
        site = np.empty([sz[0],sz[1]])
        ref = np.empty([sz2[0],sz2[1]])

        #import ipdb;ipdb.set_trace() # Denne vil lagre alle variabler naar man kommer til dette steget. MAA FJERNES.
        
        for i in range(hgt.size):
            logger.debug('Interpolating for point number %d' % i)
            site[:,i] = interp_FF(np.squeeze(FF1[:,i]),np.squeeze(FF2[:,i]),np.array([lev1, lev2]),height)    
            ref[:,i] = interp_FF(np.squeeze(FF1_r[:,i]),np.squeeze(FF2_r[:,i]),np.array([lev1, lev2]),height)
            
        
    # Remove nans from dataset
    site = site[~np.isnan(site).any(axis=1)]
    jdate = jdate[~np.isnan(site).any(axis=1)]
    logger.debug('Filtered out ' + str(sum(np.isnan(site).any(axis=1))) + ' NaN elements from site data.')
    ind = np.in1d(rdate,jdate) # Find period where jdate and rdate intersect
    ru = ref[ind,:] # reference short term

    # Find syntesized series if rdate > jdate
    sul = np.zeros((len(rdate),hgt.size))
    nvel = 30 # Number of bins to be used in syntetization
    if len(rdate)>len(jdate):
        for i in range(hgt.size):
            sul[:,i] = synt_one_sec(site[:,i],ru[:,i],ref[:,i],rdate,nvel)
            logger.debug('Syntesized point number ' + str(i) + ' from date ' +  str(matplotlib.dates.num2date(rdate[0] - 366)) + ' to date ' +  str(matplotlib.dates.num2date(rdate[-1] - 366)))
    else:
        logger.debug("Using 18 km directly. The size of jdate is %d and the size of rdate is %d" % (len(jdate),len(rdate)))
        sul = site;

    # Filter points with low or high wind speeds. Reference period for filtering wind is fixed from dstart to dend (1.1.1990-1.1.2013). Define start and end of period that are fixed (for filtering routine)
    lt_t1 = datetime.date(1990, 01, 01)
    lt_t2 = datetime.date(2013, 01, 01)
    ltstart = matplotlib.dates.date2num(lt_t1) + 366
    ltend = matplotlib.dates.date2num(lt_t2) + 366
    logger.debug('Filter points with mean wind speeds out of range. Fixed period used for filtering routine is [%s, %s]' % (str(t1),str(t2)))
    lt_ind = np.where(np.logical_and(rdate>=ltstart,rdate<=ltend))

    mu = np.squeeze(np.mean(sul[lt_ind,:],axis=1))
    if do_wind_f == False:
        muu = np.ones(mu.squeeze().size)*8
    else:    
        muu = mu
    REM = filter_wind(fname,muu,lmask) # REM equals zero means that the point is not filtered out


    # Production calculation. Find production estimates for all points    
    prod = np.zeros([len(rdate),hgt.size])
    prod_wake = np.zeros([len(rdate),hgt.size])
    prod_loss = np.zeros([len(rdate),hgt.size])
   
    if mu.size == 1:
       for i in range(hgt.size): prod[:,i], prod_wake[:,i], prod_loss[:,i] = tseries_prod(rdate,sul[:,i],mu,vel_cl,pmat,loss,w_loss,scale_w_loss,config_dir)
    else:
        for i in range(hgt.size): prod[:,i], prod_wake[:,i], prod_loss[:,i] = tseries_prod(rdate,sul[:,i],mu[i],vel_cl,pmat,loss,w_loss,scale_w_loss,config_dir)
    
    tstart = matplotlib.dates.date2num(t1) + 366
    tend = matplotlib.dates.date2num(t2) + 366
    ind = (rdate >= tstart)*(rdate <= tend)

    if muu.size > 1: # If picky
        lat_all = np.mean(lat)
        lon_all = np.mean(lon)

        if sum(REM>0)== muu.size: 
            logger.info('WARNING: This point is filtered out (too low/high mean wind speed or offshore/onshore point)')
            ws_all = np.zeros([len(rdate[ind]),1])
            prod_all = np.zeros([len(rdate[ind]),1])
        else:
            ws_filt = sul[:,REM<1] # Filter out points from filtering routine 
            prod_filt = prod_loss[:,REM<1] # Filter out points from filtering routine
            ws_all = np.mean(ws_filt[ind,:],axis=1)
            prod_all = np.mean(prod_filt[ind,:],axis=1)
    else:
        ws_all = sul[ind] # Use only
        prod_all = prod_loss[ind]
        lat_all = lat
        lon_all = lon
        logger.debug('REM is %d ' % REM)
       
    return rdate[ind], ws_all.squeeze(), prod_all.squeeze(), lat_all.squeeze(), lon_all.squeeze()

def tseries_prod(jdate,data,mean_speed,vel_cl,pmat,loss,w_loss,scale_w_loss,config_dir):
    ''' This function calculates production from a timeseries of wind ("data").
    INPUT:
    jdate:      Date vector, 1-dim np.array
    data:       Vector with wind speed. 1-dim np.array
    mean_speed: Mean speed for each point, 1 dim np.array
    vel_cl:
    pmat:
    loss:
    w_loss:
    scale_w_loss:
    config_dir:

    OUTPUT:
    prod:       Production without any losses
    prod_wake:  Estimated loss due to wake
    prod_loss:  Estimated production including loss
    '''
    from scipy.interpolate import interp1d
    if isinstance(pmat,str): # If pmat is string then get the .csv-file
        pmat = np.genfromtxt(config_dir+pmat,delimiter=',')
        vel_choose = pmat[1:,0]
        pmat = np.transpose(pmat[1:,1:]) # Remove first column and first row of the array and transpose

    prod_wake = np.zeros(len(jdate))
    prod = np.zeros(len(jdate))
    p_mat_use = np.empty(len(vel_cl))
    p_mat_use[:] = np.NAN 

    # Interpolation of production. This differs from the Matlab routine wich extrapolates
    f_loss = interp1d(vel_cl,w_loss,bounds_error=False,fill_value=0)
    if isinstance(pmat,np.ndarray): # pmat is array
        min_val = abs(vel_choose - mean_speed).argmin()
        p_mat_use = pmat[:,min_val] # pmat to be used
    else:
        p_mat_use = pmat
        
    # Interpolation of production. This differs from the Matlab routine as it does not extrapolate for velocities larger and less than vel_cl. bounds_error=False assigns out fo  bounds values assignes fill_value. Using interp1d in combination with extrap1d would give the same results as in Matlab.
    f_prod = interp1d(vel_cl,p_mat_use,bounds_error=False,fill_value=0)
    prod = f_prod(data)
    prod_wake = f_loss(data) # Interpolation of f_loss

    prod_wake = prod_wake*scale_w_loss   # Tuning of wake loss  
    prod_loss = prod*(1-loss)            # Substract losses
    prod_loss = prod_loss*(1-prod_wake)  # Substract wake loss
    return prod, prod_wake, prod_loss


def get_levels(levels, height):
    ''' Find levels to interpolate between'''
    if ~np.any(levels>=height) or ~np.any(levels<=height):
        logger.info('Height is out of range. Please select a height between %.f and %.f.' % (levels[0],levels[-1]))
        sys.exit()    
    else:
        lev1 = levels[np.max(np.where(levels<=height))]
        lev2 = levels[np.min(np.where(levels>=height))]
        logger.debug('The wind series are interpolated to height = %.f. Level 1 is %d and level 2 is %d .' % (float(height),lev1,lev2))    

    return lev1, lev2

def stepf(x):
    stepf = []
    for num in x:
        if num <= 0:
            stepf.append(0)
        elif num >= 1:
            stepf.append(1)
        else:
            stepf.append(1./(1. + math.exp((1./(num-1))+1./num)))
    stepf = np.array(stepf)
    return stepf
    
def filter_wind(fname, muu, lmask):
    ''' This routine filter out point that have lower wind speed than 6 m/s or higher than 10 m/s (12 ms/s if the point is offshore.
    REM = 5x(Point is offshore/onshore) + 1x(point has mean wind speed below minvel) + 1x(point has mean wind speed above maxvel)
    REM equal to zero means that the point shall not be filtered out.
    '''
    logger.debug('Filter routine for wind will discard points with low/high wind speeds and points thata are offshore/onshore.')
    if 'Offshore' in fname:
        isoffshore = 1
    else:
        isoffshore = 0;
    minvel = 6
    maxvel = 10 + 2*isoffshore
    REM = np.zeros(muu.size)
 
    if muu.size > 1:
        REM[lmask==isoffshore] = 5
        REM[muu<minvel] = REM[muu<minvel] + 1
        REM[muu>maxvel] = REM[muu>maxvel] + 1
        logger.debug('Total %d points flagged. (%d) (%d < %d m/s), (%d > %d m/s) and (%d with LMASK: %d) ' % (muu.size, sum(REM>0), sum(muu<minvel), minvel, sum(muu>maxvel), maxvel, sum(lmask==isoffshore), isoffshore))

    else:
        if muu < minvel: REM = 1
        elif muu > maxvel: REM = 1
        elif lmask == isoffshore: REM = 5
        else: REM = 0
            
    return REM

def synt_one_sec(su,ru,rul,rdatel,nvel):
    ''' Function that syntesize a short time series for a long time period.
    Parameters su,sd,ru,rd,rul,rdl,rdatel,number_sec_synt are all required.
    su - wind speed vector at site
    ru -  wind speed vector at ref (sim site)
    rul - long-term wind speed vector at ref
    rdatel - long-term numeric date index of the reference time series
    number_sec_synt - number of sectors used by the methodology

    MAREN FREDBO 2016.12.14
    '''
    from scipy.interpolate import interp1d
    ref_sort = sorted(ru)
    site_sort = sorted(su)
    logger.debug('Length of ref is ' +  str(len(ref_sort)) + ', which should be the same as site: ' + str(len(site_sort)))
    end = len(site_sort)
    bin_size = end/nvel
    vel_ref = [0]
    vel_site = [0]

    # Find mean in each bin    
    for j in range(nvel):
        i1 = int(round(bin_size*j))
        i2 = int(round(bin_size*(j+1)-1))
        mean_bin = np.mean(ref_sort[i1:i2])
        vel_ref.append(np.mean(ref_sort[i1:i2]))
        vel_site.append(np.mean(site_sort[i1:i2]))
  
    vel_site.append(50*vel_site[-1]/vel_ref[-1]) # Add high wind speed at end
    vel_ref.append(50)
    
    # Interpolate to find syntesized series for the long-term period
    sul = interp1d(vel_ref,vel_site)(rul)
                    
    return sul


def interp_FF(FF_lev1,FF_lev2,zin,zout):
    '''Function that return the interpolated wind speed at height zout
    INPUT:
    FF_lev1: FF at level 1, 1-dim np.array
    FF_lev2: FF at level 2, 1-dim np.array
    zin: height at level1 1 and level 2,  1-dim np.array
    zout: height to find the interpolated values of FF
    OUTPUT:
    FFout: Interpolated FF at height zout, 1-dim np.array 
    '''
    FF = np.empty([2,len(FF_lev1)])
    FF[0,:] = FF_lev1
    FF[1,:] = FF_lev2
    # Remove indecies where FF_lev1 or FF_lev2 is NaN
    FF_new=FF[:,~np.isnan(FF).any(axis=0)]
    FF_new=FF_new[:,~(FF_new==0).any(0)]
    alpha = np.log(FF_new[1,:]/FF_new[0,:])/np.log(float(zin[1])/float(zin[0]))    
    alpha = np.mean(alpha) # Find the mean of alpha
    level = np.argmin(abs(zout-zin)) # Find nearest level
    FFout = FF[level,:]*(zout/zin[level])**alpha
    return FFout

def read_solar_panel(file):

    try:
        with open(file) as f:
            panel = yaml.safe_load(f)
            f.close()
        
        azimuth = np.array(panel['azimuth'])
        tilt = np.array(panel['tilt'])
        weight_matrix = np.array(panel['weight'])
        azimuth_median = panel['azimuth_median']
        tilt_median = panel['tilt_median']

        azimuth_matrix = np.zeros((len(azimuth),len(azimuth)))
        tilt_matrix = np.zeros((len(azimuth),len(azimuth)))

        for i in range(len(azimuth)):
            azimuth_matrix[:,i] = azimuth
            tilt_matrix[i,:]= tilt
        if not np.sum(weight_matrix)==1:
            logger.info('Warning: The sum of the weights in the solar panel configuration file do not add up to 1. Please make sure you\'re weights add up to 1.')

    except:
        logging.error('Error reading %s' % file)
        sys.exit(1)
                
    return azimuth_matrix, tilt_matrix, weight_matrix, azimuth_median, tilt_median
                            

def read_turbine(file):
    # Function that reads data fra turbine files
    try:
        with open(file) as f:
            turb = yaml.safe_load(f)
            f.close()
    
        height = float(turb['height'])
        wind_speed = turb['wind_speed']
        pmat = turb['pmat']
        loss = turb['loss']
        w_loss = turb['w_loss']
        scale_w_loss = turb['scale_w_loss']
        if not height: height = np.NAN; logging.debug('No height specified in the turbine configuration file. Height is set to NaN.')
        if not w_loss: w_loss = np.zeros(len(wind_speed)); logging.debug('No wake loss defined in turbine configuration file. Wake loss is set to 0.')
        if not loss: loss = 0; logging.debug('No loss defined in turbine configuration file. Loss is set to 0.')
        if not scale_w_loss: scale_w_loss = 1; logging.debug('No wake loss scale in turbine configuration file. Wake loss scale is set to 1')
    except:
        logging.error('Error reading %s' % file)
        sys.exit(1)
            
    return height, wind_speed, pmat, loss, w_loss, scale_w_loss

def read_region(regfile):
    # Function that reads data from region files
    try:
        with open(regfile) as f:
            region = yaml.safe_load(f)
            f.close()
        namelist = region.keys()
    except:
        logging.error('Error reading %s' % file)
        sys.exit(1)

    return namelist, region
'''
def main(argv=None):
    """Command line call"""
    parser = argparse.ArgumentParser(
        description="""Runs calc_prod.py. The program calculates production for wind and solar series.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', '-c', help='Config file with region and energy type',
                        default='europe.yml')
    parser.add_argument('--startyear', '-s',type=int, help='Format yyyy. Year to start yyyy-01-01. Minimum 1950, maximum 2015',
                        default=2015)
    parser.add_argument('--endyear', '-e',type=int, help='Format yyyy. Year to end yyyy-01-01. Minimum 1951, maximum 2016',
                        default=2016)
    parser.add_argument('--indir', '-i', help='Input directory',
                        default='data/')
    #parser.add_argument('--version', '-v', help='Print version and exit (%s).' % __VERSION__)
    parser.add_argument('--outdir', '-o', help='Output directory',
                        default='exports/')
    parser.add_argument('--configdir', '-cd', help='Config directory',
                        default='config/')

    args = parser.parse_args(argv[1:])


    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    #calc_point()
    calc_point(reg_file=area+'.yml', startyear=startyear, endyear=endyear, indir=indir, outdir=outdir,config_dir=config_dir)

if __name__ == "__main__":

    main(sys.argv)
'''

### Standard kjøring av ett område ###
#calc_point(reg_file=area+'.yml', startyear=startyear, endyear=endyear, indir=indir, outdir=outdir,config_dir=config_dir)

### LOOP over listen areas ###
for i in range(len(area)):
    calc_point(reg_file=area[i]+'.yml', startyear=startyear, endyear=endyear, indir=indir, outdir=outdir,config_dir=config_dir)