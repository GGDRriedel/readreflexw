# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:10:11 2019

@author: Riedel

imports a segy file and returns a Pandas Dataframe that contains a catalogue
"""
def make_catalogue_segy(filename='.\calibrate\B6N_CALI_004_1GHZ_RES512.sgy',outfile='catalogue_trigger.h5',picker='trigger',standardizing=0):
    '''
    Classifies first arrivals of a file by different picker criteria and 
    returns a catalogue with unique radar traces as a pandas Dataframe
    
    Also writes that catalogue in HDF5 format. 
    filename: path to input file 
    outfile: catalogue output
    picker: picker algorithm for first arrivals 
            'trigger' Seismology trigger algorithm, see module obspytriggers
            'zero' Picks value closest to zero on second pair of min/max
    '''
    
    from readreflex.utils import findmax,standardize,obspytriggers,shift
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import tqdm
   

    with segyio.open(filename) as f:
        f.mmap()
        # Get Sample Time:
        meas_sampletime=f.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL]
        meas_samplecount=f.header[0][segyio.TraceField.TRACE_SAMPLE_COUNT ]
        meas_linecount=f.ilines[-1] # Last one must be biggest
        meas_endtime=meas_samplecount*meas_sampletime
        meas_time=np.linspace(0,meas_endtime,meas_samplecount)
        
    
    meas_data = segyio.tools.cube(filename)
    #print(meas_data)
    meas_data = np.squeeze(meas_data)
#    plt.imshow(data.T,aspect='auto')
    if standardizing!=0:
        meas_data=standardize(meas_data)

    if picker=='zero':
        maxpeaks,minpeaks,zeroindexes,sigzeroindexes=findmax(meas_data)
    elif picker=='trigger':
        triggerindexes=obspytriggers(meas_data,meas_linecount)
        sigzeroindexes=triggerindexes
    else:
        print('No valid picker given, using zero')
        maxpeaks,minpeaks,zeroindexes,sigzeroindexes=findmax(meas_data)
        
            
        
    catalogue=[]
    pdcatalogue=pd.DataFrame(index=meas_time)
   # plt.title('Detected Valid Trace-Categories')
    for i in np.unique(sigzeroindexes):
        currentcorrection=np.squeeze(np.array(np.where(np.array(sigzeroindexes)==i)))
        if currentcorrection.size>0:
            #print(currentcorrection.shape)
            #lets care for single traces
            if currentcorrection.size==1:
                catalogue.append(meas_data[currentcorrection,:])
                #Pandas:
                pdcatalogue[str(int(i))]=meas_data[currentcorrection,:]
                #plt.plot(meas_data[currentcorrection,:])
                
            else:                
               #shift them all to have the same  
               meantrace=np.mean(meas_data[currentcorrection,:],axis=0)
               #plt.figure()
               #plt.plot(meas_data[currentcorrection,:].T)
               #plt.title(i)
              # plt.pause(0.1)
               #np.append(catalogue,meantrace,axis=0)
               catalogue.append(meantrace)
               pdcatalogue[str(int(i))]=meantrace
               
  #  plt.figure()
   # plt.imshow(pdcatalogue,aspect='auto')          
    pdcatalogue.to_hdf(outfile,key='pdcatalogue',mode='w')
    #pdcatalogue.plot()
    return pdcatalogue, sigzeroindexes



def make_catalogue_array(data,outfile='catalogue_trigger.h5',picker='trigger',standardizing=0):
    '''
    Classifies first arrivals of a file by different picker criteria and 
    returns a catalogue with unique radar traces as a pandas Dataframe
    
    Also writes that catalogue in HDF5 format. 
    data: readreflex radargram object
    outfile: catalogue output
    picker: picker algorithm for first arrivals 
            'trigger' Seismology trigger algorithm, see module obspytriggers
            'zero' Picks value closest to zero on second pair of min/max
    '''
    
    from readreflex.utils import findmax,standardize,obspytriggers
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    meas_data=data.traces
    meas_sampletime=data.header["timeincrement"]
    meas_samplecount=data.header["samplenumber"]
    meas_linecount=data.header["tracenumber"]
    meas_endtime=meas_samplecount*meas_sampletime
    
    meas_time=data.header["time"]
#    plt.imshow(data.T,aspect='auto')
    if standardizing!=0:
        meas_data=standardize(meas_data)

    if picker=='zero':
        maxpeaks,minpeaks,zeroindexes,sigzeroindexes=findmax(meas_data)
    elif picker=='trigger':
        triggerindexes=obspytriggers(meas_data,meas_linecount)
        sigzeroindexes=triggerindexes
    else:
        print('No valid picker given, using zero')
        maxpeaks,minpeaks,zeroindexes,sigzeroindexes=findmax(meas_data)
        
            
        
    catalogue=[]
    pdcatalogue=pd.DataFrame(index=meas_time)
   # plt.title('Detected Valid Trace-Categories')
    for i in tqdm.tqdm(np.unique(sigzeroindexes),'Gernerating Catalogue'):
        currentcorrection=np.squeeze(np.array(np.where(np.array(sigzeroindexes)==i)))
        if currentcorrection.size>0:
            #print(currentcorrection.shape)
            #lets care for single traces
            if currentcorrection.size==1:
              
                catalogue.append(meas_data[currentcorrection,:])
                #Pandas:
                pdcatalogue[str(int(i))]=meas_data[currentcorrection,:]
                #plt.plot(meas_data[currentcorrection,:])
                
            else:                
               #shift them all to have the same 
                
               meantrace=np.mean(meas_data[currentcorrection,:],axis=0)
               #plt.figure()
               #plt.plot(meas_data[currentcorrection,:].T)
               #plt.title(i)
              # plt.pause(0.1)
               #np.append(catalogue,meantrace,axis=0)
               catalogue.append(meantrace)
               pdcatalogue[str(int(i))]=meantrace
               
  #  plt.figure()
   # plt.imshow(pdcatalogue,aspect='auto')          
    pdcatalogue.to_hdf(outfile,key='pdcatalogue',mode='w')
    #pdcatalogue.plot()
    return pdcatalogue, sigzeroindexes




#==========================================================
# %%
    
def read_catalogue(filename='.\catalogue.h5'):
    import pandas as pd
    
    try:
        catalogue=pd.read_hdf(filename)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    else: 
        return catalogue
    
    
    
def make_data_catalogue(radgram,outfile='catalogue_trigger.h5',picker='zero',
                        standardizing=0,
                        baselinediff=1,
                        plotoption=0,
                        smin=None,
                        smax=None,
                        dmin=None, 
                        dmax=None,resamplemulti=1):



    '''
    Classifies first arrivals of a file by different picker criteria and 
    returns a catalogue with unique radar traces as a pandas Dataframe
    
    Also writes that catalogue in HDF5 format. 
    filename: path to input file 
    outfile: catalogue output
    picker: picker algorithm for first arrivals 
            'trigger' Seismology trigger algorithm, see module obspytriggers
            'zero' Picks value closest to zero on second pair of min/max
            
    returns:    pdcatalogue         catalogue
                sigzeroindexes      zero cross indexes array
                baselineinds        baseline (direct signal)
                dif_to_median       common meadian of the base  
                counts_sigzeros     count of traces that went into detection
    '''
    meas_data=radgram.traces
    
    meas_linecount=radgram.header["tracenumber"]
    
    
    from readreflex.utils import findmax,findmaxv2,standardize,obspytriggers,shift
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import tqdm
    
    
   
    if standardizing!=0:
        meas_data=standardize(meas_data)

    if picker=='zero':

        maxpeaks,minpeaks,zeroindexes,sigzeroindexes,detailsigzeroes,baselineinds=findmaxv2(meas_data,directmaxbuffer=25*resamplemulti, direct_min=dmin,
                                                                                            direct_max=dmax,smax=smax,smin=smin)
        #maxpeaks,minpeaks,zeroindexes,sigzeroindexes,detailsigzeroes,baselineinds=findmaxv2(meas_data,direct_min=94*8,
              

       # maxpeaks,minpeaks,zeroindexes,sigzeroindexes,detailsigzeroes,baselineinds=findmax(meas_data)

    elif picker=='trigger':
        triggerindexes=obspytriggers(meas_data,meas_linecount)
        sigzeroindexes=triggerindexes
    else:
        print('No valid picker given, using zero')
        maxpeaks,minpeaks,zeroindexes,sigzeroindexes,detailsigzeroes=findmax(meas_data)
        
            
    if  baselinediff==1:
        sigzeroindexes=sigzeroindexes-baselineinds

    # #shift all data so the base is equal 
    # dif_to_median=(np.median(baselineinds)-baselineinds)    
    # for i in tqdm.tqdm(range(0, meas_linecount),'Shifting detection_base_data to common zero'):
    #     meas_data[i,:]=shift(meas_data[i,:],dif_to_median[i])    
    
    #shift all data so the base is 676
    #dif_to_median=(676-baselineinds)    
    #for i in tqdm.tqdm(range(0, meas_linecount),'Shifting detection_base_data to common zero'):
    #    meas_data[i,:]=shift(meas_data[i,:],dif_to_median[i])    
    
    #plot some information:
    uniques, counts_sigzeros=np.unique(sigzeroindexes,return_counts=True)
    # for quality assurance we need to throw away all corrections that do not have more than 3 counts
    countstoolow=counts_sigzeros<3
    throwawayuniques=uniques[countstoolow]
    
    #find where sigzeroindexes contains these values
    deletethesesigzeros_indexes=np.where(np.in1d(sigzeroindexes,throwawayuniques ))[0]
    
    print("Threw away {} detection levels for having not enough detections".format(len(throwawayuniques)))
    print("This makes up {} traces in the current radargram".format(len(deletethesesigzeros_indexes)))
    print("We re-interpolate these gaps. Notice this does not mean that the values are gone")
    #plt.plot(sigzeroindexes)
    #plt.scatter(deletethesesigzeros_indexes,sigzeroindexes[deletethesesigzeros_indexes],4,'r')
    # unfortunately we need to use float so we go to pd for all kinds of cool methods
    
    sigzeroworkframe=pd.Series( sigzeroindexes).astype(float)
    #set them to zero
    sigzeroworkframe[deletethesesigzeros_indexes]=np.nan
    #fill with interpolate
    sigzeroworkframe.interpolate(inplace=True)
    #put the rounded int values back into the array
    sigzeroindexes=sigzeroworkframe.round().values.astype(int)
    
    uniques, counts_sigzeros=np.unique(sigzeroindexes,return_counts=True)
   # plt.figure()
    #plt.scatter(uniques,counts_sigzeros)
   # plt.title('Detection Counts')
   # plt.xlabel("Unique Detections")
    #plt.ylabel("Count")
    

    catalogue=[]
    pdcatalogue=pd.DataFrame()
    basecatalogue=pd.DataFrame()
    #plt.figure()
    #plt.title('Detected Valid Trace-Categories')
    for i in tqdm.tqdm(np.unique(sigzeroindexes)):
        currentcorrection=np.squeeze(np.array(np.where(np.array(sigzeroindexes)==i)))
        if i in [76,510,505]:
            print(currentcorrection)
            print('Debugtime')
        if currentcorrection.size>0:
            #print(currentcorrection.shape)
            #lets care for single traces
            if currentcorrection.size==1:
                print("Correction {} has a seize of 1 trace".format(sigzeroindexes[currentcorrection]))
                catalogue.append(meas_data[currentcorrection,:])
                #Pandas:
                pdcatalogue[str(int(i))]=meas_data[currentcorrection,:]
                if plotoption==1:
                    plt.plot(meas_data[currentcorrection,:])
                
            else:                
              
               
               #meantrace=np.mean(meas_data[currentcorrection,:],axis=0)
               print("Correction {} tracecount is {}".format(i,len(sigzeroindexes[currentcorrection])))
               meantrace=np.mean(meas_data[currentcorrection,:],axis=0)
               #plt.figure()
               #plt.plot(meas_data[currentcorrection,:].T)
               #plt.title(i)
              # plt.pause(0.1)
               #np.append(catalogue,meantrace,axis=0)
               catalogue.append(meantrace)
               pdcatalogue[str(int(i))]=meantrace
               

    if plotoption==1:
        plt.figure()
        extent=[pdcatalogue.columns.astype(int).min(),pdcatalogue.columns.astype(int).max(),pdcatalogue.shape[0],0]
        plt.imshow(pdcatalogue,aspect='auto',extent=extent)    
    #xi = list(range(len(pdcatalogue.columns)))
    #plt.xticks(xi,pdcatalogue.columns().astype(int))
    #plt.xlabel("")      
    pdcatalogue.to_hdf(outfile,key='pdcatalogue',mode='w')
    #pdcatalogue.plot()
    return pdcatalogue,sigzeroindexes,baselineinds,counts_sigzeros,maxpeaks


def export_catalogue(exportfile:str,catalogue,zeros,*args):

    ''' Exports pandas catalogue file and detection lines
    sav
    exportfile: filepath
    cataloguefile: pandas dataframe
    *args: may be provided if baseline was detected
    if args is there, we export a pck ascii file
    '''
    import pandas as pd 

    ''' Exports pandas catalogue file and detextion lines
    sav
    exportfile: filepath
    cataloguefile: pandas dataframe
    *args: may be provided if baseline was detected'''

    
    if args:
        filepieces=exportfile.split('.')
        exportfile=filepieces[0]+'_baselined_'+'.'+filepieces[1]
        print("Writing to ",exportfile)

    catalogue.to_excel(exportfile,header=False)
    print("Done")
    catalogue.to_hdf(exportfile.replace('.xlsx','.hdf5'),key='cat',mode='w')
    print("Wrote to hdf5")
    detectfilename=exportfile.replace('.xlsx','.csv')
    detectfilename=detectfilename.replace('CAT_','CAT_Detects_')
    zeroswritefile=pd.DataFrame(data=catalogue.columns.astype(int))
    zeroswritefile.to_csv(detectfilename,header=False)
    
    

   # catalogue.to_excel(exportfile)
    print("Done")

    
    
        
        

if __name__ == '__main__':
    make_catalogue()
    read_catalogue()
    

