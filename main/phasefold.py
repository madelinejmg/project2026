import numpy as np
import matplotlib.pyplot as plt

def phasefold(T0,time,period,flux):
    """
    This function will phase-fold the input light curve (time, flux)
    using a Mid-transit time and orbital period. The resulting phase
    is centered on the input Mid-transit time so that the transit
    occurs at phase 0.
    
    Input Parameters
    ----------
    time: array
        An array of timestamps from TESS observations.        
    TO : float
        The Mid-transit time of a periodic event.
    period : float
        An orbital period of a periodic event.
    flux : array
        An array of flux values from TESS observations.
    Returns
    -------
        * phase : array
            An array of Orbital phase of the phase-folded light curve.
        * flux : array
            An array of flux values from TESS observations of the 
            phase-folded light curve.
    """          
    phase=(time- T0 + 0.5*period) % period - 0.5*period
    ind=np.argsort(phase, axis=0)
    return phase[ind],flux[ind]


def plot_phasefolded(ax, target, lc, color, label):
    target_P= target['Orbital Period (days) Value'].item()
    target_T0= target['Orbital Epoch Value'].item()
    target_Dep = target['Transit Depth Value'].item()/1e6
    target_Dur = target['Transit Duration (hours) Value'].item()
       
    pf,ff = phasefold(T0=target_T0, time=lc['Time'].to_numpy(), 
                      period=target_P, flux=lc['Corrected Flux'].to_numpy()/np.nanmedian(lc['Corrected Flux'].to_numpy()))
    
    ax.scatter(24*pf,ff, s=3, color=color,label=label)
    ax.set_xlim(-5*target_Dur,5*target_Dur)
    
    ymin = np.nanmin(ff[np.abs(24*pf<5*target_Dur)]) - 3*np.nanstd(ff[np.abs(24*pf<5*target_Dur)])
    ymax = np.nanmax(ff[np.abs(24*pf<5*target_Dur)]) + 3*np.nanstd(ff[np.abs(24*pf<5*target_Dur)])
    return ymin, ymax
    
