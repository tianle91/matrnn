import numpy as np

class TSSparse(object):

    def __init__(self, evindicator):
        '''initialize sparse arrivals data.
        Args:
            evindicator: has shape (nev, ntime) where it is 1 iff event arrived
        '''
        evinddim = len(evindicator.shape)
        if evinddim != 2:
            raise ValueError('dim of evindicator:%d is not 2' % evinddim)
        evindunique = np.unique(evindicator)
        if not np.all(np.in1d(evindunique, np.array([0., 1.]))):
            raise ValueError('evindicator has unique values:', evindunique)

        self.ind = evindicator
        self.nev, self.ntime = evindicator.shape
        self.tse = self.make_tse()
        self.tte = self.make_tte()

    def make_tse(self):
        out = np.zeros(self.ind.shape)
        accum_tse = np.zeros(self.nev)
        for tdex in range(self.ntime):
            # where no event at tdex, tse increases by 1
            accum_tse[np.where(self.ind[:, tdex] == 0)] += 1
            # otherwise event occured, so tse is 0
            accum_tse[np.where(self.ind[:, tdex] == 1)] = 0
            out[:, tdex] = accum_tse
        return out

    def make_tte(self):
        out = np.zeros(self.ind.shape)
        accum_tte = np.zeros(self.nev)
        for tdex in range(self.ntime-1, 0, -1):
            # where no event at dex, tte incrases by 1
            accum_tte[np.where(self.ind[:, tdex] == 0)] += 1
            # otherwise event occured, so tte is 0
            accum_tte[np.where(self.ind[:, tdex] == 1)] = 0
            out[:, tdex-1] = accum_tte
        return out

    def make_firstev(self):
        return np.argmax(self.ind, axis=1)

    def make_lastev(self):
        # np.argmax returns first instance of hitting max (i.e. 1)
        # so ::-1 reverses order of self.ind and we find the last index
        # and -1*(...) gives the actual last location in non-reversed self.ind
        return -1+(-1*np.argmax(self.ind[:, ::-1], axis=1))

    def make_unc(self):
        lastev = self.make_lastev()
        out = np.ones(self.ind.shape)
        for evdex in range(self.nev):
            out[evdex, lastev[evdex]:] = 0
        return out

    def make_pch(self):
        firstev = self.make_firstev()
        out = np.ones(self.ind.shape)
        for evdex in range(self.nev):
            out[evdex, :firstev[evdex]] = 0
        return out

    def transformations(self):
        out = {
            'time_since_event': self.tse,
            'time_to_event': self.tte,
            'is_uncensored': self.make_unc(),
            'is_notfirstrodeo': self.make_pch()
        }
        return out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    x = np.zeros((10, 20))
    x[0, 0] = 1
    x[1, 1] = 1
    x[2, ::1] = 1
    x[3, ::2] = 1
    x[4, ::3] = 1
    x[5, ::4] = 1
    x[6, ::5] = 1
    x[7, ::6] = 1
    x[8, -1] = 1
    x[9, -2] = 1
    
    print ('x array:')
    print (x)    
    
    tssobj = TSSparse(x)
    tssobjtstfs = tssobj.transformations()
    print ('firstev:\n', tssobj.make_firstev())
    print ('lastev:\n', tssobj.make_lastev())
    
    
    print ('x imshow')
    plt.imshow(x)
    plt.show()

    print ('tse imshow')
    plt.imshow(tssobjtstfs['time_since_event'])
    plt.show()
    
    print ('tte imshow')
    plt.imshow(tssobjtstfs['time_to_event'])
    plt.show()
    
    print ('unc imshow')
    plt.imshow(tssobjtstfs['is_uncensored'])
    plt.show()
    
    print ('pch imshow')
    plt.imshow(tssobjtstfs['is_notfirstrodeo'])
    plt.show()