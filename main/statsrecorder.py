import numpy as np
class StatsRecorder:
    def __init__(self):
        """Accumulates normalization statistics across mini-batches.
        """
        self.nobservations = 0   # running number of observations

    def update(self, data):
        # initialize stats and dimensions on first batch
        if self.nobservations == 0:
            self.mean = np.mean(data)
            self.std  = np.std(data)
            self.nobservations = data.shape[0]
        else:

            # find mean of new mini batch
            newmean = np.mean(data)
            newstd  = np.std(data)

            # update number of observations
            m = self.nobservations * 1.0
            n = data.shape[0]

            # update running statistics
            tmp = self.mean
            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            # update total number of seen samples
            self.nobservations += n
