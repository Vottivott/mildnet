import logging

from keras_preprocessing.image import Iterator
from keras_preprocessing.image.iterator import BatchFromFilesMixin


class MildIterator(Iterator):
    """Abstract base class for image data iterators.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self,
                 batch_size,
                 shuffle,
                 seed,
                 triplet_path):
        super(MildIterator, self).__init__(0, batch_size, shuffle, seed)
        count = 0
        f = open(triplet_path)
        f_read = f.read()

        q, p, n = set(), set(), set()
        for line in f_read.split('\n'):
            filenames = line.split(",")
            q.add(filenames[0])
            p.add(filenames[1])
            n.add(filenames[2])
            if len(line) > 1:
                count += 1

        logging.info(
            'Found %d images belonging to %d classes. Query Images: %d, Positive Image: %d, Negative Images: %d' % (
                count, 3, len(q), len(p), len(n)))

        count = count // batch_size * batch_size
        f.close()
        self.n = count * 3

        self.index_generator = self._flow_index()


class MildBatchFromFilesMixin(BatchFromFilesMixin):

    def _get_batches_of_transformed_samples(self, index_array):
        output = super(MildBatchFromFilesMixin, self)._get_batches_of_transformed_samples(index_array)
        if len(output) == 1:
            return output
        else:
            batch_x = output[0]
            batch_y = output[1]
            return [batch_x, batch_x, batch_x], batch_y
