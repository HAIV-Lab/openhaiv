

class BaseMeter(object):
    """Base class for meters.

    Attributes:
        value (Any): Value of meter.
        
    """
    def __init__(self):
        super(BaseMeter, self).__init__()
        self.reset()

    @property
    def value(self):
        """Get value of meter."""
        raise NotImplementedError

    def reset(self):
        """Reset meter."""
        raise NotImplementedError
    
    def update(self, val, n: int = 1):
        """Update meter.
        
        Args:
            val (Any): Value to update.
            n (int): Number of times to update.
        """
        raise NotImplementedError


class AverageMeter(BaseMeter):
    """Computes and stores the average and current value.

    Attributes:
        val (Any): Current value.
        avg (Any): Average value.
        sum (Any): Sum of values.
        count (int): Number of values.
        value (Any): Average value.

    Example:
        >>> meter = AverageMeter()
        >>> meter.update(1)
        >>> meter.avg
        1.0
        >>> meter.update(2)
        >>> meter.avg
        1.5
        
    """
    @property
    def value(self):
        return self.avg

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
