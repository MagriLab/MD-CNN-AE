'''Assorted functions not related to machine learning.'''

class GetAttr(object):
    '''Convert a dictionary to data class

    The values of the dictionary can be accessed as self.key
    '''
    def __init__(self, _dict):
        self.__dict__.update(_dict)