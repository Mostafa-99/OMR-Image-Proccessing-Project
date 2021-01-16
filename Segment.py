from Class import *
from Features import *
import numpy as np

class BoundingBox:
    def __init__(self, minx, maxx, miny, maxy, integer=False):
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        if(integer):
            self.ToInt()

    def ToInt(self):
        self.minx = self.minx.astype(int)
        self.maxx = self.maxx.astype(int)
        self.miny = self.miny.astype(int)
        self.maxy = self.maxy.astype(int)
    
    def __repr__(self):
        return '[%i, %i, %i, %i]' % (self.minx, self.maxx, self.miny, self.maxy)
    
    def __add__(self, other):
        _minx = self.minx + other.minx
        _maxx = self.maxx + other.maxx
        _miny = self.miny + other.miny
        _maxy = self.maxy + other.maxy
        return BoundingBox(_minx, _maxx, _miny, _maxy)

class Segment:
    def __init__(self, BoundingBox = None, VerticalHistogram = None, HorizontalHistogram = None, BoundingBoxToLineSpaceRatio = None
                 , Features=None ):
        self.VerticalHistogram = VerticalHistogram
        self.HorizontalHistogram = HorizontalHistogram
        self.BoundingBoxToLineSpaceRatio = BoundingBoxToLineSpaceRatio
        self.BoundingBox = BoundingBox
        self.Features=Features
        
    
    def CreateImage(self):
        self.Image = np.zeros((self.BoundingBox.maxy - self.BoundingBox.miny + 1, self.BoundingBox.maxx - self.BoundingBox.minx + 1))
        self.FilledImage = np.copy(self.Image)
        

class Note:
    def __init__(self, BoundingBox = None):
        self.Type = 'single' #single or chord
        self.Noteheads = [] #incase of chord will have len > 1
        self.BoundingBox = BoundingBox
        self.Hollow = []
    def __repr__(self):
        string = 'Type is : ' +self.Type
        string+= '\nNoteheads are :\n'
        k = 0
        for note in self.Noteheads:
            string += '('+str(note[1]) + ', ' + str(note[0]) + ')\n'
            string += 'Notehead is: '
            if(self.Hollow[k]):
                string+= ' Hollow\n'
            else:
                string+= ' Not Hollow\n'
            k = k + 1
        string += '\nBounding box is ' + repr(self.BoundingBox) + '\n\n'
        return string
        
    

class Symbols:
    def __init__(self, BoundingBox = None):
        self.Type = 'timing' #single or chord
        self.Center = [] #incase of chord will have len > 1
        self.BoundingBox = BoundingBox
    def __repr__(self):
        string = 'Type is : ' +self.Type
        string+= '\nCenter is :\n'
        k = 0
        string += '('+str(self.Center[1]) + ', ' + str(self.Center[0]) + ')\n'
        string += '\nBounding box is ' + repr(self.BoundingBox) + '\n\n'
        return string
        
        