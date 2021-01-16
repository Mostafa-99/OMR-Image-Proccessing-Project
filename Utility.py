from commonfunctions import *
from Segment import *
from Features import *

#Functions collections
'''          
1- General functions
2- Preproccessing functions
3- Line thickness, space and removal functions
4- Segmentation functions
5- NoteHeads Functions
6- Classif.
7- Calling functions(Req For main)
'''          
#****************************************************General Functions*********************************************************
    
def addToFreqArr(uniqueArrays,frqArr,val1,val2):
    found=False
    index=-1
    y=0
    for (u1,u2) in uniqueArrays:
        if(val1==u1 and val2==u2):
            found=True
            index=y
            break
        y=y+1
    if(found==False):
        arr1=[val1,val2]
        uniqueArrays.append(arr1)
        frqArr.append(1)
    else:
        frqArr[index]=frqArr[index]+1
    return uniqueArrays,frqArr


#*****************************************************************************************************************************#
#****************************************************Preproccessing Functions*************************************************
def getRotationAngle(image):
    edges =canny(image, sigma=1)
    hough_lines = probabilistic_hough_line(edges)
    slopes=[]
    for line in hough_lines:
        p1,p2=line
        if(p2[0]!=p1[0]):
            slopes.append((p2[1] -p1[1])/(p2[0] - p1[0]))

    deg_angles = np.degrees(np.arctan(slopes))
    
    histo = histogram(deg_angles, nbins=180)
    
    rotationAngle = histo[1][np.argmax(histo[0])]
    if(int(rotationAngle)==0):
        return 0
    if(int(rotationAngle)<0):
        rotationAngle=rotationAngle+180.0
    return rotationAngle

def crop(image,imageOrignal):
    tmp = get_OCR_H(image, OCR_SIZE=(-1, -1))

    tmpSorted = np.copy(tmp)
    tmpSorted.sort()
    avg = 0
    for i in range(5):
        avg += tmpSorted[-i]
    avg/=5;

    peak = find_peaks(tmp, height = avg*0.4)
    ys = (peak[0][0] *0.8).astype(int)
    if (ys < 100):
        ys = 0
    ye = (peak[0][-1] *1.2).astype(int)
    if ( image.shape[0] - ye < 100):
        ye = image.shape[0]
    
    tmp = get_OCR_V(image, OCR_SIZE=(-1,-1))
    xs = min(np.where(tmp >4)[0])
    xe = max(np.where(tmp > 4)[0])
    xs = (xs*0.8).astype(int)
    if (xs < 100):
        xs = 0
    xe = (xe*1.2).astype(int)
    if ( image.shape[1] - xe < 100):
        xe = image.shape[1]
    new = np.copy(image[ys:ye, xs:xe])
    newNotBinary = np.copy(imageOrignal[ys:ye, xs:xe])

    return new,newNotBinary


def preproccessing(image):
    angle = getRotationAngle(image)
    rotatedImage = rotate(image, angle,cval=0.7,resize=True)    
    imageCopy=np.copy(rotatedImage)
    thresh = threshold_sauvola(imageCopy, window_size=125)
    binary = np.where(imageCopy <= thresh , 1,0)
    if(angle!=0):
        binary = binary_dilation(binary, np.ones((2,8)))
        scanned = False
    else:
        scanned = True
    return binary,scanned,rotatedImage
#*****************************************************************************************************************************#
#****************************************************Line Functions***********************************************************

#take only top 5 lines
rleSize=11
maxNumberOfChanges=5

def calcRLE(image,colNumber):
    numberOfChanges=0
    rle=np.zeros(rleSize)
    rleIndex=0
    #calc RLE of col
    for y in range(1,image.shape[0]-1):
        if(image[y][colNumber]==1):
            if(image[y-1][colNumber]==0):
                numberOfChanges = numberOfChanges + 1
                rleIndex = rleIndex + 1
            rle[rleIndex]= rle[rleIndex]+1
        else:
            if(numberOfChanges==maxNumberOfChanges):
                rle[rleSize-1]=rle[0]
                break
                
            if(image[y-1][colNumber]==1 and image[y+1][colNumber]==0):
                rleIndex = rleIndex + 1
                
            rle[rleIndex]= rle[rleIndex]+1
                
        if(y==image.shape[0]-1):
            rle[rleSize-1]=rle[0]
    return rle

def formAndGetMostRepetedPair(arrs):
    #Get most repeated pair 
    uniqueArrays=[]
    frqArr=[]
    lineThickness=0
    lineSpace=0
    for (val1,val2) in arrs:
        addToFreqArr(uniqueArrays,frqArr,val1,val2)

    maxIndex = np.where(frqArr == np.amax(frqArr))[0][0]
    return uniqueArrays,maxIndex

def getLineThicknessSpacing_Col(image,colNumber):
    
    rle=calcRLE(image,colNumber)
   
    #Sum each two consecutive numbers
    sumConsecutive=np.zeros(rleSize-1)
    for x in range(0,sumConsecutive.shape[0]-1):
        sumConsecutive[x]=rle[x]+rle[x+1]
    
    #calc Histogram of sum
    freqArr=collections.Counter(sumConsecutive)
    
    thicknessPlusSpace=-99999
    maxValue=-9999
    #Find max sum
    for (key, value) in freqArr.items(): 
        if(value>maxValue):
            thicknessPlusSpace=key
            maxValue=value
    
    #Make pairs that thier sum is equal to max sum
    arrs=[]
    for x in range(0,rle.shape[0]-1):
        if(rle[x]+rle[x+1]==thicknessPlusSpace):
            arr1=[rle[x],rle[x+1]]
            arrs.append(arr1)
    
    uniqueArrays,maxIndex = formAndGetMostRepetedPair(arrs)
    
    
    # Line Thickness is always < Line space
    if(uniqueArrays[maxIndex][0]>uniqueArrays[maxIndex][1]):
        lineThickness=uniqueArrays[maxIndex][1]
        lineSpace=uniqueArrays[maxIndex][0]
    else:
        lineThickness=uniqueArrays[maxIndex][0]
        lineSpace=uniqueArrays[maxIndex][1]
        
    return lineThickness,lineSpace

def getLineThicknessSpacing_FullImage(image):
    #Calc Line Thickness and Space
    frqArr=[]
    uniqueArrays=[]
    for x in range(0,image.shape[1]):
        (lineThickness,lineSpace) = getLineThicknessSpacing_Col(image,x)
        if(lineThickness !=0 and lineSpace != 0):
            addToFreqArr(uniqueArrays,frqArr,lineThickness,lineSpace)
       
    maxIndex = np.where(frqArr == np.amax(frqArr))[0][0]
    lineThickness=uniqueArrays[maxIndex][0].astype(int)
    lineSpace=uniqueArrays[maxIndex][1].astype(int)
    return lineThickness,lineSpace


def groupLines(linesList, lineThickness, lineSpace):
    if(len(linesList)==0): return []
    linesList.sort()
    groups = list()
    tmp = list()
    tmp.append(linesList[0])
    i = 1
    while(i < len(linesList)):
        while(i < len(linesList) and linesList[i] - linesList[i-1] <= 1.5*(lineThickness+lineSpace)):
            tmp.append(linesList[i])
            i = i + 1
        groups.append(tmp)
        if(i >= len(linesList)): break
        tmp = list()
        tmp.append(linesList[i])
        i = i + 1
    return groups

def formLineGroups(image, lineThickness, lineSpace):
    tested_angles = np.linspace( -np.pi , np.pi , 360)
    h, theta, d = hough_line(image)

    #list of lines to be used when classifying
    linesList = list()
    
    origin = np.array((0, image.shape[1]))
   
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d,min_distance=lineSpace)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        if(abs(y0.astype(int)-y1.astype(int))<=50):
            linesList.append(((y0+y1)//2).astype(int))

    groups = groupLines(linesList, lineThickness, lineSpace) 
    return groups
def checkAndFixLineGroups(lineGroups):
    uniqueArrays=[]
    frqArr=[]
    for i in range(len(lineGroups)):
        addToFreqArrOneVal(uniqueArrays,frqArr,len(lineGroups[i]))
            
            
    maxIndex = np.where(frqArr == np.amax(frqArr))[0]
 
    numberOfGroups=uniqueArrays[maxIndex[0]]
    groupArr=[]
    for i in range (0,len(lineGroups)):
        if(len(lineGroups[i]) != numberOfGroups):
            if(i==0 or i==1):
                for z in range(0,len(lineGroups)):
                    if(len(lineGroups[z])==numberOfGroups):
                        groupArr=lineGroups[z]
                        break       
            elif(i==2):
                for z in range(1,len(lineGroups)+1):
                    if(len(lineGroups[z%4])==numberOfGroups):
                        groupArr=lineGroups[z%4]
                        break    
            elif(i==3):
                for z in range (len(lineGroups)-2,-1,-1):
                    if(len(lineGroups[z]) ==numberOfGroups):
                        groupArr=lineGroups[z]
                        break
                            
            if(len(groupArr)!=0):
                lineGroups[i]=[]
                lineGroups[i]=groupArr
                groupArr=[]
    
    for i in range (0,len(lineGroups)):
        for j in range (0,len(lineGroups[i])): 
            if(len(lineGroups[i][j]) !=5):
                fixGroup(lineGroups,i,j)
    
def fixGroup(lineGroups,i,j):
    nextArr=[]
    if(i==3):
        for z in range (len(lineGroups)-2,-1,-1):
            if(len(lineGroups[z][j]) ==5):
                if(len(nextArr)==0): #first group
                    nextArr=lineGroups[z][j]
                    break
    elif(i==2):
        if(len(lineGroups[3][j]) ==5):
            if(len(nextArr)==0):
                nextArr=lineGroups[3][j]
        else:
            for z in range (1,-1):
                if(len(lineGroups[z][j]) ==5):
                    if(len(nextArr)==0):
                        nextArr=lineGroups[z][j]
                        break
    else:
        for z in range (0,len(lineGroups)):
            if(len(lineGroups[z][j]) ==5):
                if(len(nextArr)==0): #first group
                    nextArr=lineGroups[z][j]

    lineGroups[i][j]=nextArr[:]

def detectAndRemoveHorizontalLines(image,lineThickness, lineSpace):
    tested_angles = np.linspace( -np.pi , np.pi , 360)
    h, theta, d = hough_line(image)
    
    origin = np.array((0, image.shape[1]))
    
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d,min_distance=lineSpace)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        if(abs(y0.astype(int)-y1.astype(int))<=50):
            for x in range(0,image.shape[1]):
                #remove lines if not part of a symbol
                if(image[y0.astype(int)-(lineThickness+1)][x]!=1 and image[y0.astype(int)+(lineThickness+1)][x]!=1):
                    image[y0.astype(int)-(lineThickness)-0:y0.astype(int)+(lineThickness)+0,x] = 0

    return image
#*****************************************************************************************************************************#
#****************************************************Segmentation Functions***************************************************
circlesMergeThreshold = 5
noteheadsRatioThreshold = 1.8
numberOfVotesToDelete = 2

selement = [[0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0]]
selement2=[[1,1,1],
           [1,1,1],
           [1,1,1]]

def FillPolygon(image, contours):
    img = np.copy(image)
    for contour in contours:
        #Fill the polygon to fill the holes
        rr,cc = polygon(contour[:,0],contour[:,1],image.shape)
        img[rr.astype(int),cc.astype(int)]=1
    return img

def SegmentImage(image, filledImage, contours):
    #Create a list for all the segments found in the image
    segments=list()
    #Create a collective image that has all segments
    imagecollective = np.zeros((image.shape[0],image.shape[1]))
    imagecollective[:,:]=0.5
    #Segment every contour
    for contour in contours:
        #Create new segment
        segment = Segment()
        #Get aspect ratio
        segment.BoundingBox = BoundingBox(np.min(contour[:,1]), np.max(contour[:,1]), np.min(contour[:,0]), np.max(contour[:,0]))    
        segment.BoundingBox.ToInt()
        #Create the segment image (will be replaced with segment class)
        segment.CreateImage()
        #Copy the segment from the contour we have to the segment
        rr,cc = polygon(contour[:,0],contour[:,1],image.shape)
        rr=rr.astype(int)
        cc=cc.astype(int)
        segment.Image[rr-segment.BoundingBox.miny,cc-segment.BoundingBox.minx] = image[rr,cc]
        segment.FilledImage[rr-segment.BoundingBox.miny,cc-segment.BoundingBox.minx] = filledImage[rr,cc]
        imagecollective[rr,cc] = image[rr,cc]
        segments.append(segment)
    return imagecollective, segments

def SaveSegments(segments):
    #Save every segment in a seperate file
    i = 0
    for seg in segments:
        #Check if size is large to prevent small dots/noise from being saved
        if(seg.Image.shape[0] >= 20 and seg.Image.shape[1] >= 20):
            mpl.image.imsave('segments/indiv/'+str(i)+'.bmp',seg.Image[:,])
            i=i+1
            

    
def SegmentBeaming(segment):
    index = int(segment.Image.shape[0]/2.5)
    segment.Image = np.array(segment.Image[index:,:])
    #find the contours for the image
    contours = find_contours(segment.Image, 0.9)
    filledSegment = FillPolygon(segment.Image, contours)
    #Find the contours after filling, no hole contours exist
    contours = find_contours(filledSegment,0.9)
    return SegmentImage(segment.Image, contours)[1]


def MergeHalfCircles(circles):
    newCirclesList = list()
    merged = np.zeros(len(circles))
    for i in range(len(circles)):
        if(merged[i]==1):
            continue
        c1 = circles[i]
        c2 = None
        for j in range(len(circles)):
            if(i==j or merged[j]): 
                continue            
            if(abs(c1.BoundingBox.miny - circles[j].BoundingBox.miny) < circlesMergeThreshold and abs(c1.BoundingBox.maxy - circles[j].BoundingBox.maxy) < circlesMergeThreshold and (abs(c1.BoundingBox.minx - circles[j].BoundingBox.maxx) < circlesMergeThreshold*10 or abs(c1.BoundingBox.maxx - circles[j].BoundingBox.minx) < circlesMergeThreshold*10 ) ):
                c2 = circles[j]
                merged[j] = 1
                break
        if(c2 != None and c1.BoundingBox.minx > c2.BoundingBox.minx):
            c1,c2 = c2,c1
        width = c1.Image.shape[1]
        height = c1.Image.shape[0]
        if(c2!=None):
            width = max(width, c2.Image.shape[1])
            height = max(height, c2.Image.shape[0])
        #Resize img1
        tmp = np.copy(c1.Image)
        c1.Image = np.zeros((height, width))
        c1.Image[:tmp.shape[0], :tmp.shape[1]] = tmp[:,:]
        if(c2!=None):
            #Resize img2
            tmp = np.copy(c2.Image)
            c2.Image = np.zeros((height, width))
            c2.Image[:tmp.shape[0], :tmp.shape[1]] = tmp[:,:]
            newCircle = np.concatenate((c1.Image,c2.Image),axis=1)
            segment = Segment()
            segment.BoundingBox = c1.BoundingBox
            segment.Image = newCircle
            newCirclesList.append(segment)
            merged[i] = 1
        else:
            newCirclesList.append(c1)
    connectCircles(newCirclesList)
    for circle in newCirclesList:
        contours = find_contours(circle.Image, 0.9)
        circle.FilledImage = FillPolygon(circle.Image, contours)
    return newCirclesList 

                   
def connectCircles(circles):
    for circle in circles:
        height = circle.Image.shape[0]
        width = circle.Image.shape[1]
        circle.Image[0:height//6,width//4:(3*width//4)] = 1
        circle.Image[height-height//6:,width//4:(3*width//4)] = 1
        
def classifyCircle(circle):
    contours = find_contours(circle, 0.9)
    filledCircle = FillPolygon(circle, contours)
    diff = np.copy(filledCircle)
    diff = (diff - circle).astype(int)
    selement = [[1,1,1],
               [1,1,1],
               [1,1,1]]
    diff=binary_erosion(diff,selement)
    contours = find_contours(diff, 0.9)
    
    
def DetectNoteheads(notesAndConnectedSegments):
    newList = list()
    newSegmentsList = list()
    for seg in notesAndConnectedSegments:
        image=np.copy(seg.Image)
        tmpImg = np.copy(image[1:-2,1:-2])
        ones = np.sum(image[5:-5,5:-5]==1)
        zeros = np.sum(image[5:-5,5:-5]==0)
        
        if(ones > 0 and zeros/ones>1.2):
            image=ndimage.binary_fill_holes(image).astype(int)
            image=binary_erosion(image,selement)
            image=binary_erosion(image,selement)
            image=binary_erosion(image,selement2)

            newSeg=Segment(seg)
            newSeg.Image=image
            newSeg.BoundingBox=seg.BoundingBox
            newList.append(image)
            newSegmentsList.append(newSeg)
    return newSegmentsList, newList

def CheckNoteheadsRatios(img):
    contours = find_contours(img, 0.9)
    pixels = list()
    areas = list()
    for contour in contours:
        rr,cc = polygon(contour[:,0],contour[:,1],img.shape)
        areas.append(len(rr)+1)
        pixels.append((rr,cc))
    #If one notehead, or something to be removed (should be handled)
    if(len(contours)==1):
        return True
    #If 2 noteheads, or cliff
    if(len(contours)==2):
        return (max(areas[0]/areas[1], areas[1]/areas[0]) <= noteheadsRatioThreshold)
    else:
        votes = np.zeros(len(contours))
        for i in range(len(areas)):
            for j in range(len(areas)):
                if (i == j): continue
                if(max(areas[i]/areas[j], areas[j]/areas[i]) > noteheadsRatioThreshold):
                    votes[j] = votes[j] + 1
        for i in range(len(votes)):
            if(votes[i] >= numberOfVotesToDelete):
                img[pixels[i]] = 0
        return True

def FilterNoteheads(notesAndConnectedSegments, filledNoteheads):
    return_list = list()
    i = 0
    for seg in notesAndConnectedSegments:

        if(CheckNoteheadsRatios(filledNoteheads[i])):
            return_list.append(seg)
        i = i + 1

    return return_list

def FindNoteheadsPosition(filteredList):
    FinalResult=list()
    for seg in filteredList:
        contours = find_contours(seg.Image, 0.9)
        imagecollective, segments = SegmentImage(seg.Image, contours)
        
        noteHeadsCenters=list()
        segType='note'
        for noteHead in segments:
            centerX=(noteHead.BoundingBox.minx+ noteHead.BoundingBox.maxx)//2 + seg.BoundingBox.minx
            centerY=(noteHead.BoundingBox.miny+ noteHead.BoundingBox.maxy)//2 + seg.BoundingBox.miny
            noteHeadsCenters.append((centerX,centerY))
        FinalResult.append((noteHeadsCenters,segType))
    return FinalResult

#*****************************************************************************************************************************#
#****************************************************NoteHeads Functions***************************************************
def cSum(arr,lineSpace):
    accum=0
    result=[]
    indices = []
    startX = 0
    for x in range(1,len(arr)):
        if(arr[x]==1):
            accum+=1
        elif(arr[x-1]==1):
            if(accum>=lineSpace/2 or lineSpace/2-accum<=2):
                result.append(accum)
                indices.append((startX,x))
            accum=0
            startX = x
        else:
            startX = x
    if(arr[len(arr)-1]==1):
        if(accum>=lineSpace/2 or lineSpace/2-accum<=2):
            result.append(accum)
            indices.append((startX, len(arr)-1))
    return result, indices
def calcBiggestRun(image,colNumber,color=1):
    currentMax=0
    currentRun=0

    for y in range(1,image.shape[0]):
        if(image[y][colNumber]==color):
            currentMax=max(currentMax,currentRun)
            if(image[y-1][colNumber]!=color):
                currentRun=0
            currentRun+=1
    return currentMax

def calcBiggestRunRows(image,rowNumber,color=1):
    currentMax=0
    currentRun=0

    for x in range(1,image.shape[1]):
        if(image[rowNumber][x]==color):
            currentMax=max(currentMax,currentRun)
            if(image[rowNumber][x-1]!=color):
                currentRun=0
            currentRun+=1
    return currentMax

def findNoteHeadsHorizontal(seg,lineThickness, lineSpace):
    image=np.copy(seg[1])
    flags=np.zeros(seg[1].shape[0])
    for y in range(0,seg[1].shape[0]):
        biggestRun=calcBiggestRunRows(seg[1],y)
        if(biggestRun<(3/2)*lineSpace and biggestRun> 2*lineThickness):
            flags[y]=1
    lenghts, indices = cSum(flags,lineSpace)
    
    segments = []
    yPosOffset = []
    for y1, y2 in indices:
        tmp = np.copy(seg[0])
        tmp2 = np.copy(seg[1])
        img1 = np.copy(tmp[y1:y2,:])
        img2 = np.copy(tmp2[y1:y2,:])
    
        s = str(y1) + "," + str(y2)
        segments.append([img1, img2])
        yPosOffset.append(y1)
    return segments, yPosOffset


def findNoteHeads(seg,lineThickness, lineSpace):
    image=np.copy(seg.Image)
    ones = np.sum(image[5:-5,5:-5]==1)
    zeros = np.sum(image[5:-5,5:-5]==0) 
    if(ones <= 0 or zeros/ones<=1):
        return None, None
    flags=np.zeros(seg.Image.shape[1])
    for x in range(0,seg.Image.shape[1]):
        biggestRun=calcBiggestRun(seg.Image,x)
        if(biggestRun<(3/2)*lineSpace and biggestRun> 2*lineThickness):
            flags[x]=1
   
    lenghts, indices = cSum(flags,lineSpace)
    segments = []
    xPosOffset=[]
    for x1, x2 in indices:
        img1 = np.copy(seg.Image[:,x1:x2])
        img2 = np.copy(seg.FilledImage[:,x1:x2])
        segments.append((img1,img2))
        xPosOffset.append(x1)
    return segments, xPosOffset

def addToFreqArrOneVal(uniqueArrays,frqArr,val1):
    found=False
    index=-1
    y=0
    for (u1) in uniqueArrays:
        if(val1==u1):
            found=True
            index=y
            break
        y=y+1
    if(found==False):
        uniqueArrays.append(val1)
        frqArr.append(1)
    else:
        frqArr[index]=frqArr[index]+1
    return uniqueArrays,frqArr

def getNumberOfLinesForSegment(image,THD=15):
    #Calc Line Thickness and Space
    frqArr=[0]
    uniqueArrays=[0]
    for x in range(0,image.shape[1]):
        numberOfLines = getNumberOfLines_Col(image,x)
        if(numberOfLines!=0):
            addToFreqArrOneVal(uniqueArrays,frqArr,numberOfLines)
       
    maxIndex = np.where(frqArr == np.amax(frqArr))[0]
    if(frqArr[maxIndex[0]] <THD):
        return 0
    numberOfLines=uniqueArrays[maxIndex[0]]
    return numberOfLines

def getNumberOfLines_Col(image, x):
    numOfLines = 0
    for y in range(0,image.shape[0]-1):
        if(image[y,x] == 1 and image[y+1,x] == 0):
            numOfLines+=1
    return numOfLines


def GetBeamLines(image,imageFilled,THD=15):
    tmp = np.copy(image)
    selementHorizontal = [[0,0,0],
                         [1,1,1],
                         [0,0,0]]
    selementVetrical=[[0,1,0],
                     [0,1,0],
                     [0,1,0]]
    tmp = binary_erosion(tmp, selementVetrical)
    tmp = skeletonize(tmp)
    tmp = binary_erosion(tmp, selementHorizontal)
    v=getNumberOfLinesForSegment(tmp,THD)
    return v

def GetBeamLinesFlag(image, imageFilled,THD,lineSpace): 

    selementDiagonal=  [[1,0,0],
                        [0,1,0],
                        [0,0,1]]

    tmp = np.copy((image))
    tmp = binary_erosion(tmp, selementDiagonal)  
    tmp = binary_dilation(tmp,selementDiagonal)
    h, theta, d = hough_line(tmp)
    v=0
    origin = np.array((0, tmp.shape[1]))
    thres = 0* np.max(h)
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d,min_distance=(lineSpace//4))):
        #print("angle: ",angle)
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)

        if(angle<0 and abs(angle) <= ((50/180)*np.pi) and abs(angle) >= ((40/180)*np.pi)):
            v+=1
    return v

def NoteHeadValid(image,imageFilled,beamLines):
    if(IsHollow(image,imageFilled) == False):
        return 1
    
    tmp=np.zeros((image.shape[0]+2,image.shape[1]+2))
    tmp[1:image.shape[0]+1,1:image.shape[1]+1]=image
    tmp = skeletonize(tmp)
    tmp = binary_dilation(tmp)

    if(GetBeamLines(tmp,tmp,8)==beamLines and beamLines !=0):
        return 0
    return 1

def GetCoordinates(notehead, boundingBox, ypos, xpos):
    centerX = notehead.shape[1]//2 + boundingBox.minx + xpos
    centerY = notehead.shape[0]//2 + boundingBox.miny + ypos   
    return (centerX,centerY)
    
def IsHollow(note, filledNote):
    ones1 = np.sum(filledNote[:,:]==1)
    ones2 = np.sum(note[:,:]==1)
    return (ones1/ones2) > 1.1


#Returns:
#Note structure consists of: -NoteType = Single, Chord (in case of cord having beamlines, then it is a beam)
#         -Vector of noteheads: Every note head has: xpos,ypos
#         -BoundingBox (minx, miny, maxx, maxy)
#         -return Number of lines (normal = 0)
def GetNote(segment, lineThickness, lineSpace):
    noteList = [] 
    note = Note(segment.BoundingBox)
    verticalSegments, vertSegmentXPos = findNoteHeads(segment,lineThickness, lineSpace)
    
    if(verticalSegments == None): return None
    beamLines = GetBeamLines(segment.Image, segment.FilledImage, 15)
    b = segment.BoundingBox
    asp_ratio = (b.maxy-b.miny)/(b.maxx-b.minx)
    if (asp_ratio > 1.25):
        beamLines = 0

    horizontalSegmentsInSecondVerticalSegment = ()
    for j in range(len(verticalSegments)):
        horizontalSegments, horizSegmentYPos = findNoteHeadsHorizontal(verticalSegments[j], lineThickness, lineSpace)
        if(j==1): horizontalSegmentsInSecondVerticalSegment=horizontalSegments
        for i in range(len(horizontalSegments)):
            if(NoteHeadValid(horizontalSegments[i][0],horizontalSegments[i][1],beamLines)):
                note.Noteheads.append(GetCoordinates(horizontalSegments[i][0],segment.BoundingBox,horizSegmentYPos[i],vertSegmentXPos[j]))
                note.Hollow.append(IsHollow(horizontalSegments[i][0], horizontalSegments[i][1]))
        
    if(len(note.Noteheads) == 1):
        note.Type = 'single'
        beamLines = 0
    elif(len(note.Noteheads) == 2):
        if(len(verticalSegments)==2 and len(horizontalSegmentsInSecondVerticalSegment)==1):
            if((verticalSegments[1][0].shape[0]*verticalSegments[1][0].shape[1])/(horizontalSegmentsInSecondVerticalSegment[0][0].shape[0]*horizontalSegmentsInSecondVerticalSegment[0][0].shape[1])<=3):
                beamLines=GetBeamLinesFlag(verticalSegments[1][0],verticalSegments[1][1],15,lineSpace)
                note.Type = 'single'
                note.Noteheads.pop()
            else:#one line or chord
                tmp = np.copy(horizontalSegmentsInSecondVerticalSegment[0][0])
                tmp = binary_erosion(tmp, selement)
                ones=np.sum(tmp[:,:] == 1)
                zeros=np.sum(tmp[:,:] == 0)
                if(ones<100):
                    beamLines=GetBeamLinesFlag(verticalSegments[1][0],verticalSegments[1][1],15,lineSpace)
                    note.Type = 'single'
                    note.Noteheads.pop()
                else:
                    note.Type = 'chord'           
        else:
            note.Type = 'chord'
    else:
        note.Type = 'chord'

    newNoteheadsList = list()
    taken = np.zeros(len(note.Noteheads))
    for i in range(len(note.Noteheads)):
        if (taken[i] == 0):
            newNoteheadsList.append(note.Noteheads[i])
        else:
            temp = (note.Noteheads[len(newNoteheadsList)-1][0],note.Noteheads[len(newNoteheadsList)-1][1]+lineSpace//4)
            note.Noteheads[len(newNoteheadsList)-1]=temp
            newNoteheadsList[len(newNoteheadsList)-1] = note.Noteheads[len(newNoteheadsList)-1]
        for j in range(len(note.Noteheads)):
            if (abs(note.Noteheads[i][0]-note.Noteheads[j][0]) <= lineSpace/2 and abs(note.Noteheads[i][1] - note.Noteheads[j][1]) <= (lineSpace+lineThickness) / 1.8):
                taken[j] = 1
    note.Noteheads = newNoteheadsList
    ret = [note, beamLines]
    return ret           

                   
def GetCircle(segment):
    noteList = [] 
    note = Note(segment.BoundingBox)
    note.Noteheads.append(GetCoordinates(segment.Image,segment.BoundingBox,0,0))
    note.Hollow.append(IsHollow(segment.Image, segment.FilledImage))
    note.Type = 'circle'
    ret = [note, 0]
    return ret            
          
#*****************************************************************************************************************************#
#****************************************************Classif Functions***************************************************
path_to_dataset = [r'./segments/classify/new/']
outputPathFeatures ='./features.txt'
outputPathLabels= './labels.txt'
target_img_size = (32, 32) # fix image size because classification algorithms THAT WE WILL USE HERE expect that
def get_OCR_V(img,OCR_SIZE=(32,32)):
    newImg=np.copy((img))
    if(OCR_SIZE!=(-1,-1)):
        newImg=resize(newImg,OCR_SIZE)
    OCR_v = np.zeros(newImg.shape[1])

    for i in range(0,newImg.shape[1]):
        for j in range(0,newImg.shape[0]):
            OCR_v[i] = OCR_v[i] + newImg[j][i]
    return OCR_v


def get_OCR_H(img,OCR_SIZE=(32,32)):
    newImg=np.copy((img))
    if(OCR_SIZE!=(-1,-1)):
        newImg=resize(newImg,OCR_SIZE)
    
    OCR_h = np.zeros(newImg.shape[0])

    for i in range(0,newImg.shape[0]):
        for j in range(0,newImg.shape[1]):
            OCR_h[i] = OCR_h[i] + newImg[i][j]
    return OCR_h

def get_ratio(image):
    return image.shape[0]/lineSpace

def extract_segments_ones(img, target_img_size = (64,64)):
    imgCopied = resize(img, target_img_size)
    mx = np.amax(imgCopied)
    imgCopied = imgCopied / mx
    imgCopied = np.where(imgCopied > 0.5, 1, 0)
    arr = list()
    for i in range(0, 4):
        arr.append(np.sum(imgCopied[i * 8:(i + 1) * 8, i * 8:(i + 1) *8 ] == 1)/64)
    return arr

def readFeatures():
    df=pd.read_csv('featuresAndLabels.csv',index_col=0)
    features=df.iloc[:,:-1].values
    labels=df.iloc[:,-1].values
    return features,labels

def train_KNN(features, labels):
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(features, labels)
    KNeighborsClassifier(...)
    return neigh
#*****************************************************************************************************************************#
#****************************************************Output Functions**********************************************************
def getShape(lineGroups,arr,x,lineSpace,accidental,Type,Hollow,imgWidth):                    
    shape=""
    quad = getQuad(x,imgWidth)
    for lines in lineGroups[quad]:
        try:
            if(lines[0]-lineSpace-lineSpace/2-lineSpace/3 < arr < lines[4]+lineSpace+lineSpace/3):
                if(abs(lines[0]-lineSpace-lineSpace/2-arr)<lineSpace/3):
                    if len(accidental)!=0:
                        shape = " b" + accidental + "2"
                    else:
                        shape=" b2"
                elif(abs((lines[0]-lineSpace)-arr)<=lineSpace/3):
                    if len(accidental)!=0:
                        shape = " a" + accidental + "2"
                    else:
                        shape=" a2"
                elif(abs(lines[0]-lineSpace/2-arr)<=lineSpace/3):
                    if len(accidental)!=0:
                        shape = " g" + accidental + "2"
                    else:
                        shape=" g2"
                elif(abs(lines[0]-arr)<=lineSpace/3):
                    if len(accidental)!=0:
                        shape = " f" + accidental + "2"
                    else:
                        shape=" f2"
                elif(abs(lines[0]+lineSpace/2-arr)<=lineSpace/3):
                    if len(accidental)!=0:
                        shape = " e" + accidental + "2"
                    else:
                        shape=" e2"
                elif(abs(lines[1]-arr)<=lineSpace/3):
                    if len(accidental)!=0:
                        shape = " d" + accidental + "2"
                    else:
                        shape=" d2"
                elif(abs(lines[1]+lineSpace/2-arr)<=lineSpace/3):
                    if len(accidental)!=0:
                        shape = " c" + accidental + "2"
                    else:
                        shape=" c2"
                elif(abs(lines[2]-arr)<=lineSpace/3):
                    if len(accidental)!=0:
                        shape = " b" + accidental + "1"
                    else:
                        shape=" b1"
                elif(abs(lines[2]+lineSpace/2-arr)<=lineSpace/3):
                    if len(accidental)!=0:
                        shape = " a" + accidental + "1"
                    else:
                        shape=" a1"
                elif(abs(lines[3]-arr)<=lineSpace/3):
                    if len(accidental)!=0:
                        shape = " g" + accidental + "1"
                    else:
                        shape=" g1"
                elif(abs(lines[3]+lineSpace/2-arr)<=lineSpace/3):
                    if len(accidental)!=0:
                        shape = " f" + accidental + "1"
                    else:
                        shape=" f1"
                elif(abs(lines[4]-arr)<=lineSpace/3):
                    if len(accidental)!=0:
                        shape = " e" + accidental + "1"
                    else:
                        shape=" e1"
                elif(abs(lines[4]+lineSpace/2-arr)<=lineSpace/3):
                    if len(accidental)!=0:
                        shape = " d" + accidental + "1"
                    else:
                        shape=" d1"
                elif(abs(lines[4]+lineSpace-arr)<=lineSpace/3):
                    if len(accidental)!=0:
                        shape = " c" + accidental + "1"
                    else:
                        shape=" c1"
                if(Type=="single" and len(shape)!=0):
                    if(Hollow == True):
                        shape+="/2"
                    else:
                        shape+="/4"
                    return shape
                elif(Type=="circle" and len(shape)!=0):    
                    shape+="/1"
                    return shape
                if(Hollow == True and len(shape)!=0):
                    shape+="/2"
                elif(len(shape)!=0):
                    shape+="/4"
                return shape
        except:
            return ""
def getQuad(x,imgWidth):
    if x < imgWidth/4:
        quad = 0
    elif x < imgWidth/2:
        quad = 1
    elif x < 3*imgWidth/4:
        quad = 2
    else:
        quad = 3
    return quad

def sortAllNotes(notesList,timersList,classifList,dotsList,lineGroups,imgWidth):
    notesList=notesList+timersList+classifList+dotsList
    notesList.sort(key=lambda x: x[0].BoundingBox.miny)
    finalSorted = [[]]
    sortedLine = []
    counter = 0
    if len(lineGroups[0]) == 1:
        notesList.sort(key=lambda x: x[0].BoundingBox.minx)
        notesList = checkMetersSequence(notesList)
        finalSorted = finalSorted + [notesList]
    else:
        clearence = (lineGroups[0][1][0] - lineGroups[0][0][4]) / 2
        for note in notesList:
            quad = getQuad(note[0].BoundingBox.minx,imgWidth)
            try:
      
                if note[0].BoundingBox.maxy-note[0].BoundingBox.miny > 2*(lineGroups[quad][0][4] - lineGroups[quad][0][0]):
                    continue
                if (((note[0].BoundingBox.maxy-note[0].BoundingBox.miny)/2)+note[0].BoundingBox.miny) <= ((lineGroups[quad][counter][4])+clearence):
                    sortedLine.append(note)
                else:
                    sortedLine.sort(key=lambda x: x[0].BoundingBox.minx)
                    sortedLine = checkMetersSequence(sortedLine)
                    finalSorted.append(sortedLine.copy())
                    sortedLine.clear()
                    sortedLine.append(note)
                    counter += 1
            except:
                continue
        sortedLine.sort(key=lambda x: x[0].BoundingBox.minx)
        sortedLine = checkMetersSequence(sortedLine)
        finalSorted.append(sortedLine.copy())
    return finalSorted
        
def checkMetersSequence(sortedLine):
    noteHeads =[]
    if(sortedLine == []):return []
    if hasattr(sortedLine[0][0], 'Noteheads'):
        noteHeads = [val for sublist in sortedLine[0][0].Noteheads for val in sublist]
    if sortedLine[0][0].Type == "chord" and sortedLine[0][1] == 0 and 48<ord(sortedLine[1][0].Type[0])<58 and 48<ord(sortedLine[2][0].Type[0])<58:
        del sortedLine[0]
    elif sortedLine[0][0].Type == "chord" and len(noteHeads) >= 6:
        del sortedLine[0]
    if 48<ord(sortedLine[0][0].Type[0])<58 and len(sortedLine)>1 and 48<ord(sortedLine[1][0].Type[0])<58:
        if sortedLine[0][0].BoundingBox.miny > sortedLine[1][0].BoundingBox.miny:
            sortedLine[0],sortedLine[1] = sortedLine[1],sortedLine[0]
        if sortedLine[2][0].Type == "dot" and sortedLine[3][0].Type == "dot":
            del sortedLine[2]
            del sortedLine[2]
        elif sortedLine[2][0].Type == "dot":
            del sortedLine[2]
    return sortedLine

########################################################################################################
#Processess calling 
def imagePreproccessing(image_original):
    if(image_original.all()<=1):
        temp=np.copy(image_original*255)
    else:
        temp=np.copy(image_original)
    scanned=True
    ones=np.sum(temp[:,:] >= 200)
    zeros=np.sum(temp[:,:] <= 20)
    if(ones+zeros<image_original.shape[0]*image_original.shape[1]*0.9):
        scanned = False
        #Resize
        r1 = image_original.shape[0] / 1700
        r2 = image_original.shape[1] / 1700
        r=max(r1,r2)
        if(r > 1):
            image = resize(image_original, (image_original.shape[0]//r, image_original.shape[1]//r))
    
    image = np.copy(image_original)
    image,_,rotatedImage= preproccessing(image)
    #image,newScanned = crop(image,rotatedImage)
    return image,scanned,rotatedImage

def imageLineGroups(image,lineThickness, lineSpace):
    #Get Lines Groups
    imageCopy=np.copy(image)
    firstHalf=imageCopy[:,0:imageCopy.shape[1]//2]
    secondHalf=imageCopy[:,imageCopy.shape[1]//2:image.shape[1]]
    lineGroups1 = formLineGroups(firstHalf[:,0:firstHalf.shape[1]//2],lineThickness, lineSpace)
    lineGroups2 = formLineGroups(firstHalf[:,firstHalf.shape[1]//2:firstHalf.shape[1]],lineThickness, lineSpace)
    lineGroups3 = formLineGroups(secondHalf[:,0:secondHalf.shape[1]//2],lineThickness, lineSpace)
    lineGroups4 = formLineGroups(secondHalf[:,secondHalf.shape[1]//2:secondHalf.shape[1]],lineThickness, lineSpace)
    lineGroups=[lineGroups1,lineGroups2,lineGroups3,lineGroups4]
    checkAndFixLineGroups(lineGroups)
    return lineGroups

def removeStaffLines(image,scanned,lineThickness, lineSpace):
    #Staff line removal
    if(scanned):
        image= detectAndRemoveHorizontalLines(image,lineThickness,lineSpace )
    else:
        if(image.shape[0]<3100):
            rows=9
        elif(image.shape[0]<4800):
            rows=12
        horizontalStructure=np.ones((1,image.shape[1]//30))
        verticalStructure=np.ones((rows,1))
        horizontal = binary_erosion(image, horizontalStructure)
        horizontal = binary_dilation(horizontal, horizontalStructure)
        vertical = binary_erosion(image, verticalStructure)
        vertical = binary_dilation(vertical, verticalStructure)
        vertical = binary_erosion(vertical)
        vertical = binary_erosion(vertical)

        image=vertical
    return image
    
def getImageSegments(image):
    #structure element used for vertical opening to remove what remained from thick lines
    selement = [[0,1,0],
               [0,1,0],
               [0,0,0]]
    image=binary_erosion(image,selement)
    image = binary_dilation(image, selement)
    img_filled=np.copy(image)

    #find the contours for the image
    contours = find_contours(image, 0.9)

    #fill holes in contours 
    img_filled = FillPolygon(img_filled, contours)

    #Find the contours after filling, no hole contours exist
    contours = find_contours(img_filled,0.9)


    imagecollective, segments = SegmentImage(image, img_filled, contours)
    finalSegments = list()
    #The output of the image after segmentation
    #SaveSegments(segments)
    return segments,imagecollective

def divideSegments(segments,lineThickness, lineSpace):
    circlesSegments=list()
    classSegments= list()
    notesAndConnectedSegments=list()
    dots = list()
    for segment in segments:
        ratio = segment.Image.shape[0]/lineSpace
        asp_ratio =segment.Image.shape[0]/segment.Image.shape[1] 
        if(ratio <= 0.8):
            if(asp_ratio <= 1.5 and asp_ratio >= 0.5):
                dots.append(segment)
        if(ratio >=0.9 and ratio<=1.5):
            ones = np.sum(segment.Image == 1)
            zeros = np.sum(segment.Image == 0)
            if(ones/zeros < 1.29):
                circlesSegments.append(segment)
            else:
                classSegments.append(segment)
        elif(ratio >=1.8 and ratio<=3.8):
            classSegments.append(segment)
        elif(ratio >=4 ):
            notesAndConnectedSegments.append(segment)
    return circlesSegments, classSegments, notesAndConnectedSegments, dots

def classifySymbols(features2,labels,classSegments):
    neigh=train_KNN(features2,labels)

    classifList = [] 
    timersList = [] 

    for seg in classSegments:
        ii=np.copy(seg.Image)

        ftr = extract_segments_ones(ii)
        label=neigh.predict([ftr])[0]
        label = label.split('_')[0]

        symbol = Symbols(seg.BoundingBox)
        symbol.Center=GetCoordinates(seg.Image,seg.BoundingBox,0,0)
        symbol.Type = label 
        if(label.isnumeric()):     
            timersList.append([symbol,0])
        else:

            classifList.append([symbol,0])
    return classifList, timersList

def getDots(dots):
    dotsList = []
    for dot in dots:
        symbol = Symbols(dot.BoundingBox)
        symbol.Center=GetCoordinates(dot.Image,dot.BoundingBox,0,0)
        symbol.Type = 'dot' 
        dotsList.append([symbol,0])
    return dotsList
def getNotesList(notesAndConnectedSegments, newCircles,lineThickness, lineSpace,imagecollective,imageName):
    notesList = []
    for seg in notesAndConnectedSegments:
        note=GetNote(seg, lineThickness, lineSpace)
        if(note!=None):
            notesList.append(note)

    for seg in newCircles:
        note = GetCircle(seg)
        notesList.append(note)
    #uncomment for debugging (saves output image each notehead is marked by its type)    
    '''
    imgggg = np.copy(imagecollective)
    z = 1
    for note in notesList:
        if(note!=None):
            k = 0
            for head in note[0].Noteheads:
                x=head[0]
                y=head[1]
                rr,cc = rectangle(start = (y-1,x-10), end = (y+1,x+10) )
                if(note[0].Hollow[k] ==False):
                    imgggg[rr.astype(int),cc.astype(int)] = 0.1
                else:
                    imgggg[rr.astype(int),cc.astype(int)] = 0.4
                k+=1
    mpl.image.imsave(imageName+'.png', imgggg)
    '''
    return notesList

def writeToOutputFile(finalSorted, lineGroups,lineThickness, lineSpace,outputPath,image):
    f = open(outputPath, "w")
    if len(lineGroups[0]) != 1:
        f.write("{\n")
    for j,noteLine in enumerate(finalSorted[1:]):
        accidental = ""
        for i,Element in enumerate(noteLine):
            if i == 0:
                f.write("[")
            if 48<ord(Element[0].Type[0])<58:
                if i == 0:
                    f.write(" \meter<\"")
                    f.write(Element[0].Type[0])
                    f.write("/")
                elif i == 1:
                    f.write(Element[0].Type[0])
                    f.write("\">")
            elif Element[0].Type == "hash":
                accidental = "#"
            elif Element[0].Type == "b":
                accidental = "&"
            elif Element[0].Type == "bb":
                accidental = "&&"
            elif Element[0].Type == "x":
                accidental = "##"
            elif Element[0].Type == "e":
                accidental = ""
            elif Element[0].Type == "dot":
                if noteLine[i-1][0].Type == "single" and i+1<len(noteLine) and noteLine[i+1][0].Type == "dot":
                    if noteLine[i-1][1] != 0:
                        f.write("..")
                    elif 0 < Element[0].Center[0] - noteLine[i-1][0].BoundingBox.maxx  < 20:
                        f.write("..")
                elif noteLine[i-1][0].Type == "single":
                    if noteLine[i-1][1] != 0:
                        f.write(".")
                    elif 0 < Element[0].Center[0] - noteLine[i-1][0].BoundingBox.maxx  < 20:
                        f.write(".")
            else:
                if(len(Element[0].Noteheads)!=0):
                    arr = [val for sublist in Element[0].Noteheads for val in sublist]
                    if(len(arr)<=2 and Element[1]==0):
                        shape = getShape(lineGroups,arr[1],arr[0],lineSpace,accidental,Element[0].Type,Element[0].Hollow[0],image.shape[1])
                        accidental=""
                        if shape is not None:
                            if len(shape) != 0:
                                f.write(shape)
                    elif Element[1] == 0 and Element[0].Type == "chord":
                        accidental=""
                        shape = []
                        for k in range(0,len(arr)//2):
                            s = getShape(lineGroups,arr[2*k+1],arr[2*k],lineSpace,accidental,Element[0].Type,Element[0].Hollow[k],image.shape[1])
                            if s is not None:
                                shape.append(s)
                        if len(shape) != 0:
                            shape = sorted(shape)
                            f.write(" {")
                            for s in range(len(shape)):
                                f.write(shape[s])
                                if(s!=len(shape)-1):
                                    f.write(",")
                            f.write(" }")
                    elif Element[1]!=0 and Element[0].Type == "chord":
                        accidental=""
                        for k in range(0,len(arr)//2):
                            shape = getShape(lineGroups,arr[2*k+1],arr[2*k],lineSpace,accidental,Element[0].Type,Element[0].Hollow[k],image.shape[1])
                            if shape is not None:
                                if Element[1]==1:
                                    shape= shape[0:4] + "8"
                                elif Element[1]==2:
                                    shape= shape[0:4] + "16"
                                if len(shape) != 0:
                                    f.write(shape)
                    elif Element[1]!=0 and Element[0].Type == "single":
                        shape = getShape(lineGroups,arr[1],arr[0],lineSpace,accidental,Element[0].Type,Element[0].Hollow[0],image.shape[1])
                        if shape is not None:
                            if Element[1]==1:
                                shape= shape[0:4] + "8"
                            elif Element[1]==2:
                                shape= shape[0:4] + "16"
                            elif Element[1]==3:
                                shape= shape[0:4] + "32"
                            if len(shape) != 0:
                                f.write(shape)
        if j==len(lineGroups[0])-1:
            f.write(" ]\n") 
        elif (len(noteLine)!=0):
            f.write(" ],\n")
    if len(lineGroups[0]) != 1:
        f.write("}")
    f.close()