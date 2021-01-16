from Utility import *
import argparse

#####################################################################################################

def pipline(image_original,imageName='01',outputPath='./output.txt'):
    try:
        image,scanned,rotatedImage = imagePreproccessing(image_original)
    except:
        pass
    
#####################################################################################################
    try:
        lineThickness, lineSpace=getLineThicknessSpacing_FullImage(image)
    except:
        pass
#####################################################################################################
    try:
        lineGroups=imageLineGroups(image,lineThickness, lineSpace)
    except:
        pass
#####################################################################################################
    #Staff line removal
    try:
        image = removeStaffLines(image,scanned,lineThickness, lineSpace)
    except:
        pass
#####################################################################################################
    #Segmentation
    try:
        segments,imagecollective = getImageSegments(image)
    except:
        pass
#####################################################################################################
    #Segments types division 
    try:
        circlesSegments, classSegments, notesAndConnectedSegments, dots = divideSegments(segments,lineThickness, lineSpace)
    except:
        pass
#####################################################################################################
    #Merge half circles 
    try:
        newCircles = MergeHalfCircles(circlesSegments)
    except:
        pass
####################################################################################################
    try:
        features,labels=readFeatures()
    except:
        pass

    try:
        classifList, timersList = classifySymbols(features,labels,classSegments)
        dotsList = getDots(dots)
        notesList = getNotesList(notesAndConnectedSegments, newCircles,lineThickness, lineSpace,imagecollective,imageName)
    except:
        pass
    try:
        finalSorted = sortAllNotes(notesList,timersList,classifList,dotsList,lineGroups,image.shape[1])    
    except:
        pass
    
    try:    
        writeToOutputFile(finalSorted, lineGroups,lineThickness, lineSpace,outputPath,image)
    except:
        f = open(outputPath, "w")
        f.write("{\n")
        f.write("[ ]\n")
        f.write("}\n")
        pass
#########################################################################################################
if __name__ == "__main__":

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfolder", help = "Input File")
    parser.add_argument("outputfolder", help = "Output File")
    args = parser.parse_args()

    for filename in os.listdir(args.inputfolder):
          
        imageName = filename.split('.')[0]
        inputPath = os.path.join(args.inputfolder, filename)
        try:
            img =io.imread(inputPath)
            img = rgb2gray(img)

            outputPath = os.path.join(args.outputfolder, imageName+".txt")
            pipline(img,imageName,outputPath)
        except:
            pass