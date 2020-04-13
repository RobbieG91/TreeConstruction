import os
import time
import subprocess

def normalize_height(path, filtered):
    filename = "{}{}{}".format(path, "/Data/Rotterdam/Machine Learning/", filtered)
    batFile = "{}{}".format(path, "/Data/Temp/TempBatHeight.bat")
    suffix = "_height_veg.laz"

    with open(batFile, 'w') as batWrite:
        batWrite.write("lasheight ")
        batWrite.write("-i \"{}\" ".format(filename))
        batWrite.write("-o \"{}{}\" ".format(filename[:-4],suffix))
        batWrite.write("-store_precise_as_extra_bytes ")
        batWrite.write("\n ")

    proc = subprocess.Popen(batFile, shell=True)
    proc.wait()

    return "{}{}".format(filename[:-4],suffix)

def classify_points(path, normalized):
    batFile = "{}{}".format(path, "/Data/Temp/TempBatClassify.bat")
    suffix = "_classified.laz"
    print "classfunc"
    print normalized

    with open(batFile, 'w') as batWrite:
        batWrite.write("lasclassify ")
        batWrite.write("-i \"{}\" ".format(normalized))
        batWrite.write("-o \"{}{}\" ".format(normalized[:-4],suffix))
        batWrite.write("-height_in_attribute 0 ")
        batWrite.write("-ground_offset 2.0 ") #default value = 2.0
        batWrite.write("-planar 0.1 ") #default value
        batWrite.write("-rugged 0.4 ") #default value
        batWrite.write("\n ")
    proc = subprocess.Popen(batFile, shell=True)
    proc.wait()

    return "{}{}".format(normalized[:-4], suffix)

def filter_veg(path, classified):
    batFile = "{}{}".format(path, "/Data/Temp/TempBatFilter.bat")
    suffix = "_ONLYveg.laz"
    print "filtfunc"
    print classified

    with open(batFile, 'w') as batWrite:
        batWrite.write("las2las ")
        batWrite.write("-i \"{}\" ".format(classified))
        batWrite.write("-o \"{}{}\" ".format(classified[:-4], suffix))
        batWrite.write("-keep_class 5 ")
        batWrite.write("\n ")
    proc = subprocess.Popen(batFile, shell=True)
    proc.wait()
    return "{}{}".format(classified[:-4], suffix)

def create_dem(path, filename):
    print filename
    suffix = "_DEM0_75.tif"

    batFile = "{}{}".format(path, "/Data/Temp/TempBatDEM.bat")
    with open(batFile, 'w') as batWrite:
        batWrite.write("lasgrid ")
        batWrite.write("-i \"{}\" ".format(filename))
        batWrite.write("-o \"{}{}\" ".format(filename[:-4], suffix))
        batWrite.write("-attribute 0 ")
        batWrite.write("-step 0.75 ")
        batWrite.write("-highest ")
        batWrite.write("\n ")
    proc = subprocess.Popen(batFile, shell=True)
    proc.wait()

if __name__ == '__main__':
    """GLOBAL VARIABLES"""
    path = os.getcwd()

    """The LAS/LAZ file that needs to be classified."""
    filename = "filename.laz"

    starttime = time.clock()

    """Active part of code"""
    print "normalizing"
    normalized = normalize_height(path, filename)
    print time.clock() - starttime
    print "classifying"
    classified = classify_points(path, normalized)
    print time.clock() - starttime

    print "filtering"
    filtered = filter_veg(path, classified)
    print time.clock() - starttime
    #
    print "creating grid"
    dem = create_dem(path, filtered)