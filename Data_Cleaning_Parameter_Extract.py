from laspy.file import File
import numpy as np
import os
import time
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import sys
import ML_TreeType_Classifier
from sklearn.cluster import DBSCAN

# To view full arrays
np.set_printoptions(threshold=sys.maxsize)

""" 
    Data cleaning and parameter extraction script, needs a pre-segmented PC containing only vegetation data.
    Filters segments, removes planes, removes outliers
    Plotting of intermediate results are commented out, but remain in the code for easy access.
"""

def read_trees(path, filename, outfilename):

    readtime = time.clock()
    inFile = File("{}{}{}".format(path, "/Data/Rotterdam/", filename), mode='r')

    print "{}{:0.2f}{}{}".format("LAZ reading time: ", time.clock() - readtime, " sec. Number of points in file: ", inFile.__len__())

    treelist = []
    treelistfull = []

    for seg_id in np.unique(inFile.segment_id):
        for k in range(1):
            segtime = time.clock()
            if seg_id == 0:
                continue
            "Converting to smaller np array so filtering through the entire pointcloud is no longer needed in the following steps"
            rule = inFile.segment_id == seg_id
            seg_array = inFile.points[rule]

            "How many points is truly a tree? This implementation uses 50 points as minimum"
            seg_point_count = len(seg_array)
            if seg_point_count < 50:
                continue

            X = seg_array['point']['X']
            Y = seg_array['point']['Y']
            Z = seg_array['point']['height_above_ground']
            Z_real = seg_array['point']['Z']
            fit_3dfier = int(np.average(Z - Z_real))

            # fig = plt.figure()
            # plt.clf()
            # ax = Axes3D(fig)
            # ax.scatter(X,Y, Z, color = 'yellowgreen')
            # plt.show()

            avg_inliers = int(len(seg_array)/100.0*55.0)

            ransac = linear_model.RANSACRegressor(
                linear_model.LinearRegression(),
                stop_n_inliers=avg_inliers,
                min_samples=3
            )
            ransac.fit(np.array((X, Y)).T, Z)

            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)
            a = np.array((X[inlier_mask], Y[inlier_mask], Z[inlier_mask])).T
            b = np.array((X[inlier_mask], Y[inlier_mask], ransac.predict(np.array((X[inlier_mask], Y[inlier_mask])).T))).T
            avg_distance = np.average(np.sqrt(np.sum((a - b) ** 2, axis=1)))

            # fig = plt.figure()
            # plt.clf()
            # ax = Axes3D(fig)
            # ax.scatter(X[inlier_mask], Y[inlier_mask], Z[inlier_mask], color='yellowgreen')
            # ax.scatter(X[outlier_mask], Y[outlier_mask], Z[outlier_mask], color='gold')

            # plt.show()

            """Skip if segment consists of a single plane"""

            if avg_distance < 100:
                continue

            seg_indexes = np.where(inFile.segment_id == seg_id)[0]
            return_array = inFile.num_returns[seg_indexes]

            return_array, seg_array, X, Y, Z = clean_ransac(return_array, seg_array, X, Y, Z)
            if len(return_array) == 0:
                # This condition is met when either the avg. intensity is too high or the avg. nr of returns is too low.
                continue

            # fig = plt.figure()
            # plt.clf()
            # ax = Axes3D(fig)
            # ax.scatter(X, Y, Z, color='yellowgreen')
            # plt.show()

            if len(X) == 0:
                continue

            seg_array, X, Y, Z = remove_distant_outliers(seg_array, X, Y, Z)
            if len(X) == 0:
                continue


            # fig = plt.figure()
            # plt.clf()
            # ax = Axes3D(fig)
            # ax.scatter(X, Y, Z, color='yellowgreen')
            # plt.show()

            # ax.scatter(X, Y, Z, color='gold')
            # plt.show()

            if len(seg_array) < 10:
                continue

            """ Here comes new function, parameter extract"""
            """ Here comes new function, parameter extract"""
            """ Here comes new function, parameter extract"""
            """ Here comes new function, parameter extract"""
            """ Here comes new function, parameter extract"""
            Z = Z - fit_3dfier

            max_height = np.max(Z)
            min_height = np.min(Z)

            """Tree Top: Will be the maximum height or 99th height percentile"""
            tree_top = np.percentile(Z, 99)
            if tree_top/1000.0 > 50:
                continue

            """Tree Base: Zero, as height is calculated to be height from ground.
            This value should later be recalculated to original Z-value."""
            tree_base = 0 - fit_3dfier

            """Tree Crown Base: Will be the 1st or 5th height percentile"""
            tree_crown_base = np.percentile(Z, 5)

            """Periphery Point: Will be the height interval where most points are located"""
            """Lower and Higher Periphery: Halfway points between Periphery Point and Tree Crown Base and Treetop 
            respectively"""
            division = np.linspace(min_height, max_height, num=11)
            countlst = []

            for i in range(len(division)-1):
                division_perc = ((i+1)*10)-5.0
                space = np.logical_and(Z >= division[i], Z < division[i+1])

                # When a division has no points, move on to the next division
                div_point_count = (len(seg_array['point'][space]))
                if div_point_count == 0:
                    continue

                center = np.average(X[space]), np.average(Y[space])
                coords = np.vstack((X[space], Y[space])).transpose()
                division_height = (division[i]+division[i+1])/2

                radius = np.percentile(np.sqrt(np.sum((coords - center)**2, axis=1)), 99)
                templst = [div_point_count, division_height, division_perc, center, radius, division[i], division[i+1]]
                countlst.append(templst)
            periphery = max(countlst)
            periphery_lower = (periphery[1] + tree_crown_base) / 2.0
            periphery_higher = (periphery[1] + tree_top) / 2.0

            for div in countlst:
                if div[5] <= periphery_lower <= div[6]:
                    radius_lower = div[4]
                    if div[0] < 5:
                        radius_lower = False
                if div[5] <= periphery_higher <= div[6]:
                    radius_higher = div[4]
                    if div[0] < 5:
                        radius_higher = False
            if type(radius_higher) == bool or type(radius_lower) == bool:
                continue

            """Features for tree types ratios between periphery heights, radii and tree top vs crown base."""

            height_ratio_hi_lo = periphery_higher / periphery_lower
            height_ratio_hi_per = periphery_higher / periphery[1]
            height_ratio_per_lo = periphery[1] / periphery_lower

            radius_ratio_hi_lo = radius_higher / radius_lower
            radius_ratio_hi_per = radius_higher / periphery[4]
            radius_ratio_per_lo = periphery[4] / radius_lower

            top_base_ratio = tree_top / tree_crown_base
            per_height_per_radius_ratio = periphery[1] / periphery[4]

            tree = [seg_id, periphery[3][0], periphery[3][1], seg_point_count, tree_base, tree_crown_base, periphery[1],
                    periphery[4], periphery_lower, radius_lower, periphery_higher, radius_higher, tree_top,
                    height_ratio_hi_lo, height_ratio_hi_per, height_ratio_per_lo, radius_ratio_hi_lo,
                    radius_ratio_hi_per, radius_ratio_per_lo, top_base_ratio,
                    np.average(seg_array['point']['intensity']), np.average(return_array), per_height_per_radius_ratio]

            treelist.append(tree)
            treelistfull.append(seg_array)

    tree_array = treelist[0]
    for tree in treelist[1:]:
        tree_array = np.vstack((tree_array, tree))

    full_tree_array = treelistfull[0]
    for fulltree in treelistfull[1:]:
        full_tree_array = np.concatenate((full_tree_array, fulltree), axis=0)

    outfilename_2 = "{}{}".format(outfilename, "_Full")
    np.save("{}{}{}".format(path, "/Data/Temp/", outfilename), tree_array)
    np.save("{}{}{}".format(path, "/Data/Temp/", outfilename_2), full_tree_array)

    return tree_array

def clean_ransac(return_array, seg_array, X, Y, Z, swap=False):
    """"    Check if a part of the segment is a plane, and remove the plane from the segment.
            Check the points that have a low number of returns.
            If a plane fits well around these points, remove these points from the initial array
            Try again until the initial array has no planes to remove, and is thus cleaned
            No planes or an acceptable number of outliers.
    """
    clean_indexes = np.where(return_array == 1)[0]
    seg_array_low_nr = seg_array[clean_indexes]

    if np.average(seg_array['point']['intensity']) > 100:
        print "high intensity", np.unique(seg_array['point']['segment_id']), np.average(seg_array['point']['intensity'])
        return [], [], [], [], []
    if np.average(return_array) <= 1.5:
        print "low returns", np.unique(seg_array['point']['segment_id']), np.average(return_array), len(return_array)
        return [], [], [], [], []
    if len(seg_array_low_nr) > 10:
        X_P = seg_array_low_nr['point']['X']
        Y_P = seg_array_low_nr['point']['Y']
        Z_P = seg_array_low_nr['point']['height_above_ground']

        if swap:
            Z_P = seg_array_low_nr['point']['Y']
            Y_P = seg_array_low_nr['point']['height_above_ground']
            Z = seg_array['point']['Y']
            Y = seg_array['point']['height_above_ground']

        ransac2 = linear_model.RANSACRegressor(
            linear_model.LinearRegression(),
            min_samples=3,
        )
        ransac2.fit(np.array((X_P, Y_P)).T, Z_P)

        inlier_mask2 = ransac2.inlier_mask_
        outlier_mask2 = np.logical_not(inlier_mask2)

        X_P_I = X_P[inlier_mask2]
        Y_P_I = Y_P[inlier_mask2]
        Z_P_I = Z_P[inlier_mask2]

        # inliers X,Y,Zs
        inliers = np.array((X_P_I, Y_P_I, Z_P_I)).T

        # predictions X,Y, predicted Zs
        inlier_predictions = np.array((X_P_I, Y_P_I, ransac2.predict(np.array((X_P_I, Y_P_I)).T))).T

        # if the predictions have a decent distance from inlier values, it's a good fit
        avg_distance = np.average(np.sqrt(np.sum((inliers - inlier_predictions) ** 2, axis=1)))

        # then we predict values with the complete tree, the points that lie close enough to
        # the inlier plane should be removed. This needs to be done because otherwise we would only
        # remove points from the low nr cloud, which is only used to generate the fitting plane.
        predictions = np.array((X, Y, ransac2.predict(np.array((X, Y)).T))).T
        allsamples = np.array((X, Y, Z)).T

        distances = np.sqrt(np.sum(np.square(allsamples - predictions), axis=1))
        distances2 = []

        for point in allsamples:
            mindist = np.min(np.sqrt(np.sum(np.square(inliers - point, dtype=np.int64), axis=1)))
            distances2.append(mindist)
        distances2 = np.array(distances2)

        # fig = plt.figure()
        # plt.clf()
        # ax = Axes3D(fig)
        # ax.scatter(X_P_I, Y_P_I, Z_P_I, color='yellowgreen')
        # ax.scatter(X_P[outlier_mask2], Y_P[outlier_mask2], Z_P[outlier_mask2], color='gold')
        # ax.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2], color='green')
        # ax.scatter(X, Y, Z, color='gold')
        # plt.show()

        if avg_distance < 100:
            deletelist = []
            for x_p, y_p, z_p in np.nditer((X_P_I, Y_P_I, Z_P_I)):
                xi, yi, zi = np.where(X == x_p)[0], np.where(Y == y_p)[0], np.where(Z == z_p)[0]
                deletelist.append(np.intersect1d(xi, np.intersect1d(yi, zi))[0])

            stuff = np.where((distances < 750) & (distances2 < 2000))[0]
            mask = np.logical_not(np.isin(stuff, deletelist))

            deletelist.extend(stuff[mask])

            # fig = plt.figure()
            # plt.clf()
            # ax = Axes3D(fig)
            # ax.scatter(X[deletelist], Y[deletelist], Z[deletelist], color='red')

            seg_array = np.delete(seg_array, deletelist)
            return_array = np.delete(return_array, deletelist)

            X = seg_array['point']['X']
            Y = seg_array['point']['Y']
            Z = seg_array['point']['height_above_ground']

            # ax.scatter(X, Y, Z, color='yellowgreen')
            # plt.show()

            return clean_ransac(return_array, seg_array, X, Y, Z)

        if avg_distance >= 100 and swap is False:
            return clean_ransac(return_array, seg_array, X, Y, Z, swap=True)
        else:
            if swap:
                return return_array, seg_array, X, Z, Y
            else:
                return return_array, seg_array, X, Y, Z
    if 0 < len(seg_array_low_nr) < 10:
        """If there's that little points with these questionable properties, just remove them all together, 
        don't even bother checking for a plane of less than 10 points, these are likely outliers"""
        X_P = seg_array_low_nr['point']['X']
        Y_P = seg_array_low_nr['point']['Y']
        Z_P = seg_array_low_nr['point']['height_above_ground']

        deletelist = []
        for x_p, y_p, z_p in np.nditer((X_P, Y_P, Z_P)):
            xi, yi, zi = np.where(X == x_p)[0], np.where(Y == y_p)[0], np.where(Z == z_p)[0]
            deletelist.append(np.intersect1d(xi, np.intersect1d(yi, zi))[0])
        seg_array = np.delete(seg_array, deletelist)
        return_array = np.delete(return_array, deletelist)
        X = seg_array['point']['X']
        Y = seg_array['point']['Y']
        Z = seg_array['point']['height_above_ground']
        if swap:
            return return_array, seg_array, X, Z, Y
        else:
            return return_array, seg_array, X, Y, Z
    else:
        if swap:
            return return_array, seg_array, X, Z, Y
        else:
            return return_array, seg_array, X, Y, Z

def remove_distant_outliers(seg_array, X, Y, Z):
    coords = np.array((X, Y, Z)).T
    coordsnorm = StandardScaler().fit_transform(coords)

    db = DBSCAN(eps=0.50, min_samples=50).fit(coordsnorm)
    labels = db.labels_
    nrpoints = len(seg_array)
    remove = []

    """Remove Noise"""
    if np.any(labels == -1):
        remove = np.where(labels == -1)[0]
        if len(remove) > (0.05 * nrpoints) or len(remove) == 0:
            db = DBSCAN(eps=0.75, min_samples=50).fit(coordsnorm)
            labels = db.labels_
            if np.any(labels == -1):
                remove = np.where(labels == -1)[0]
                if len(remove) > (0.05 * nrpoints) or len(remove) == 0:
                    db = DBSCAN(eps=1.0, min_samples=50).fit(coordsnorm)
                    labels = db.labels_
                    if np.any(labels == -1):
                        remove = np.where(labels == -1)[0]
                        if len(remove) > (0.05 * nrpoints) or len(remove) == 0:
                            db = DBSCAN(eps=1.5, min_samples=50).fit(coordsnorm)
                            labels = db.labels_
                            if np.any(labels == -1):
                                remove = np.where(labels == -1)[0]
                                if len(remove) > (0.05 * nrpoints) or len(remove) == 0:
                                    db = DBSCAN(eps=2.0, min_samples=50).fit(coordsnorm)
                                    labels = db.labels_
                                    if np.any(labels == -1):
                                        remove = np.where(labels == -1)[0]

    """In case of multiple clusters -> Keep only the largest cluster."""
    if len(np.unique(labels)) > 2:
        clustercount = []
        for i in np.unique(labels):
            clustercount.append((len(np.where(labels == i)[0]), i))
        keeplabel = max(clustercount)[1]
        remove = np.where(labels != keeplabel)[0]

        # remove2 = np.hstack((remove, extra))
        # Create second remove if you want to plot clusters getting removed separately

    if len(remove) > (0.05 * nrpoints):
        remove = []

    # fig = plt.figure()
    # plt.clf()
    # ax = Axes3D(fig)
    # ax.scatter(X[remove], Y[remove], Z[remove], color='red')

    # ax.scatter(X[remove2], Y[remove2], Z[remove2], color='red')
    # ax.scatter(X[extra], Y[extra], Z[extra], color='orange')

    if len(remove) > 0:
        seg_array = np.delete(seg_array, remove)
        X = np.delete(X, remove)
        Y = np.delete(Y, remove)
        Z = np.delete(Z, remove)

    # ax.scatter(X, Y, Z, color='yellowgreen')
    # plt.show()

    if len(remove) > 0 and len(seg_array) > 0:
        return remove_distant_outliers(seg_array, X, Y, Z)
    else:
        return seg_array, X, Y, Z

if __name__ == '__main__':
    """GLOBAL VARIABLES"""
    path = os.getcwd()
    filename = "Noordereiland_height_veg_classified_ONLYveg_segmented_DEM0_75_T1_40.laz"
    outfilename = "Noordereiland_2std_100_Int_AllFeatures"

    starttime = time.clock()

    read_trees(path, filename, outfilename)
    print "{}{:0.2f}{}".format("Total Calculation time: ", time.clock() - starttime, " sec.")