import numpy as np
import os
import time
import json
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import sys

def write_cityJSON(path, filename, lod, outfilename, param_filename, convex):
    tree_array = np.load("{}{}{}".format(path, "/Data/Temp/", filename))
    if param_filename:
        param_array = np.load("{}{}{}".format(path, "/Data/Temp/", param_filename))
    jsondict = {
        "type": "CityJSON",
        "version": "1.0"
    }
    jsondict['CityObjects'] = {}
    jsondict['vertices'] = []

    vcounter = 0

    jsondict['appearance'] = {
        "materials": [
            {
                "name": "TreeTrunk",
                "ambientIntensity": 0.2000,
                "diffuseColor": [0.6, 0.4, 0.1],
                "shininess": 0.2,
                "transparency": 0.0,
                "isSmooth": False
            },
            {
                "name": "GenericTreeCrown",
                "ambientIntensity": 0.2000,
                "diffuseColor": [0.41, 0.61, 0.35],
                "shininess": 0.2,
                "transparency": 0.0,
                "isSmooth": False
            },
            {
                "name": "Yellowish",
                "ambientIntensity": 0.2000,
                "diffuseColor": [0.85, 0.85, 0.56],
                "shininess": 0.2,
                "transparency": 0.0,
                "isSmooth": False
            },
            {
                "name": "MossyGreen",
                "ambientIntensity": 0.2000,
                "diffuseColor": [0.48, 0.54, 0.23],
                "shininess": 0.2,
                "transparency": 0.0,
                "isSmooth": False
            },
            {
                "name": "BlueishGreen",
                "ambientIntensity": 0.2000,
                "diffuseColor": [0.35, 0.40, 0.31],
                "shininess": 0.2,
                "transparency": 0.0,
                "isSmooth": False
            },
            {
                "name": "VomitGreen",
                "ambientIntensity": 0.2000,
                "diffuseColor": [0.58, 0.60, 0.38],
                "shininess": 0.2,
                "transparency": 0.0,
                "isSmooth": False
            },
            {
                "name": "Grayish",
                "ambientIntensity": 0.2000,
                "diffuseColor": [0.72, 0.75, 0.67],
                "shininess": 0.2,
                "transparency": 0.0,
                "isSmooth": False
            }

        ]
    }

    # Writer for LODs, 0, 1 and 2
    if lod != 3:
        for tree in tree_array:
            jsondict['CityObjects'][tree[0]] = {
                "type": "SolitaryVegetationObject",
            }

            starttime = time.clock()

            """Make vertices for tree model"""
            """Parameters"""

            x = np.float(tree[1])/1000.0
            y = np.float(tree[2])/1000.0
            z_b = np.float(tree[4])/1000.0
            z_c = np.float(tree[5])/1000.0
            z_p = np.float(tree[6])/1000.0
            r_p = np.float(tree[7])/1000.0
            z_pl = np.float(tree[8])/1000.0
            r_pl = np.float(tree[9])/1000.0
            z_ph = np.float(tree[10])/1000.0
            r_ph = np.float(tree[11])/1000.0
            z_t = np.float(tree[12])/1000.0
            t = math.radians(60)

            colorvalue = 1
            """Genera colors
            yellowtrees = ["Ailanthus", "Amelanchier", "Prunus"]
            mossytrees = ["Alnus", "Corylus", "Pyrus", "Robinia", "Styphnolobium"]
            blueishtrees = ["Fagus", "Catalpa"]
            vomittrees = ["Ginkgo", "Pinus"]
            grayishtrees = ["Liriodendron", "Malus", "Sorbus"]

            if tree[23] in yellowtrees:
                colorvalue = 2
            if tree[23] in mossytrees:
                colorvalue = 3
            if tree[23] in blueishtrees:
                colorvalue = 4
            if tree[23] in vomittrees:
                colorvalue = 5
            if tree[23] in grayishtrees:
                colorvalue = 6
            """

            if tree[23] == "Coniferae":
                colorvalue = 5

            """LOD0: Parametrized Hexagon"""
            if lod == 0:

                """Center, Base Height"""
                v1 = [x, y, z_b]

                """Periphery"""
                v2 = [x - r_p, y, z_b]
                v3 = [x - (math.cos(t) * r_p), y + (math.sin(t) * r_p), z_b]
                v4 = [x + (math.cos(t) * r_p), y + (math.sin(t) * r_p), z_b]
                v5 = [x + r_p, y, z_b]
                v6 = [x + (math.cos(t) * r_p), y - (math.sin(t) * r_p), z_b]
                v7 = [x - (math.cos(t) * r_p), y - (math.sin(t) * r_p), z_b]

                boundaries = []
                values = []

                """Make indices for tree model"""
                """Tree trunk base"""
                for i in range(1, 7):
                    if i != 6:
                        boundaries.append([[vcounter, vcounter + i + 1, vcounter + i]])
                        values.append(colorvalue)
                    else:
                        boundaries.append([[vcounter, vcounter + i - 5, vcounter + i]])
                        values.append(colorvalue)

                for i in range(7):
                    jsondict['vertices'].append(
                        list(np.around(eval("{}{}".format("v", i+1)), decimals=2))
                    )

                """If you want to see ALL features in final output, uncomment this"""
                """
                jsondict['CityObjects'][tree[0]]['attributes'] = {
                    "TreeType": tree[23],
                    "Classification_Certainty": tree[24],
                    "Point Count": tree[3],
                    "Height Crown Base": z_c,
                    "Periphery Height": z_p,
                    "Periphery Radius": r_p,
                    "Lower Periphery Height": z_pl,
                    "Lower Periphery Radius": r_pl,
                    "Higher Periphery Height": z_ph,
                    "Higher Periphery Radius": r_ph,
                    "Tree Top": z_t,
                    "Ratio Height High + Low": tree[13],
                    "Ratio Height High + Periphery": tree[14],
                    "Ratio Height Periphery + Low": tree[15],
                    "Ratio Radius High + Low": tree[16],
                    "Ratio Radius High + Periphery": tree[17],
                    "Ratio Radius Periphery + Low": tree[18],
                    "Ratio Height Tree Top + Crown Base": tree[19],
                    "Average Intensity": tree[20],
                    "Average Number of Returns": tree[21],
                    "Ratio Periphery Height + Periphery Radius": tree[22]
                }
                """


                jsondict['CityObjects'][tree[0]]['attributes'] = {
                    "TreeType": tree[23],
                    "Classification_Certainty": tree[24],
                    "Point Count": tree[3],
                    "Average Intensity": tree[20],
                    "Average Number of Returns": tree[21]
                }

                jsondict['CityObjects'][tree[0]]['geometry'] = [{
                    "type": "MultiSurface",
                    "lod": lod,
                    "boundaries": boundaries,
                    "material": {
                        "visual": {
                            "values": values
                        },
                    },
                }]

                vcounter += 7


            """LOD1: Parametrized Raised Hexagon"""
            if lod == 1:
                """Center, Base Height"""
                v1 = [x, y, z_b]

                """Lower Bound"""
                v2 = [x - r_p, y, z_b]
                v3 = [x - (math.cos(t) * r_p), y + (math.sin(t) * r_p), z_b]
                v4 = [x + (math.cos(t) * r_p), y + (math.sin(t) * r_p), z_b]
                v5 = [x + r_p, y, z_b]
                v6 = [x + (math.cos(t) * r_p), y - (math.sin(t) * r_p), z_b]
                v7 = [x - (math.cos(t) * r_p), y - (math.sin(t) * r_p), z_b]

                """Higher Bound"""
                v8 = [x - r_p, y, z_t]
                v9 = [x - (math.cos(t) * r_p), y + (math.sin(t) * r_p), z_t]
                v10 = [x + (math.cos(t) * r_p), y + (math.sin(t) * r_p), z_t]
                v11 = [x + r_p, y, z_t]
                v12 = [x + (math.cos(t) * r_p), y - (math.sin(t) * r_p), z_t]
                v13 = [x - (math.cos(t) * r_p), y - (math.sin(t) * r_p), z_t]

                """Treetop"""
                v14 = [x, y, z_t]

                boundaries = []
                values = []

                """Make indices for tree model"""
                """Raised Hexagon Ground"""
                for i in range(1, 7):
                    if i != 6:
                        boundaries.append([[vcounter, vcounter + i, vcounter + i + 1]])
                        values.append(colorvalue)
                    else:
                        boundaries.append([[vcounter, vcounter + i, vcounter + i - 5]])
                        values.append(colorvalue)

                """Raised Hexagon Sides"""
                for i in range(1, 7):
                    if i != 6:
                        boundaries.append([[vcounter + i, vcounter + i + 6, vcounter + i + 7, vcounter + i + 1]])
                        values.append(colorvalue)
                    else:
                        boundaries.append([[vcounter + i, vcounter + i + 6, vcounter + i + 1, vcounter + i - 5]])
                        values.append(colorvalue)

                """Raised Hexagon Treetop"""
                for i in range(7, 13):
                    if i != 12:
                        boundaries.append([[vcounter + 13, vcounter + i + 1, vcounter + i]])
                        values.append(colorvalue)
                    else:
                        boundaries.append([[vcounter + 13, vcounter + i - 5, vcounter + i]])
                        values.append(colorvalue)

                for i in range(14):
                    jsondict['vertices'].append(
                        list(np.around(eval("{}{}".format("v", i + 1)), decimals=2))
                    )

                """If you want to see ALL features in final output, uncomment this"""
                """
                jsondict['CityObjects'][tree[0]]['attributes'] = {
                    "TreeType": tree[23],
                    "Classification_Certainty": tree[24],
                    "Point Count": tree[3],
                    "Height Crown Base": z_c,
                    "Periphery Height": z_p,
                    "Periphery Radius": r_p,
                    "Lower Periphery Height": z_pl,
                    "Lower Periphery Radius": r_pl,
                    "Higher Periphery Height": z_ph,
                    "Higher Periphery Radius": r_ph,
                    "Tree Top": z_t,
                    "Ratio Height High + Low": tree[13],
                    "Ratio Height High + Periphery": tree[14],
                    "Ratio Height Periphery + Low": tree[15],
                    "Ratio Radius High + Low": tree[16],
                    "Ratio Radius High + Periphery": tree[17],
                    "Ratio Radius Periphery + Low": tree[18],
                    "Ratio Height Tree Top + Crown Base": tree[19],
                    "Average Intensity": tree[20],
                    "Average Number of Returns": tree[21],
                    "Ratio Periphery Height + Periphery Radius": tree[22]
                }
                """

                jsondict['CityObjects'][tree[0]]['attributes'] = {
                    "TreeType": tree[23],
                    "Classification_Certainty": tree[24],
                    "Point Count": tree[3],
                    "Average Intensity": tree[20],
                    "Average Number of Returns": tree[21]
                }

                jsondict['CityObjects'][tree[0]]['geometry'] = [{
                    "type": "MultiSurface",
                    "lod": lod,
                    "boundaries": boundaries,
                    "material": {
                        "visual": {
                            "values": values
                        },
                    },
                }]

                vcounter += 14

            """LOD2: Parametrized Tree Model"""
            if lod == 2:
                #trunk radius
                r_t = r_p/10
                """Center, Base Height"""
                v1 = [x, y, z_b]

                """Trunk, Ground"""
                v2 = [x - r_t, y, z_b]
                v3 = [x - (math.cos(t) * r_t), y + (math.sin(t) * r_t), z_b]
                v4 = [x + (math.cos(t) * r_t), y + (math.sin(t) * r_t), z_b]
                v5 = [x + r_t, y, z_b]
                v6 = [x + (math.cos(t) * r_t), y - (math.sin(t) * r_t), z_b]
                v7 = [x - (math.cos(t) * r_t), y - (math.sin(t) * r_t), z_b]

                """Crownbase"""
                v8 = [x - r_t, y, z_c]
                v9 = [x - (math.cos(t) * r_t), y + (math.sin(t) * r_t), z_c]
                v10 = [x + (math.cos(t) * r_t), y + (math.sin(t) * r_t), z_c]
                v11 = [x + r_t, y, z_c]
                v12 = [x + (math.cos(t) * r_t), y - (math.sin(t) * r_t), z_c]
                v13 = [x - (math.cos(t) * r_t), y - (math.sin(t) * r_t), z_c]

                """Periphery lower bound"""
                v14 = [x - r_pl, y, z_pl]
                v15 = [x - (math.cos(t) * r_pl), y + (math.sin(t) * r_pl), z_pl]
                v16 = [x + (math.cos(t) * r_pl), y + (math.sin(t) * r_pl), z_pl]
                v17 = [x + r_pl, y, z_pl]
                v18 = [x + (math.cos(t) * r_pl), y - (math.sin(t) * r_pl), z_pl]
                v19 = [x - (math.cos(t) * r_pl), y - (math.sin(t) * r_pl), z_pl]

                """Periphery"""
                v20 = [x - r_p, y, z_p]
                v21 = [x - (math.cos(t) * r_p), y + (math.sin(t) * r_p), z_p]
                v22 = [x + (math.cos(t) * r_p), y + (math.sin(t) * r_p), z_p]
                v23 = [x + r_p, y, z_p]
                v24 = [x + (math.cos(t) * r_p), y - (math.sin(t) * r_p), z_p]
                v25 = [x - (math.cos(t) * r_p), y - (math.sin(t) * r_p), z_p]

                """Periphery higher bound"""
                v26 = [x - r_ph, y, z_ph]
                v27 = [x - (math.cos(t) * r_ph), y + (math.sin(t) * r_ph), z_ph]
                v28 = [x + (math.cos(t) * r_ph), y + (math.sin(t) * r_ph), z_ph]
                v29 = [x + r_ph, y, z_ph]
                v30 = [x + (math.cos(t) * r_ph), y - (math.sin(t) * r_ph), z_ph]
                v31 = [x - (math.cos(t) * r_ph), y - (math.sin(t) * r_ph), z_ph]

                """Treetop"""
                v32 = [x, y, z_t]

                for i in range(32):
                    jsondict['vertices'].append(
                        list(np.around(eval("{}{}".format("v", i+1)), decimals=2))
                    )

                boundaries = []
                values = []

                """Make indices for tree model"""
                """Tree trunk base"""
                for i in range(1, 7):
                    if i != 6:
                        boundaries.append([[vcounter, vcounter + i, vcounter + i + 1]])
                        values.append(0)
                    else:
                        boundaries.append([[vcounter, vcounter + i, vcounter + i - 5]])
                        values.append(0)
                """Tree trunk"""
                for i in range(1, 7):
                    if i != 6:
                        boundaries.append([[vcounter + i, vcounter + i + 6, vcounter + i + 7, vcounter + i + 1]])
                        values.append(0)
                    else:
                        boundaries.append([[vcounter + i, vcounter + i + 6, vcounter + i + 1, vcounter + i - 5]])
                        values.append(0)

                """Lower periphery"""
                for i in range(7, 13):
                    if i != 12:
                        boundaries.append([[vcounter + i, vcounter + i + 6, vcounter + i + 7, vcounter + i + 1]])
                        values.append(colorvalue)
                    else:
                        boundaries.append([[vcounter + i, vcounter + i + 6, vcounter + i + 1, vcounter + i - 5]])
                        values.append(colorvalue)

                """Periphery"""
                for i in range(13, 19):
                    if i != 18:
                        boundaries.append([[vcounter + i, vcounter + i + 6, vcounter + i + 7, vcounter + i + 1]])
                        values.append(colorvalue)
                    else:
                        boundaries.append([[vcounter + i, vcounter + i + 6, vcounter + i + 1, vcounter + i - 5]])
                        values.append(colorvalue)

                """Higher periphery"""
                for i in range(19, 25):
                    if i != 24:
                        boundaries.append([[vcounter + i, vcounter + i + 6, vcounter + i + 7, vcounter + i + 1]])
                        values.append(colorvalue)
                    else:
                        boundaries.append([[vcounter + i, vcounter + i + 6, vcounter + i + 1, vcounter + i - 5]])
                        values.append(colorvalue)
                """Tree top"""
                for i in range(25, 31):
                    if i != 30:
                        boundaries.append([[vcounter + 31, vcounter + i + 1, vcounter + i]])
                        values.append(colorvalue)
                    else:
                        boundaries.append([[vcounter + 31, vcounter + i - 5, vcounter + i]])
                        values.append(colorvalue)

                """If you want to see ALL features in final output, uncomment this"""
                """
                jsondict['CityObjects'][tree[0]]['attributes'] = {
                    "TreeType": tree[23],
                    "Classification_Certainty": tree[24],
                    "Point Count": tree[3],
                    "Height Crown Base": z_c,
                    "Periphery Height": z_p,
                    "Periphery Radius": r_p,
                    "Lower Periphery Height": z_pl,
                    "Lower Periphery Radius": r_pl,
                    "Higher Periphery Height": z_ph,
                    "Higher Periphery Radius": r_ph,
                    "Tree Top": z_t,
                    "Ratio Height High + Low": tree[13],
                    "Ratio Height High + Periphery": tree[14],
                    "Ratio Height Periphery + Low": tree[15],
                    "Ratio Radius High + Low": tree[16],
                    "Ratio Radius High + Periphery": tree[17],
                    "Ratio Radius Periphery + Low": tree[18],
                    "Ratio Height Tree Top + Crown Base": tree[19],
                    "Average Intensity": tree[20],
                    "Average Number of Returns": tree[21],
                    "Ratio Periphery Height + Periphery Radius": tree[22]
                }
                """

                jsondict['CityObjects'][tree[0]]['attributes'] = {
                    "TreeType": tree[23],
                    "Classification_Certainty": tree[24],
                    "Point Count": tree[3],
                    "Average Intensity": tree[20],
                    "Average Number of Returns": tree[21]
                }

                jsondict['CityObjects'][tree[0]]['geometry'] = [{
                    "type": "MultiSurface",
                    "lod": lod,
                    "boundaries": boundaries,
                    "material": {
                        "visual": {
                            "values": values
                        },
                    },
                }]

                vcounter += 32

        with open("{}{}{}".format(path, "/Data/Output/", outfilename), 'w') as json_file:
            json.dump(jsondict, json_file, indent=2)

        print time.clock() - starttime
    # Writer for LODs 3.0 and 3.1
    if lod == 3:
        starttime = time.clock()
        skipped_trees = 0

        for seg_id in np.unique(tree_array['point']['segment_id']):
            # print seg_id
            rule = tree_array['point']['segment_id'] == seg_id
            index_param = np.where(np.array(param_array[:, 0], dtype=float) == seg_id)[0]

            if not index_param:
                continue

            tree = tree_array[rule]
            tree_param = np.array(param_array[index_param][0, :23], dtype=float)
            tree_type = param_array[index_param][0, 23]
            type_certainty = param_array[index_param][0, 24]

            if len(tree) < 50:
                continue

            colorvalue = 1

            """ Genera materials.
            yellowtrees = ["Ailanthus", "Amelanchier", "Prunus"]
            mossytrees = ["Alnus", "Corylus", "Pyrus", "Robinia", "Styphnolobium"]
            blueishtrees = ["Fagus", "Catalpa"]
            vomittrees = ["Ginkgo", "Pinus"]
            grayishtrees = ["Liriodendron", "Malus", "Sorbus"]

            if tree_type in yellowtrees:
                colorvalue = 2
            if tree_type in mossytrees:
                colorvalue = 3
            if tree_type in blueishtrees:
                colorvalue = 4
            if tree_type in vomittrees:
                colorvalue = 5
            if tree_type in grayishtrees:
                colorvalue = 6
            """

            if tree_type == "Coniferae":
                colorvalue = 5

            x = tree['point']['X']/1000.0
            y = tree['point']['Y']/1000.0
            z = tree['point']['height_above_ground']/1000.0

            Z_real = tree['point']['Z']/1000.0
            fit_3dfier = int(np.average(z - Z_real))
            z = z - fit_3dfier

            pos = np.array((x, y, z)).T

            """Alpha shapes"""
            if not convex:
                convex = False
                alpha = 0.5

                xnorm = MinMaxScaler().fit_transform(x.reshape(-1, 1))
                ynorm = MinMaxScaler().fit_transform(y.reshape(-1, 1))
                znorm = MinMaxScaler().fit_transform(z.reshape(-1, 1))
                posnorm = np.array((xnorm, ynorm, znorm)).T[0]
                tetra = Delaunay(posnorm)

                # Find radius of the circumsphere.
                # By definition, radius of the sphere fitting inside the tetrahedral needs
                # to be smaller than alpha value

                tetrapos = np.take(posnorm, tetra.vertices, axis=0)
                normsq = np.sum(tetrapos ** 2, axis=2)[:, :, None]
                ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))
                a = np.linalg.det(np.concatenate((tetrapos, ones), axis=2))
                Dx = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [1, 2]], ones), axis=2))
                Dy = -np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 2]], ones), axis=2))
                Dz = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 1]], ones), axis=2))
                c = np.linalg.det(np.concatenate((normsq, tetrapos), axis=2))

                r = np.sqrt((Dx ** 2 + Dy ** 2 + Dz ** 2) - (4 * a * c)) / (2 * np.abs(a))

                # Find tetrahedrals
                tetras = tetra.vertices[r < alpha, :]

                # triangles
                TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
                Triangles = tetras[:, TriComb].reshape(-1, 3)
                Triangles = np.sort(Triangles, axis=1)

                # Remove triangles that occur twice, because they are within shapes
                TrianglesDict = defaultdict(int)
                for tri in Triangles:
                    TrianglesDict[tuple(tri)] += 1
                Triangles = np.array([tri for tri in TrianglesDict if TrianglesDict[tri] == 1])
                # edges
                EdgeComb = np.array([(0, 1), (0, 2), (1, 2)])

                Edges = Triangles[:, EdgeComb].reshape(-1, 2)
                Edges = np.sort(Edges, axis=1)
                Edges = np.unique(Edges, axis=0)

                Vertices = np.unique(Edges)


                verts = np.hstack((np.where(Triangles[0][0] == Vertices)[0],
                                   np.where(Triangles[0][1] == Vertices)[0],
                                   np.where(Triangles[0][2] == Vertices)[0]))

                for triangle in Triangles[1:]:
                    verts = np.vstack((verts, np.hstack((np.where(triangle[0] == Vertices)[0],
                                                         np.where(triangle[1] == Vertices)[0],
                                                         np.where(triangle[2] == Vertices)[0]))))
                unsorted_triangles = verts
                vertices = pos[Vertices]
                boundaries = []
                values = []

            """Convex Hulls"""
            if convex:
                convexpoints = ConvexHull(pos).vertices
                convdelly = Delaunay(pos[convexpoints])
                unsorted_triangles = convdelly.convex_hull
                vertices = pos[convexpoints]
                boundaries = []
                values = []

            """Find half-edges between triangles in convex hull and make faces only facing outward"""
            starttriangle = unsorted_triangles[0]
            neighborlist = []
            boundary_array = None

            sorted_triangles = sort_triangles(unsorted_triangles, starttriangle, neighborlist, boundary_array)
            if type(sorted_triangles) == bool:
                # Case: One or more triangles in the mesh has more than 3 neighboring triangles, this shouldn't happen.
                skipped_trees += 1
                continue

            ccw_sorted_triangles = ccw_orientation(sorted_triangles, vertices)
            ccw_sorted_triangles = ccw_sorted_triangles + vcounter

            for boundary in ccw_sorted_triangles:
                boundaries.append([boundary.tolist()])
                values.append(colorvalue)

            vcounter += len(vertices)

            """Add tree trunk"""

            x = tree_param[1]/1000.0
            y = tree_param[2]/1000.0
            z_b = tree_param[4]/1000.0
            z_c = tree_param[5]/1000.0
            r_p = tree_param[7]/1000.0
            t = math.radians(60)
            r_t = r_p / 10

            """Center, Ground"""
            v1 = [x, y, z_b]

            """Trunk, Ground"""
            v2 = [x - r_t, y, z_b]
            v3 = [x - (math.cos(t) * r_t), y + (math.sin(t) * r_t), z_b]
            v4 = [x + (math.cos(t) * r_t), y + (math.sin(t) * r_t), z_b]
            v5 = [x + r_t, y, z_b]
            v6 = [x + (math.cos(t) * r_t), y - (math.sin(t) * r_t), z_b]
            v7 = [x - (math.cos(t) * r_t), y - (math.sin(t) * r_t), z_b]

            """Trunktop"""
            v8 = [x - r_t, y, z_c]
            v9 = [x - (math.cos(t) * r_t), y + (math.sin(t) * r_t), z_c]
            v10 = [x + (math.cos(t) * r_t), y + (math.sin(t) * r_t), z_c]
            v11 = [x + r_t, y, z_c]
            v12 = [x + (math.cos(t) * r_t), y - (math.sin(t) * r_t), z_c]
            v13 = [x - (math.cos(t) * r_t), y - (math.sin(t) * r_t), z_c]

            """Center, Trunktop"""
            v14 = [x, y, z_c]

            """Tree Trunk Boundaries"""
            boundaries2 = []
            values2 = []

            """Make indices for implicit tree trunk model"""
            """Raised Hexagon Ground"""
            for i in range(1, 7):
                if i != 6:
                    boundaries2.append([[vcounter, vcounter + i, vcounter + i + 1]])
                    values2.append(0)
                else:
                    boundaries2.append([[vcounter, vcounter + i, vcounter + i - 5]])
                    values2.append(0)

            """Raised Hexagon Sides"""
            for i in range(1, 7):
                if i != 6:
                    boundaries2.append([[vcounter + i, vcounter + i + 6, vcounter + i + 7, vcounter + i + 1]])
                    values2.append(0)
                else:
                    boundaries2.append([[vcounter + i, vcounter + i + 6, vcounter + i + 1, vcounter + i - 5]])
                    values2.append(0)

            """Raised Hexagon Treetop"""
            for i in range(7, 13):
                if i != 12:
                    boundaries2.append([[vcounter + 13, vcounter + i + 1, vcounter + i]])
                    values2.append(0)
                else:
                    boundaries2.append([[vcounter + 13, vcounter + i - 5, vcounter + i]])
                    values2.append(0)

            vcounter += 14
            boundaries.extend(boundaries2)
            values.extend(values2)

            jsondict['CityObjects'][str(seg_id)] = {
                "type": "SolitaryVegetationObject",
            }

            jsondict['CityObjects'][str(seg_id)]['attributes'] = {
                "TreeType": tree_type,
                "Classification_Certainty": type_certainty,
                "Point Count": tree_param[3],
                "Average Intensity": np.around(tree_param[20], decimals=2),
                "Average Number of Returns": np.around(tree_param[21], decimals=2)
            }

            jsondict['CityObjects'][str(seg_id)]['geometry'] = [{
                "type": "MultiSurface",
                "lod": lod,
                "boundaries": boundaries,
                "material": {
                    "visual": {
                        "values": values
                    },
                },
            }]

            vertices = np.around(vertices, decimals=2).tolist()

            """Vertices for Alpha-Shapes or ConvexHulls"""
            for i in vertices:
                jsondict['vertices'].append(i)

            """Vertices for TreeTrunks"""
            for i in range(14):
                jsondict['vertices'].append(
                    list(np.around(eval("{}{}".format("v", i + 1)), decimals=2))
                )

        # prettyprint
        with open("{}{}{}".format(path, "/Data/Output/", outfilename), 'w') as json_file:
            json.dump(jsondict, json_file, indent=2)

        # compact
        # with open("{}{}{}".format(path, "/Data/Output/FinalOutput/", outfilename), 'w') as json_file:
        #     json.dump(jsondict, json_file)

        print "This many trees were not included due to a too low alpha value:", skipped_trees
        print time.clock() - starttime

def sort_triangles(unsorted_triangles, starttriangle, neighborlist, boundary_array, boundary_array_cond=False):
    if len(neighborlist) > 0:
        neighborlist = neighborlist[1:]

    i = unsorted_triangles[:, 0]
    j = unsorted_triangles[:, 1]
    k = unsorted_triangles[:, 2]

    l = starttriangle[0]
    m = starttriangle[1]
    n = starttriangle[2]

    neighbors = unsorted_triangles[np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(
        np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(
            np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(

                np.logical_and(np.logical_and(l == i, m == j), n != k),
                np.logical_and(np.logical_and(l == i, m != j), n == k)),
                np.logical_and(np.logical_and(l != i, m == j), n == k)),

                np.logical_and(np.logical_and(l == i, m == k), n != j)),
                np.logical_and(np.logical_and(l == i, m != k), n == j)),
                np.logical_and(np.logical_and(l != i, m == k), n == j)),

            np.logical_and(np.logical_and(l == j, m == k), n != i)),
            np.logical_and(np.logical_and(l == j, m != k), n == i)),
            np.logical_and(np.logical_and(l != j, m == k), n == i)),

            np.logical_and(np.logical_and(l == j, m == i), n != k)),
            np.logical_and(np.logical_and(l == j, m != i), n == k)),
            np.logical_and(np.logical_and(l != j, m == i), n == k)),

        np.logical_and(np.logical_and(l == k, m == i), n != j)),
        np.logical_and(np.logical_and(l == k, m != i), n == j)),
        np.logical_and(np.logical_and(l != k, m == i), n == j)),

        np.logical_and(np.logical_and(l == k, m == j), n != i)),
        np.logical_and(np.logical_and(l == k, m != j), n == i)),
        np.logical_and(np.logical_and(l != k, m == j), n == i)
    )]

    if boundary_array_cond == False:
        unsorted_triangles = np.delete(unsorted_triangles, np.where((starttriangle == unsorted_triangles).all(axis=1).any())[0], 0)
        boundary_array = np.array(starttriangle)
        boundary_array_cond = True

    if len(neighbors) > 3:
        return False
    for neighbor in neighbors:
        if np.ndim(boundary_array) == 2:
            ax_var = 1
        else:
            ax_var = None
        if not (starttriangle == boundary_array).all(axis=ax_var).any():
            boundary_array = np.vstack((boundary_array, starttriangle))
        if not (neighbor == boundary_array).all(axis=ax_var).any():
            if (np.flip(neighbor) == boundary_array).all(axis=ax_var).any():
                continue
            o, p, q = starttriangle[0], starttriangle[1], starttriangle[2]
            r, s, t = neighbor[0], neighbor[1], neighbor[2]
            if np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(
                    np.logical_and(np.logical_and(o == r, p == s), q != t),
                    np.logical_and(np.logical_and(o == r, p != s), q == t)),
                    np.logical_and(np.logical_and(o != r, p == s), q == t)),

                    np.logical_and(np.logical_and(o == s, p == t), q != r)),
                    np.logical_and(np.logical_and(o == s, p != t), q == r)),
                    np.logical_and(np.logical_and(o != s, p == t), q == r)),

                    np.logical_and(np.logical_and(o == t, p == r), q != s)),
                    np.logical_and(np.logical_and(o == t, p != r), q == s)),
                    np.logical_and(np.logical_and(o != t, p == r), q == s)):
                boundary_array = np.vstack((boundary_array, np.flip(neighbor)))
                neighborlist.append(np.flip(neighbor))
            else:
                boundary_array = np.vstack((boundary_array, neighbor))
                neighborlist.append(neighbor)

    if len(neighborlist) > 0:
        next_starttriangle = neighborlist[0]
        return sort_triangles(unsorted_triangles, next_starttriangle, neighborlist, boundary_array, boundary_array_cond)
    else:
        sorted_triangles = boundary_array
        return sorted_triangles

def ccw_orientation(boundaries, vertices):
    triangles = vertices[boundaries]
    maxdex = np.argmax(np.average(triangles[:, :, 2], axis=1))
    bottom_triangle = triangles[maxdex]

    v0 = bottom_triangle[0]
    v1 = bottom_triangle[1]
    v2 = bottom_triangle[2]

    normal = np.cross(v1 - v0, v2 - v1)
    if normal[2] > 0:
        return boundaries
    else:
        return np.flip(boundaries)


if __name__ == '__main__':
    """GLOBAL VARIABLES"""
    path = os.getcwd()
    filename = "Noordereiland_NewFeatures_ML.npy"
    param_filename = False
    lod = 3
    convex = True


    sys.setrecursionlimit(10000)
    if lod == 0:
        outfilename = "Noordereiland_Classified_lod0.json"
    if lod == 1:
        outfilename = "Noordereiland_Classified_lod1.json"
    if lod == 2:
        outfilename = "Noordereiland_Classified_lod2.json"
    if lod == 3 and convex:
        filename = "Noordereiland_2std_100_Int_AllFeatures_Full.npy"
        param_filename = "Noordereiland_NewFeatures_ML.npy"
        outfilename = "Noordereiland_Classified_lod3C.json"
    if lod == 3 and not convex:
        filename = "Noordereiland_2std_100_Int_AllFeatures_Full.npy"
        param_filename = "Noordereiland_NewFeatures_ML.npy"
        outfilename = "Noordereiland_Classified_lod3A.json"
    else:
        pass

    write_cityJSON(path, filename, lod, outfilename, param_filename, convex)
