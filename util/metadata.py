# **************************************************************************
# *
# * Authors:  J. M. de la Rosa Trevin (delarosatrevin@gmail.com)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# **************************************************************************

import sys
# from itertools import izip
from collections import OrderedDict
import copy


LABELS = {
    'rlnImageName': str,
    'rlnCubeSize':int,
    'rlnCropSize':int,
    'rlnSnrFalloff':float,
    'rlnDeconvStrength':float,
    'rlnPixelSize':float,
    'rlnDefocus':float,
    'rlnCorrectedTomoName':str,
    'rlnNumberSubtomo': int,
    'rlnMaskName': str,
    'rlnVoltage': float,
    'rlnDefocusU': float,
    'rlnDefocusV': float,
    'rlnDefocusAngle': float,
    'rlnSphericalAberration': float,
    'rlnDetectorPixelSize': float,
    'rlnCtfFigureOfMerit': float,
    'rlnMagnification': float,
    'rlnAmplitudeContrast': float,
    'rlnOriginalName': str,
    'rlnCtfImage': str,
    'rlnCoordinateX': float,
    'rlnCoordinateY': float,
    'rlnCoordinateZ': float,
    'rlnNormCorrection': float,
    'rlnMicrographName': str,
    'rlnGroupName': str,
    'rlnGroupNumber': str,
    'rlnOriginX': float,
    'rlnOriginY': float,
    'rlnAngleRot': float,
    'rlnAngleTilt': float,
    'rlnAnglePsi': float,
    'rlnClassNumber': int,
    'rlnLogLikeliContribution': float,
    'rlnRandomSubset': int,
    'rlnParticleName': str,
    'rlnOriginalParticleName': str,
    'rlnNrOfSignificantSamples': float,
    'rlnNrOfFrames': int,
    'rlnMaxValueProbDistribution': float,
    'rlnIndex': str,
    'rlnSubtomoIndex': str,
    'rlnMaskPercentile': float,
    'rlnMaskThreshold': float
}


class Label():
    def __init__(self, labelName):
        self.name = labelName
        # Get the type from the LABELS dict, assume str by default
        self.type = LABELS.get(labelName, str)

    def __str__(self):
        return self.name

    def __cmp__(self, other):
        return self.name == str(other)


class Item():
    """
    General class to store data from a row. (e.g. Particle, Micrograph, etc)
    """

    def copyValues(self, other, *labels):
        """
        Copy the values form other object.
        """
        for l in labels:
            setattr(self, l, getattr(other, l))

    def clone(self):
        return copy.deepcopy(self)


class MetaData():
    """ Class to parse Relion star files
    """
    def __init__(self, input_star=None):
        if input_star:
            self.read(input_star)
        else:
            self.clear()

    def clear(self):
        self._labels = OrderedDict()
        self._data = []

    def _setItemValue(self, item, label, value):
        setattr(item, label.name, label.type(value))

    def _addLabel(self, labelName):
        self._labels[labelName] = Label(labelName)

    def read(self, input_star):
        self.clear()
        found_label = False
        f = open(input_star)

        for line in f:
            values = line.strip().split()

            if not values: # empty lines
                continue

            if values[0].startswith('_rln'):  # Label line
                # Skip leading underscore in label name
                self._addLabel(labelName=values[0][1:])
                found_label = True

            elif found_label:  # Read data lines after at least one label
                # Iterate in pairs (zipping) over labels and values in the row
                item = Item()
                # Dynamically set values, using label type (str by default)
                for label, value in zip(self._labels.values(), values):
                    self._setItemValue(item, label, value)

                self._data.append(item)

        f.close()

    def _write(self, output_file):
        output_file.write("\ndata_\n\nloop_\n")
        line_format = ""

        # Write labels and prepare the line format for rows
        for i, l in enumerate(self._labels.values()):
            output_file.write("_%s #%d \n" % (l.name, i+1))
            # Retrieve the type of the label
            t = l.type
            if t is float:
                line_format += "%%(%s)f \t" % l.name
            elif t is int:
                line_format += "%%(%s)d \t" % l.name
            else:
                line_format += "%%(%s)s \t" % l.name

        line_format += '\n'

        for item in self._data:
            output_file.write(line_format % item.__dict__)

        output_file.write('\n')

    def write(self, output_star):
        output_file = open(output_star, 'w')
        self._write(output_file)
        output_file.close()

    def printStar(self):
        self._write(sys.stdout)

    def size(self):
        return len(self._data)

    def __len__(self):
        return self.size()

    def __iter__(self):
        for item in self._data:
            yield item

    def getLabels(self):
        return [l.name for l in self._labels.values()]

    def setLabels(self, **kwargs):
        """ Add (or set) labels with a given value. """
        for key, value in kwargs.iteritems():
            if key not in self._labels:
                self._addLabel(labelName=key)

        for item in self._data:
            for key, value in kwargs.iteritems():
                self._setItemValue(item, self._labels[key], value)

    def _iterLabels(self, labels):
        """ Just a small trick to accept normal lists or *args
        """
        for l1 in labels:
            if isinstance(l1, list):
                for l2 in l1:
                    yield l2
            else:
                yield l1

    def addLabels(self, *labels):
        """
        Register labes in the metadata, but not add the values to the rows
        """
        for l in self._iterLabels(labels):
            if l not in self._labels.keys():
                self._addLabel(l)

    def removeLabels(self, *labels):
        for l in self._iterLabels(labels):
            if l in self._labels:
                del self._labels[l]

    def addItem(self, item):
        """ Add a new item to the MetaData. """
        self._data.append(item)

    def setData(self, data):
        """ Set internal data with new items. """
        self._data = data

    def addData(self, data):
        """ Add new items to internal data. """
        for item in data:
            self.addItem(item)