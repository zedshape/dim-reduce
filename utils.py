import csv
import sys

# Here we follow Allen's original abbreviation form
# e = matches
# s = left-matches
# f = right-matches
# c = contains
# o = overlaps
# m = meets
# b = follows

R = {"e", "s", "f", "c", "o", "m", "b"}

# Database is a collection of e-sequences
class Database:
    def __init__(self, database):
        self.sequences = []
        self.initialSupport = {}
        self.frequentSecondElements = set()

        for id, eSeqList in enumerate(database):
            newSeq = EventSequence(id)
            eventList = newSeq.processAdding(eSeqList)
            self.sequences.append(newSeq)

            # Add initial support
            for label in eventList:
                if label not in self.initialSupport:
                    self.initialSupport[label] = 0
                self.initialSupport[label] += 1
        
        # sort might be helpful
        for eSeq in self.sequences:
            eSeq.sequences.sort()

    def remove(self):
        for idx, seq in enumerate(self.sequences):
            for iidx, evt in enumerate(seq.sequences):
                if evt.label not in self.initialSupport.keys():
                    del self.sequences[idx].sequences[iidx]

    # print function
    def __str__(self):
        rst = []
        for i, eSeq in enumerate(self.sequences):
            rst.append(format("eSeq %d : %s" % (i, eSeq.__str__())))
        return "\n".join(rst)


# Event-interval sequence (e-sequence)
class EventSequence:
    # init function will receive list-parsed sequence and change it into our own structure
    def __init__(self, id):
        self.id = id
        self.sequences = []  # order of event

    def processAdding(self, eSeqList):
        eventList = set()
        for event in eSeqList:
            newInterval = Interval(event[0], event[1], event[2])
            self.sequences.append(newInterval)
            eventList.add(newInterval.label)
        self.sequences = sorted(self.sequences)
        return eventList

    def __repr__(self):
        rst = []
        for event in self.sequences:
            rst.append(event.__str__())
        return "(" + ", ".join(rst) + ")"


# Interval is a triplet composed of stat time, end time, and label
# it will follow lexicographical rule
class Interval:
    def __init__(self, label, start, end):
        self.label = label
        self.start = start
        self.end = end

    # get the whole duration of each event
    def getDuration(self):
        return self.end - self.start

    # for python-supported hash function (for set operations)
    def __hash__(self):
        return hash((self.label, self.start, self.end))

    def __repr__(self):
        return format("(%s, %d, %d)" % (self.label, self.start, self.end))

    # built-in comparing function (for ordering)
    # our ordering is based on start -> end -> label
    def __lt__(self, other):
        if self.start == other.start:
            if self.end == other.end:
                return self.label < other.label
            else:
                return self.end < other.end
        else:
            return self.start < other.start


# e = equals
# l = left-contains
# r = right-contains
# c = contains
# o = overlaps
# m = meets
# f = follows

def getRelation(A, B):
    relation = None

    if B[0] - A[0] == 0:
        if B[1] - A[1] == 0:
            relation = 1 #equal
        else:
            relation = 2 #overlap
    # guarantees: B.start > A.start
    # else:
        # if B.start == A.end:
            # relation = "o"
        # elif B.start - A.end == 0:
        #     relation = "m"
    elif B[0] - A[1] >= 0:
        relation = 3 #follow

    return relation


# function for preprocessing data into the shape that the algorithm takes
def preprocess(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        your_list = list(reader)

    distinct_events = set()
    new_list = []
    final_list = []
    timelength = {}
    max_index = 0
    for i in your_list:
        new_list.append(i[0].split(" "))

    for i in new_list:
        max_index = max(int(i[0]), max_index)

    for i in range(max_index + 1):
        final_list.append([])

    for i in new_list:
        final_list[int(i[0])].append((str(i[1]), int(i[2]), int(i[3])))
        distinct_events.add(str(i[1]))
        if int(i[0]) not in timelength:
            timelength[int(i[0])] = 0
        timelength[int(i[0])] = max(timelength[int(i[0])], int(i[3]))

    tseq = len(final_list)
    tdis = len(distinct_events)
    tintv = len(new_list)
    aintv = len(new_list) / len(final_list)
    avgtime = sum(timelength.values()) / len(timelength.keys())

    return tseq, tdis, tintv, aintv, avgtime, final_list

def getEventIntervalSequences(z):
    return z[0]


# export F structure to csv file
def exportDisprop(dataname, FL1, FL2, n1, n2, constraints):
    filename = (
        "Disprop"
        + "_"
        + dataname
        + "_"
        + str(constraints["minSupPercent"])
        + "_"
        + str(constraints["epsilon"])
        + "_"
        + str(constraints["gap"])
        + "_"
        + str(constraints["timeoutseconds"])
        + ".csv"
    )
    F = []
    for k in FL1:
        for E in FL1[k]:
            for R in FL1[k][E]:
                if E not in FL2[k] or R not in FL2[k][E]:
                    F.append({"events": E, "relations": R, "freq1": len(FL1[k][E][R]), "freq2": 0,
                    "relSup1": len(FL1[k][E][R])/n1, "relSup2": 0,
                    "disprop": ((len(FL1[k][E][R])+1)/n1)/(1/n2)})
                else:
                    F.append({"events": E, "relations": R, "freq1": len(FL1[k][E][R]), "freq2": len(FL2[k][E][R]),
                    "relSup1": len(FL1[k][E][R])/n1, "relSup2": len(FL2[k][E][R])/n2,
                    "disprop": ((len(FL1[k][E][R])+1)/n1)/((len(FL2[k][E][R])+1)/n2)})

    with open(filename, "w") as output_file:
        dict_writer = csv.DictWriter(output_file, ["events", "relations", "freq1", "freq2", "relSup1", "relSup2", "disprop"])
        dict_writer.writeheader()
        dict_writer.writerows(F)
    return filename

# export F structure to csv file
def exportF(dataname, FL, constraints):
    filename = (
        "F"
        + "_"
        + dataname
        + "_"
        + str(constraints["minSupPercent"])
        + "_"
        + str(constraints["epsilon"])
        + "_"
        + str(constraints["gap"])
        + "_"
        + str(constraints["timeoutseconds"])
        + ".csv"
    )
    F = []
    for k in FL:
        for E in FL[k]:
            for R in FL[k][E]:
                F.append({"events": E, "relations": R, "frequency": len(FL[k][E][R])})
    with open(filename, "w") as output_file:
        dict_writer = csv.DictWriter(output_file, ["events", "relations", "frequency"])
        dict_writer.writeheader()
        dict_writer.writerows(F)
    return filename


# export L structure to csv file
def exportL(dataname, FL, constraints):
    filename = (
        "L"
        + "_"
        + dataname
        + "_"
        + str(constraints["minSupPercent"])
        + "_"
        + str(constraints["epsilon"])
        + "_"
        + str(constraints["gap"])
        + "_"
        + str(constraints["timeoutseconds"])
        + ".csv"
    )
    L = []

    for k in FL:
        for E in FL[k]:
            for R in FL[k][E]:
                if k == 2:
                    for S in FL[k][E][R]:
                        for s1 in FL[k][E][R][S]:
                            for s2 in FL[k][E][R][S][s1]:
                                location = s1.__repr__() + ", " + s2.__repr__()
                                L.append(
                                    {
                                        "events": E,
                                        "relations": R,
                                        "frequency": len(FL[k][E][R]),
                                        "e-sequence": S,
                                        "intervals": location,
                                    }
                                )
                else:
                    for S in FL[k][E][R]:
                        for z in FL[k][E][R][S]:
                            location = ""
                            for si in getEventIntervalSequences(z):
                                location += si.__repr__()
                                location += ","
                            location = location[:-1]
                            L.append(
                                {
                                    "events": E,
                                    "relations": R,
                                    "frequency": len(FL[k][E][R]),
                                    "e-sequence": S,
                                    "intervals": location,
                                }
                            )

    with open(filename, "w") as output_file:
        dict_writer = csv.DictWriter(
            output_file, ["events", "relations", "frequency", "e-sequence", "intervals"]
        )
        dict_writer.writeheader()
        dict_writer.writerows(L)
    return filename
