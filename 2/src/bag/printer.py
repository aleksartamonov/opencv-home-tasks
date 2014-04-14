def print_to_arff(filename, descriptors, classes):

    f = open(filename, "w")
    K = len(descriptors[0].beans)
    f.write("@relation caltech.bagofwords"+"\n")
    for i in xrange(K):
        f.write("@attribute "+str(i)+"th numeric"+"\n")
    f.write("@attribute class {" + ','.join([str(i) for i in classes]) + "}"+"\n")
    f.write("@data"+"\n")
    for descriptor in descriptors:
        f.write(str(descriptor)+"\n")
    f.close()