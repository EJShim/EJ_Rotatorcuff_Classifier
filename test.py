import gdcm
import numpy


dataset = None

class DataElement(object):
    def __init__(self, swig_element, name, value):
        self._swig_element = swig_element

        self.name = name
        self.value = value
        self.VR = str(swig_element.GetVR()).strip()
        self.VL = str(swig_element.GetVL()).strip()

        tg = swig_element.GetTag()

        self.tag = {
            "group": hex(int(tg.GetGroup())),
            "element": hex(int(tg.GetElement())),
            "str": str(swig_element.GetTag()).strip()
        }

    def __repr__(self):
        return "<DataElement {0} {1}>".format(self.name, self.tag('str'))

    def __str__(self):
        return str(self.name)

def walk(fn):
    if dataset == None:
        return

    if not hasattr(fn, "__call__"):
        raise TypeError("wal kdataset requires a function as its parameters")


    iterator = dataset.GetDES().begin()


    while not iterator.equal(dataset.GetDES().end()):
        data_element = iterator.next()
        yield fn(data_element)




if __name__ == "__main__":
    import sys
    r = gdcm.ImageReader()
    r.SetFileName("/home/ej/data/RCT/RCT/1/ser001img00001.dcm"  )
    if not r.Read():
        print("cannot read")
        sys.exit(1)

    _file = r.GetFile()
    _str_filter = gdcm.StringFilter()
    _str_filter.SetFile(_file)

    dataset = _file.GetDataSet()


    def ds(data_element):
        value = _str_filter.ToStringPair(data_element.GetTag())
        if value[1]:
            return DataElement(data_element, value[0].strip(), value[1].strip())

    results = [data for data in walk(ds) if data is not None]

    print(results)
