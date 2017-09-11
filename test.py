import gdcm
import mudicom

path = "/Users/EJ/Desktop/ser004img00008.dcm"

reader = gdcm.Reader()
reader.SetFileName(path)
if not reader.Read():
    raise InvalidDicom("Not a valid DICOM file")


_file = reader.GetFile()
_header = _file.GetHeader()
_dataset = _file.GetDataSet()
_str_filter = gdcm.StringFilter()


print(_header)
print(_dataset)


# mu = mudicom.load(path)

# # returns array of data elements as dicts
# mu.read()


# # returns dict of errors and warnings for DICOM
# mu.validate()

# # basic anonymization
# mu.anonymize()
# # save anonymization
# mu.save_as("dicom.dcm")

# # creates image object
# img = mu.image # before v0.1.0 this was mu.image()
# # returns numpy array
# img.numpy # before v0.1.0 this was mu.numpy()

# # using Pillow, saves DICOM image
# img.save_as_pil("ex1.jpg")
# # using matplotlib, saves DICOM image
# img.save_as_plt("ex1_2.jpg")