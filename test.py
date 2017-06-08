import gdcm
import numpy

def get_gdcm_to_numpy_typemap():
    """Returns the GDCM Pixel Format to numpy array type mapping."""
    _gdcm_np = {gdcm.PixelFormat.UINT8  :numpy.int8,
                gdcm.PixelFormat.INT8   :numpy.uint8,
                #gdcm.PixelFormat.UINT12 :numpy.uint12,
                #gdcm.PixelFormat.INT12  :numpy.int12,
                gdcm.PixelFormat.UINT16 :numpy.uint16,
                gdcm.PixelFormat.INT16  :numpy.int16,
                gdcm.PixelFormat.UINT32 :numpy.uint32,
                gdcm.PixelFormat.INT32  :numpy.int32,
                #gdcm.PixelFormat.FLOAT16:numpy.float16,
                gdcm.PixelFormat.FLOAT32:numpy.float32,
                gdcm.PixelFormat.FLOAT64:numpy.float64 }
    return _gdcm_np

def get_numpy_array_type(gdcm_pixel_format):
    """Returns a numpy array typecode given a GDCM Pixel Format."""
    return get_gdcm_to_numpy_typemap()[gdcm_pixel_format]

def gdcm_to_numpy(image):
    """Converts a GDCM image to a numpy array.
    """
    pf = image.GetPixelFormat()

    assert pf.GetScalarType() in get_gdcm_to_numpy_typemap().keys(), \
           "Unsupported array type %s"%pf

    shape = image.GetDimension(0) * image.GetDimension(1), pf.GetSamplesPerPixel()
    if image.GetNumberOfDimensions() == 3:
      shape = shape[0] * image.GetDimension(2), shape[1]

    dtype = get_numpy_array_type(pf.GetScalarType())
    print("Dtype : ", dtype)

    gdcm_array = image.GetBuffer()
    gdcm_array = str(gdcm_array, 'utf-8')

    result = numpy.fromstring(gdcm_array, dtype=dtype)
    result.shape = shape
    return result

if __name__ == "__main__":
  import sys
  r = gdcm.ImageReader()
  r.SetFileName("/home/ej/data/RCT/RCT/1/ser001img00001.dcm"  )
  if not r.Read():
    sys.exit(1)

  numpy_array = gdcm_to_numpy( r.GetImage() )
  print (numpy_array)\

# import numpy as np
#
# a = np.array(list(b'hello world'))
#
# print(a)
