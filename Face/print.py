import win32print
import win32ui
from PIL import Image, ImageWin

PHYSICALWIDTH = 110
PHYSICALHEIGHT = 111

printer_name = win32print.GetDefaultPrinter ()
file_name = "merged_image.jpg"

hDC = win32ui.CreateDC ()
hDC.CreatePrinterDC (printer_name)
printer_size = hDC.GetDeviceCaps (PHYSICALWIDTH), hDC.GetDeviceCaps (PHYSICALHEIGHT)

bmp = Image.open (file_name)
if bmp.size[0] < bmp.size[1]:
  bmp = bmp.rotate (0)

hDC.StartDoc (file_name)
hDC.StartPage ()

dib = ImageWin.Dib (bmp)
dib.draw (hDC.GetHandleOutput (), (int(printer_size[1]/3)-500, 0, 2*int(printer_size[1]/3)-500,printer_size[1]))

hDC.EndPage ()
hDC.EndDoc ()