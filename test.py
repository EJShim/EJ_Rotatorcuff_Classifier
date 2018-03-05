# import os
# try:
#     with open('./path_tmp', 'r') as text_file:
#         path = text_file.read().replace('\n', '')
# except:
#     with open('./path_tmp', 'w') as text_file:
#         print("~/", file=text_file)
import xlsxwriter

# Create an new Excel file and add a worksheet.
workbook = xlsxwriter.Workbook('demo.xlsx')
worksheet = workbook.add_worksheet()

# Widen the first column to make the text clearer.
worksheet.set_column('A:A', 20)

# Add a bold format to use to highlight cells.
bold = workbook.add_format({'bold': True})

# Write some simple text.
# worksheet.write('A1', 'Hello')

# # Text with formatting.
# worksheet.write('A2', 'World', bold)

# # Write some numbers, with row/column notation.
# worksheet.write(2, 0, 123)
# worksheet.write(3, 0, 123.456)

for i in range(0, 10):
    worksheet.write(i, 0, i*10)
# Insert an image.
#worksheet.insert_image('B5', 'logo.png')

#workbook.close()