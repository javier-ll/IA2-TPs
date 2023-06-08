import csv

with open('orders.txt', 'r') as txt_file, open('orders.csv', 'w', newline='') as csv_file:
    txt_reader = csv.reader(txt_file, delimiter='|')
    csv_writer = csv.writer(csv_file)
    
    for row in txt_reader:
        csv_writer.writerow(row)
    
    csv_writer.writerow('$')