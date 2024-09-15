import csv

students_folder = 'students'

areas_dict = {
    'Mat': 25,
    'Bio': 8,
    'Qui': 8,
    'Fis': 9,
    'Geo': 5,
    'Uni': 5,
    'Col': 5,
    'Fil': 10,
    'Tex': 18,
    'Ima': 20
}

percents = []

for student in range(1,33):
    students_csv = f'{students_folder}/{student}/summary.csv'

    result = [student]
    with open(students_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Process each row of data
        for row in reader:
            [a, s] = row
            if a in areas_dict.keys():
                result.append(a)
                result.append(f'{100*int(s)/areas_dict[a]:.2f}%')
    
    percents.append(result)

print(percents)

filename = 'percents.csv'

# Write the dictionary of lists to the CSV file
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row with keys as column names
    writer.writerows(percents)