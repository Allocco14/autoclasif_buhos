import csv
import numpy as np


img_name = 'IMG_15'
areas = ['Mat', 'Nat', 'Soc', 'Tex', 'Ima']

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

materias_dict = {
    'Mat': [0,24],
    'Bio': [25,32],
    'Qui': [33,40],
    'Fis': [41,49],
    'Geo': [50,54],
    'Uni': [55,59],
    'Col': [60,64],
    'Fil': [65,74],
    'Tex': [74,91],
    'Ima': [92,111]
}

reports_folder = 'reports'
students_folder = 'students'

for i in range(5):
    area_txt = f'{reports_folder}/students_area_{i}.txt'
    results = np.loadtxt(area_txt)
    mean = np.mean(results)
    std = np.std(results)
    print((results-mean)/std+10)

students_scores = []

for i in range(1,33):
    students_csv = f'{students_folder}/{i}/summary.csv'
    with open(students_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Process each row of data
        for row in reader:
            [a, s] = row
            if a in areas:
                students_scores.append([i, a, s])

global_results = {}

for a in areas:
    scores = np.array([int(x[2]) for x in students_scores if x[1] == a])
    mean = np.mean(scores)
    std = np.std(scores)
    global_results[a] = (scores-mean)/std+10
    print(global_results)

definite = []

for i in range(0,32):
    global_student = 0
    for a in areas:
        global_student += global_results[a][i]
    definite.append(global_student)

final = np.array(definite)
mean = np.mean(final)
std = np.std(final)

final = 100*((final-mean)/std)+500
global_results['Global'] = final

filename = 'globales.csv'

# Write the dictionary of lists to the CSV file
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row with keys as column names
    writer.writerow(global_results.keys())

    # Write the data rows
    writer.writerows(zip(*global_results.values()))