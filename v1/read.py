# External libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Local libraries

from utils import infit_outfit_statistics, show_img, get_circles_with_neighbors, detect_circles, img2mtx, estimate_rasch_model, visual_check

folder = 'scan'
students_folder = 'students'
prefix_img = 'IMG_20230722_00'

dict_areas ={
    'mat': {
        'name': 'Mat',
        'x1': 590,
        'x2': 800,
        'y1': 720,
        'y2': 1990,
        'answers': 25

    },
    'nat1': {
        'name': 'Nat1',
        'x1': 590,
        'x2': 800,
        'y1': 2110,
        'y2': 2770,
        'answers': 13
    },
    'nat2': {
        'name': 'Nat2',
        'x1': 1000,
        'x2':1210,
        'y1': 720,
        'y2': 1340,
        'answers': 12
    },
    'soc': {
        'name': 'Soc',
        'x1': 1000,
        'x2':1210,
        'y1': 1450,
        'y2': 2740,
        'answers': 25
    },
    'tex': {
        'name': 'Tex',
        'x1': 1420,
        'x2': 1640,
        'y1': 730,
        'y2': 1650,
        'answers': 18
    },
    'ima': {
        'name': 'Ima',
        'x1': 1840,
        'x2': 2050,
        'y1': 730,
        'y2': 1770,
        'answers': 20
        },
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

# Get right answers
right_answers = []


mat_irt = []
nat_irt = []
soc_irt = []
tex_irt = []
ima_irt = []

csv_file = "right.csv"
with open(csv_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    # Process each row of data
    for row in reader:
        right_answers.append(row[0][-1])

# Student outputs
csv_file = "s_answers.csv"
options = ['A', 'B', 'C', 'D']

for ending in range(1,33):

    student_folder = f'{students_folder}/{ending}'

    if not os.path.exists(student_folder):
        os.makedirs(student_folder)

    if ending < 10:
        img_name = f'{folder}/{prefix_img}0{ending}.png'
    else:
        img_name = f'{folder}/{prefix_img}{ending}.png'

    img = cv.imread(img_name, cv.IMREAD_COLOR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    ret, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)
    # histogram = cv.calcHist([gray], [0], None, [256], [0, 256])

    student_answers = []
    q_counter = 1
    # Get img for each area
    for area in dict_areas.values():
        crop = blur[area['y1']:area['y2'], area['x1']:area['x2']]
        circles, mask = detect_circles(img_name, crop, area['answers'])
        ans_mtrx = img2mtx(student_folder , area['name'], crop, circles, area['answers'])

        for i in range(area['answers']):
            pos = np.where(ans_mtrx[i])[0]
            if len(pos) == 1:
                is_right = right_answers[q_counter-1] == options[pos[0]]
                student_answers.append([q_counter, options[pos[0]], right_answers[q_counter-1], is_right])
                if q_counter < 26:
                    mat_irt.append((ending, q_counter, is_right))
                elif q_counter < 51:
                    nat_irt.append((ending, q_counter-25, is_right))
                elif q_counter < 76:
                    soc_irt.append((ending, q_counter-50, is_right))
                elif q_counter < 94:
                    tex_irt.append((ending, q_counter-75, is_right))
                else: 
                    ima_irt.append((ending, q_counter-93, is_right))
            else:
                student_answers.append([q_counter, 'X', right_answers[i], False])
                if q_counter < 26:
                    mat_irt.append((ending, q_counter, False))
                elif q_counter < 51:
                    nat_irt.append((ending, q_counter-25, False))
                elif q_counter < 76:
                    soc_irt.append((ending, q_counter-50, False))
                elif q_counter < 94:
                    tex_irt.append((ending, q_counter-75, False))
                else: 
                    ima_irt.append((ending, q_counter-93, False))

            q_counter += 1

    results = f"{student_folder}/results.csv"
    # Write the result to the CSV file
    with open(results, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(student_answers)

    # Create summary 
    summary_data = []
    for materia in materias_dict:
        [i, e] = materias_dict[materia]
        column = [row[3] for row in student_answers[i:e]]
        summary_data.append([materia, np.count_nonzero(column)])

    
    summary_data.append(['Nat', summary_data[1][1] + summary_data[2][1] + summary_data[3][1]])
    summary_data.append(['Soc', summary_data[4][1] + summary_data[5][1] + summary_data[6][1] + summary_data[7][1]])

    summary = f"{student_folder}/summary.csv"
    # Write the result to the CSV file
    with open(summary, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(summary_data)

irt = [mat_irt, nat_irt, soc_irt, tex_irt, ima_irt]
ans = [25, 25, 25, 18, 20]
    
for i, materia in enumerate(irt):
    theta_est, beta_est = estimate_rasch_model(materia, 32, ans[i])
    print("Estimated Individual Abilities (Theta):", theta_est)
    print("Estimated Item Parameters (Beta):", beta_est)
    visual_check(32, ans[i], theta_est, beta_est)
    item_fit_stats = infit_outfit_statistics(mat_irt, theta_est, beta_est)
    fitness_data = []
    for item_id, fit_stats in item_fit_stats.items():
        print(f"Item {item_id}: Infit = {fit_stats['Infit']:.2f}, Outfit = {fit_stats['Outfit']:.2f}")
        fitness_data.append([fit_stats['Infit'], fit_stats['Outfit']])
    s_report = f'reports/students_area_{i}.txt'
    np.savetxt(s_report, theta_est, delimiter=',', fmt='%.8f')
    q_report = f'reports/questions_area_{i}.txt'
    np.savetxt(q_report, beta_est, delimiter=',', fmt='%.8f')
    
    fitness = f'reports/fitness_{i}.csv'
    with open(fitness, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(fitness_data)
