import cv2 as cv
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def show_img(name, img, scale=5):
    shape = img.shape
    new_width = int(shape[1]/scale)
    new_height = int(shape[0]/scale)
    resized_img = cv.resize(img, (new_width, new_height))
    cv.imshow(name, resized_img)
    cv.waitKey(1000)
    cv.destroyAllWindows()

def get_circles_with_neighbors(circles):
    list_index = []
    if circles is not None:
        for (index, (x,y,r)) in enumerate(circles):
            neighbors = 0
            # Check if the current circle intersects with any other circle
            for (x2, y2, r2) in circles:
                if (x, y) != (x2, y2):
                    pixel1 = np.array([x, y], dtype=np.int32)
                    pixel2 = np.array([x2, y2], dtype=np.int32)
                    # Calculate the Euclidean distance
                    distance = np.linalg.norm(pixel2 - pixel1)
                    # Check if the circles intersect
                    if 2*r < distance < 5*r:
                        neighbors += 1
                        if neighbors > 2:
                            # Circle has neighboring circles
                            break
            if neighbors < 3:
                list_index.append(index)
    circles = np.delete(circles, list_index, axis=0)
    return circles

def get_non_overlapping_circles(circles):

    # Create a list to store non-overlapping circles
    non_overlapping_circles = []
    
    # Iterate over the circles
    for (x1, y1, r1) in circles:
    
    # Flag to indicate if the current circle overlaps with any previously added circles
        overlaps = False
            
    # Check for overlap with previously added circles
        for (x2, y2, r2) in non_overlapping_circles:
        # Calculate the distance between the centers of the two circles
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            # Check if the circles intersect
            if distance < r1 + r2:
                overlaps = True
                break
            
        # Add the circle to the non-overlapping list if it doesn't overlap with any previously added circles
        if not overlaps:
            non_overlapping_circles.append((x1, y1, r1))
    return non_overlapping_circles

def detect_circles(img_name:str, image, answers:int):
    # Find circles in the image
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, minDist=2, param1=20, param2=17, minRadius=15, maxRadius=24)
    mask = np.zeros_like(image)
    # Verify there are circles in the image
    if circles[0,:] is not None: 
        circles = np.int32(np.around(circles))[0,:]
        # Get non-overlapping circles
        circles = get_non_overlapping_circles(circles)
        # Draw detected circles 
        for (x,y,r) in circles:
            # cv.circle(image, (x, y), r, (0), thickness=-1)
            cv.circle(mask, (x, y), r, (0), thickness=-1)
        # Raise error if not find image
        if(len(circles) != answers*4):
            print(f'Error en la detección en la imagen: {img_name}')
            print(f'Detectó {len(circles)} círculos de {answers*4} esperados')
            show_img('Mala', image, 2)
    return circles, mask

def is_selected(thresh, circle):
    radius = int(4*circle[2]/5)
    roi = thresh[circle[1]-radius:circle[1]+radius, circle[0]-radius:circle[0]+radius]

    average_intensity = roi.mean()
    if average_intensity < 90:
        return True
    return False

def img2mtx(folder, name, img, circles, answers):
    ans_mtx = np.zeros((answers, 4), dtype=bool)
    kernel = 19
    blur = cv.GaussianBlur(img, (kernel, kernel), 0)
    blur = cv.GaussianBlur(blur, (kernel, kernel), 0)
    ret, thresh = cv.threshold(blur, 170, 255, cv.THRESH_BINARY)
    blur = cv.GaussianBlur(thresh, (kernel, kernel), 0)
    ret, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)
    columns = [[] for _ in range(4)]  
    for circle in circles:
        x, y = circle[:2]
        column = x // (img.shape[1] // 4)
        columns[column].append(circle)

    for i, column in enumerate(columns):
        for j, circle in enumerate(sorted(column, key=lambda x: (x[1]))):
            if j >= answers:
                print(f'{j} es mayor que {answers}')
                show_img('Img', img, 2)
            ans_mtx[j, i] = is_selected(thresh, circle)
            x, y, r = circle
            if ans_mtx[j, i]:
                cv.circle(img, (x, y), r, (0, 255, 0), thickness=-1)
            else:
                cv.circle(img, (x, y), r, (0, 0, 255), thickness=2)
    selected = np.count_nonzero(ans_mtx)

    cv.imwrite(f'{folder}/{name}.png', img)

    if selected != answers:
        print(f'Found {selected} of {answers} options')
        show_img('Img', img, 2)
        show_img('Thresh', thresh, 2)
    return ans_mtx



def rasch_likelihood(theta, beta, response):
    """
    Rasch model likelihood function for a single response to an item.

    Parameters:
        theta (float): Individual ability parameter (latent trait).
        beta (float): Item parameter (difficulty).
        response (int): Binary response to the item (0 or 1).

    Returns:
        float: Log-likelihood of the response given theta and beta.
    """
    logit = theta - beta
    prob_correct = 1 / (1 + np.exp(-logit))
    likelihood = (response * prob_correct) + ((1 - response) * (1 - prob_correct))
    return np.log(likelihood)

def rasch_log_likelihood(params, responses, n_students, n_items):
    """
    Rasch model log-likelihood function for all responses.

    Parameters:
        params (array): Array containing theta (individual abilities) and beta (item parameters).
        responses (array): Array of tuples (student_id, item_id, boolean_ans) representing the responses.
        n_students (int): Total number of unique students.
        n_items (int): Total number of unique items.

    Returns:
        float: Negative log-likelihood of the Rasch model given the data and parameters.
    """
    theta = params[:n_students]
    beta = params[n_students:]

    total_log_likelihood = 0
    for student_id, item_id, boolean_ans in responses:
        idx_student = student_id - 1
        idx_item = item_id - 1
        total_log_likelihood += rasch_likelihood(theta[idx_student], beta[idx_item], boolean_ans)

    return -total_log_likelihood

def estimate_rasch_model(responses, n_students, n_items):
    """
    Estimate the Rasch model parameters using MLE.

    Parameters:
        responses (list): List of tuples (student_id, item_id, boolean_ans) representing the responses.
        n_students (int): Total number of unique students.
        n_items (int): Total number of unique items.

    Returns:
        tuple: Tuple containing the estimated theta (individual abilities) and beta (item parameters).
    """
    initial_params = np.zeros(n_students + n_items)

    # Perform optimization
    result = minimize(rasch_log_likelihood, initial_params, args=(responses, n_students, n_items), method='L-BFGS-B')

    # Get the estimated parameters
    estimated_params = result.x
    theta = estimated_params[:n_students]
    beta = estimated_params[n_students:]

    return theta, beta


def visual_check(n_students, n_items, theta_est, beta_est):
    # Scatter plot for estimated theta (individual abilities)
    plt.scatter(range(1, n_students + 1), theta_est)
    plt.xlabel('Student ID')
    plt.ylabel('Estimated Theta')
    plt.title('Estimated Individual Abilities')
    plt.show()

    # Scatter plot for estimated beta (item parameters)
    plt.scatter(range(1, n_items + 1), beta_est)
    plt.xlabel('Item ID')
    plt.ylabel('Estimated Beta')
    plt.title('Estimated Item Parameters')
    plt.show()

def infit_outfit_statistics(responses, theta_est, beta_est):
    """
    Calculate Infit and Outfit statistics for each item.

    Parameters:
        responses (list): List of tuples (student_id, item_id, boolean_ans) representing the responses.
        theta_est (array): Array of estimated individual abilities (theta).
        beta_est (array): Array of estimated item parameters (beta).

    Returns:
        dict: Dictionary containing Infit and Outfit statistics for each item.
    """
    item_stats = {}

    for item_id in range(1, len(beta_est) + 1):
        item_responses = [(student_id, ans) for student_id, item, ans in responses if item == item_id]
        item_theta = np.array([theta_est[student_id - 1] for student_id, ans in item_responses])
        item_ans = np.array([ans for student_id, ans in item_responses])

        logit = item_theta - beta_est[item_id - 1]
        prob_correct = 1 / (1 + np.exp(-logit))
        expected_responses = np.round(prob_correct)
        squared_diff = (item_ans - expected_responses) ** 2

        num_responses = len(item_responses)
        infit = np.sum(squared_diff) / num_responses
        outfit = np.sum(squared_diff) / (num_responses + 1)

        item_stats[item_id] = {'Infit': infit, 'Outfit': outfit}

    return item_stats