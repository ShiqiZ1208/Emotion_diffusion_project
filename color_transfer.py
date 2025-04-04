import cv2
import numpy as np

def lab_color_transfer(content_img, style_img, alpha):
    # Convert images to LAB color space
    content_lab = cv2.cvtColor(content_img, cv2.COLOR_BGR2LAB)
    style_lab = cv2.cvtColor(style_img, cv2.COLOR_BGR2LAB)
    
    # Split the Lab images into L, A, and B channels
    l_s, a_s, b_s = cv2.split(content_lab)
    l_t, a_t, b_t = cv2.split(style_lab)
    
    # Compute the mean and standard deviation of the L, A, and B channels
    l_s_mean, l_s_std = cv2.meanStdDev(l_s)
    a_s_mean, a_s_std = cv2.meanStdDev(a_s)
    b_s_mean, b_s_std = cv2.meanStdDev(b_s)
    l_t_mean, l_t_std = cv2.meanStdDev(l_t)
    a_t_mean, a_t_std = cv2.meanStdDev(a_t)
    b_t_mean, b_t_std = cv2.meanStdDev(b_t)
    
    # Apply the color transfer by adjusting the A and B channels (and preserving L)
    i_l_s = alpha*((l_s - l_s_mean) / l_s_std * l_t_std + l_t_mean)
    i_a_s = alpha*((a_s - a_s_mean) / a_s_std * a_t_std + a_t_mean)
    i_b_s = alpha*((b_s - b_s_mean) / b_s_std * b_t_std + b_t_mean)
    
    l_s = (1-alpha)*l_s + i_l_s
    a_s = (1-alpha)*a_s + i_a_s
    b_s = (1-alpha)*b_s + i_b_s
    # Clip the values to valid range
    l_s = np.clip(l_s, 0, 255).astype(np.uint8)
    a_s = np.clip(a_s, 0, 255).astype(np.uint8)
    b_s = np.clip(b_s, 0, 255).astype(np.uint8)
    
    # Merge the adjusted channels back
    result_lab = cv2.merge([l_s, a_s, b_s])
    
    # Convert back to BGR color space
    styled_image = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    
    return styled_image