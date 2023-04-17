import numpy as np
import cv2 as cv

# preprocess and predict mask from image 
def predict_model_mask(img, model):
    # prepare picture for prediction
    x = np.zeros((1,) + img_size + (1,), dtype="uint8")
    x[0] = np.expand_dims(img, 2)

    predicted = model.predict(x,verbose=0)

    mask = np.argmax(predicted[0], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    return mask

# get image region of interest 
def img_roi(idx_x, idx_y,offset_x,offset_y,img):
    # Calculate pixels for region of interest
    min_x=np.amin(idx_x)-offset_x
    max_x=np.amax(idx_x)+offset_x
    min_y=np.amin(idx_y)-offset_y
    max_y=np.amax(idx_y)+offset_y
    region_of_interest = img[min_y:max_y,min_x:max_x]
    return region_of_interest,[min_x,max_x,min_y,max_y] 


# get the area and the center of image 
def img_area_centers(img):
    # search black pixel in image
    idx_y, idx_x=np.where(img==1)
    # get center and area
    if idx_x.size !=0:
        cX=np.sum(idx_x)/idx_x.size
    else:
        cX=np.nan
    if idx_y.size !=0:
        cY=np.sum(idx_y)/idx_y.size
    else:
        cY=np.nan
    center=[cX,cY]
    pixel_area=idx_x.size
     
    return center, pixel_area



# compare ref and predicted mask and use only pixels that are 1 in both images
def get_right_pixels_of_part(ref, dilation, mask):
    # set everything to zero and convert from 2D to 1D
    mask_part_only = np.zeros((ref.shape[0],ref.shape[1]))
    mask_part_only_flat=mask_part_only.flatten(order='C')
    ref_flat=dilation.flatten(order='C')
    mask_flat=mask.flatten(order='C')
    
    # iterate over every pixel and check if dilated and mask are both black
    for index in range(0, len(ref_flat)):
        if ref_flat[index]==1 and mask_flat[index]==1:
            mask_part_only_flat[index]=1
    
    # reshape from 1D to 2D
    mask_part_only = np.reshape(mask_part_only_flat, (-1, ref.shape[1]))
    return mask_part_only


def unit_vector(vector):
    # Returns the unit vector of the vector
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    # Returns the angle in radians between vectors 'v1' and 'v2'
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# norm the area according to a "count" 
def norm_results(count, reference, predicted):
    # convert into arrays
    ref_array=np.asarray(reference)
    pred_array=np.asarray(predicted)
    # if count>0 take the first count elements and norm
    if count>=0:
        normed_array=ref_array[0:count]/pred_array[0:count]
    # if count < 0 take the first count valid (non zero) prediction elements and norm 
    else:
        count=-count
        #delete zeros in prediction and take same values from reference
        pred_no_zeros=pred_array[pred_array!=0]
        ref_no_zeros=ref_array[pred_array!=0]
        #check if enough values are left in no_zeros arrays
        if (len(ref_no_zeros)>=count) and (len(pred_no_zeros)>=count):
            normed_array=ref_no_zeros[0:count]/pred_no_zeros[0:count]
        else:
            normed_array=ref_no_zeros/pred_no_zeros

    #drop inf and nan
    df=pd.DataFrame(normed_array)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    normed_array=np.asarray(df)
    
    # take the mean of all entries 
    normed=np.mean(normed_array)
    return normed

# calculate gradient of two values (with respect to height)
def calc_gradient(data):
    gradient=[(y1-y0) for y1, y0 in zip(data[1:], data)]
    return gradient

# XOR image
def substract_mask_from_ref(ref_prepared, mask):
    ref_prepared[ref_prepared > 1] = 1
    subtracted=cv.bitwise_xor(ref_prepared,mask)
    return subtracted
    
    