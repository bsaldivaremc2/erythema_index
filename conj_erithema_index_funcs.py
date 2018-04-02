import numpy as np
import pandas as pd
from PIL import Image
import os
from matplotlib.path import Path
import matplotlib.pyplot as plt
import imp


def points(regs,max_points={'0':11,'1':4},poly_names ={'0':'p','1':'s'},sufix='_polygon'):
    """
    Help function to extract points from dataFrame json columns. 
    Creates a column called polygons for each polygon in the DataFrame
    *regs: dictionary that is inside a column of a a dataframe holding the coordinates
    *max_points: number of points to extract from the regions dictionary. 
     It is useful when errors are present in the number of points per polygon.
    *poly_names: Is the mapping relation between the polygon order [0,1] to the initials [p,s]
     respectively. p and s will appear as a prefix in the points column names.
    *sufix: is the sufix to be added in the polygon columns. 
     the polygon columns will take as prefix the values in poly_names
    """
    k = regs.keys()
    xpd,ypd = {},{}
    for __ in k:
        d = regs[__]['shape_attributes']
        xps = d['all_points_x']
        yps = d['all_points_y']        
        kn = poly_names[__]
        polygon=[]
        for ___ in range(0,max_points[__]):
            xpd[kn+'x'+str(___)]=xps[___]
            ypd[kn+'y'+str(___)]=yps[___]
            polygon.append((xps[___],yps[___]))
        xpd[kn+sufix]=polygon
    xpd.update(ypd)
    return xpd.copy()

def generate_template_image(df,filename_col,image_index=0,v=False):
    """
    # Create a template of tuples that contains each pair of coordinates of a numpy image.
    * df: DataFrame that contains the image location:
    ** filename_col: Column name in df that holds the full path to the image to be opened.
    ** image_index: index of the df of the desired file to be opened. Row of df.
    *v: verbosity: Set to True to see the progress.
    """
    img_file = df[filename_col][image_index]
    if v==True:
        print("Opening image:",img_file)
    img_pil = Image.open(img_file)
    img_np = np.asarray(img_pil)
    img_pil.close()
    nis = img_np.shape
    if v==True:
        print("Creating image template of size:",nis)
    x, y = np.meshgrid(np.arange(nis[1]), np.arange(nis[0]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T 
    return points 

def polygon_rgb_matrix (polygon,img_points,numpy_image):
    nis = numpy_image.shape
    nis2 = (nis[0],nis[1])
    tupVerts=polygon
    p = Path(tupVerts) # make a polygon
    grid = p.contains_points(img_points)
    mask = grid.reshape(nis[0],nis[1])
    output = numpy_image[:,:,:][mask]
    return output
def rgb_poly_matrix(iDf,_index,file_name_column,polygon_column,img_points):
    i_ = iDf[file_name_column][_index]
    p_ = iDf[polygon_column][_index]
    i__ = Image.open(i_)
    ni = np.asarray(i__)
    i__.close()
    return polygon_rgb_matrix(p_,img_points,ni)

def polygon_rgb (polygon,img_points,numpy_image):
    nis = numpy_image.shape
    tupVerts=polygon
    p = Path(tupVerts) # make a polygon
    grid = p.contains_points(img_points)
    mask = grid.reshape(nis[0],nis[1])
    return numpy_image[mask]
def rgb_poly(iDf,_index,file_name_column,polygon_column,img_points):
    i_ = iDf[file_name_column][_index]
    p_ = iDf[polygon_column][_index]
    i__ = Image.open(i_)
    ni = np.asarray(i__)
    i__.close()
    return polygon_rgb (p_,img_points,ni)

def erythema_index(input_matrix,input_type='image',v=False,epsilon=1e-6):
    """
    # Calculate the erythema index.
    ##The log function present replicates what ImageJ does with 8-bit images:
    ### References: http://imagejdocu.tudor.lu/doku.php?id=gui:process:math
    input_matrix: is a numpy array
    input_type: if is image, an [with,height,3] matrix is expected
        if is matrix, an [pixels,3] matrix is expected
    v: if set to True will print information of the calculation process
    epsilon: is a small number added to the log function in order to avoid division by zero
    """
    def print_stats(inp,name="name"):
        print(name+" stats. Mean:",inp.mean(),"max:",inp.max(),"min:",inp.min())
    if type(input_matrix)==np.ndarray:
        ims = input_matrix.shape
        if input_type=='image':
            expected_dim = 3
        elif input_type == 'matrix':
            expected_dim = 2
        if len(ims)==expected_dim:
            if input_type=='image':
                red = input_matrix[:,:,0]
                green = input_matrix[:,:,1]
            elif input_type=='matrix':
                red = input_matrix[:,0]
                green = input_matrix[:,1]
                
            log_255 = np.log(255)
            
            red[red>255]=255
            if v==True:
                print_stats(red,name="red")
                print_stats(green,name="green")
                
            rlog = np.log(red+epsilon)            
            rlog = rlog*255/log_255
            rlog[rlog<0]=0
            glog = np.log(green+epsilon)
            glog = glog*255/log_255
            glog[glog<0]=0
            
            if v==True:
                print_stats(rlog,name="red log")
                print_stats(glog,name="green log")
            EI = rlog - glog
            if v==True:
                print_stats(EI,name="EI")
            return EI.copy()
        else:
            print("Please use an RGB image")
    else:
        print("Use a numpy array")

def callibration_ei(df,image_index,file_name_column,square_poly_col,conj_poly_col,img_points,
                   reference_factor=200,v=False,return_only_mean=True,epsilon=1e-6):
    """
    # Function that returns the Erithema index (EI) with the calibration process.   
    Outputs: EI value of the conjunctiva zone.  
    * df: DataFrame that contains the following information:  
    ** file_name_column: name of the column that has the full path to the image to be opened.  
    ** square_poly_col: name of the column that has the polygon (list of tuples) that surrounds   
     the calibration white square
    ** conj_poly_col: name of the column that has the polygon (list of tuples) that surrounds  
     the conjunctiva.  
    *img_points: image template numpy array used to find the pixels in the images given the polygons
    *image_index: the index of the image to be opened from the df
    *reference_factor: is the multiplication factor as referenced in the paper.
    *v: verbosity. if set to True displays information of the process.
    *return_only_mean: if set to True will return only the mean value of the conjunctiva region.
     if set to False will return a numpy array of all the pixels in this area.
    *epsilon: small value added to log calculation to avoid division by zero.
    """
    rgb_c = rgb_poly_matrix(df,_index=image_index,file_name_column=file_name_column,polygon_column=conj_poly_col,
                 img_points=img_points)
    rgb_s = rgb_poly_matrix(df,_index=image_index,file_name_column=file_name_column,polygon_column=square_poly_col,
                 img_points=img_points)
    F = reference_factor/rgb_s.mean(0)
    if v==True:
        print(F)
    rgb_cF = rgb_c*F
    EI = erythema_index(input_matrix=rgb_cF,input_type='matrix',v=v,epsilon=epsilon)
    if return_only_mean==True:
        return EI.mean()
    else:
        return EI
def calibration_ei_image(df,file_name_column,square_poly_col,conj_poly_col,
                img_points,image_index=0,
                   reference_factor=200,v=False,return_only_mean=True,epsilon=1e-6):
    """
    # Function to return the Erithema index (EI)  with the calibration process.      
    Outputs: Numpy image, EI image of one channel as numpy array and the EI value of the conjunctiva zone.  
    * df: DataFrame that contains the following information:  
    ** file_name_column: name of the column that has the full path to the image to be opened.  
    ** square_poly_col: name of the column that has the polygon (list of tuples) that surrounds   
     the calibration white square
    ** conj_poly_col: name of the column that has the polygon (list of tuples) that surrounds  
     the conjunctiva.  
    *img_points: image template numpy array used to find the pixels in the images given the polygons
    *image_index: the index of the image to be opened from the df
    *reference_factor: is the multiplication factor as referenced in the paper.
    *v: verbosity. if set to True displays information of the process.
    *return_only_mean: if set to True will return only the mean value of the conjunctiva region.
     if set to False will return a numpy array of all the pixels in this area.
    *epsilon: small value added to log calculation to avoid division by zero.
    """
    def print_stats(inp,name="name"):
        print(name+" stats. Mean:",inp.mean(),"max:",inp.max(),"min:",inp.min())
    rgb_s = rgb_poly(df,_index=image_index,file_name_column=file_name_column,polygon_column=square_poly_col,
                 img_points=img_points)
    F = reference_factor/rgb_s.mean(0)
    if v==True:
        print(F)
    img_file = df[file_name_column][image_index]
    img_pil = Image.open(img_file)
    img_np = np.asarray(img_pil)
    img_pil.close()

    rgb_cF = img_np*F
    EI = erythema_index(input_matrix=rgb_cF,input_type='image',v=v,epsilon=epsilon)    
    EI_conj = polygon_rgb (df[conj_poly_col][image_index],img_points,EI)
    output = EI_conj
    if return_only_mean==True:
        output = EI_conj.mean()
    if v==True:
        print_stats(EI_conj,name="EI conj")
    return [img_np,EI,output]
def ei_to_hb(iei,predictor='PCI',output_scale='g/L'):
    """
    Predict the Hemoglobin (Hb) levels given the Erythema index (EI).
    To predict the Hb select a predictor method PCI, PCLX5 and FCLX5:
    *'PCI':
    **'name':'Palpebral conjunctival EI (Iphone)',
    **'coef':1.38471542,
    **'intercept':70.118455,
    **'r2 score': 0.346012,
    *'PCLX5':
    **'name':'Palpebral conjunctival EI (LX5)',
    **'coef':2.92842233,
    **'intercept':57.647633,
    **'r2 score': 0.270722,
    *'FCLX5':
    **'name':'Forniceal conjunctival EI (LX5)',
    **'coef':2.18110503,
    **'intercept':84.138215,
    **'r2_score':0.088311}
    These values were computed using the data present in the paper and applying linear regression.
    
    Output scale is by default in 'g/L'. You can set it to 'g/dL' 
    """
    pred_params = {
    'PCI':{'name':'Palpebral conjunctival EI (Iphone)','coef':1.38471542,'intercept':70.118455,'r2_score':0.346012},
    'PCLX5':{'name':'Palpebral conjunctival EI (LX5)','coef':2.92842233,'intercept':57.647633,'r2_score':0.270722},
    'FCLX5':{'name':'Forniceal conjunctival EI (LX5)','coef':2.18110503,'intercept':84.138215,'r2_score':0.088311}
    }
    pred=pred_params[predictor]
    hb = iei*pred['coef']+pred['intercept']
    if output_scale=='g/dL':
        hb=hb/10
    return hb
