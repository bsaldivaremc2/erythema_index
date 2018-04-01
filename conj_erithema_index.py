import numpy as np
import pandas as pd
from PIL import Image
import os
from matplotlib.path import Path
import matplotlib.pyplot as plt

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
class Conjunctiva_ei:
    """
    Library to extract the erithema index (EI) given the polygon coordinates of
    a square of calibration and the conjuctiva.
    Based on the paper:
        Collings S, Thompson O, Hirst E, Goossens L, George A, Weinkove R (2016) NonInvasive
        Detection of Anaemia Using Digital Photographs of the Conjunctiva. PLoS ONE 11(4):
        e0153286. doi:10.1371/journal.pone.0153286
    Coded by:
    Bryan P. Saldivar-Espinoza.
    bsaldivar.emc2@gmail.com
    """
    def __init__(self,df,image_index,filename_col,square_poly_col,conj_poly_col,reference_factor=200,epsilon=1e-6):
        """
        Define  the dataframe that contains:
        * square polygon coordinates
        * conjunctiva polygon coordinates
        """
        self.df=df
        self.image_index = image_index
        self.filename_col=filename_col
        self.square_poly_col=square_poly_col
        self.conj_poly_col=conj_poly_col
        for c in [self.filename_col,self.square_poly_col,self.conj_poly_col]:
            if c not in self.df.columns:
                print(c,"not present in DataFrame columns")
        
        self.reference_factor=reference_factor
        self.epsilon=epsilon
        
        self.img_points=None       
        self.EI=None
        self.EI_mean=None
        self.np_image=None
        self.ei_image=None
        
    def generate_template_image(self,v=False):
        """
        # Create a template of tuples that contains each pair of coordinates of a numpy image.
        * df: DataFrame that contains the image location:
        ** filename_col: Column name in df that holds the full path to the image to be opened.
        ** image_index: index of the df of the desired file to be opened. Row of df.
        *v: verbosity: Set to True to see the progress.
        """
        img_file = self.df[self.filename_col][self.image_index]
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
        self.img_points = np.vstack((x,y)).T 
        
    def polygon_rgb_matrix (self,polygon,img_points,numpy_image):
        nis = numpy_image.shape
        tupVerts=polygon
        p = Path(tupVerts) # make a polygon
        grid = p.contains_points(img_points)
        mask = grid.reshape(nis[0],nis[1])
        output = numpy_image[:,:,:][mask]
        return output
    def rgb_poly_matrix(self,iDf,_index,file_name_column,polygon_column,img_points):
        i_ = iDf[file_name_column][_index]
        p_ = iDf[polygon_column][_index]
        i__ = Image.open(i_)
        ni = np.asarray(i__)
        i__.close()
        return self.polygon_rgb_matrix(p_,img_points,ni)
    def polygon_rgb (self,polygon,img_points,numpy_image):
        nis = numpy_image.shape
        tupVerts=polygon
        p = Path(tupVerts) # make a polygon
        grid = p.contains_points(img_points)
        mask = grid.reshape(nis[0],nis[1])
        return numpy_image[mask]
    def rgb_poly(self,iDf,_index,file_name_column,polygon_column,img_points):
        i_ = iDf[file_name_column][_index]
        p_ = iDf[polygon_column][_index]
        i__ = Image.open(i_)
        ni = np.asarray(i__)
        i__.close()
        return self.polygon_rgb (p_,img_points,ni)

    def erythema_index(self,input_matrix,input_type='image',v=False,epsilon=1e-6):
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
    def calibration_ei_image(self,df,file_name_column,square_poly_col,conj_poly_col,
                    img_points,image_index=0,reference_factor=200,v=False,epsilon=1e-6):
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
        rgb_s = self.rgb_poly(df,_index=image_index,file_name_column=file_name_column,polygon_column=square_poly_col,
                     img_points=img_points)
        F = reference_factor/rgb_s.mean(0)
        if v==True:
            print(F)
        img_file = df[file_name_column][image_index]
        img_pil = Image.open(img_file)
        img_np = np.asarray(img_pil)
        img_pil.close()

        rgb_cF = img_np*F
        EI = self.erythema_index(input_matrix=rgb_cF,input_type='image',v=v,epsilon=epsilon)    
        EI_conj = self.polygon_rgb (df[conj_poly_col][image_index],img_points,EI)
        output = EI_conj
        """
        if return_only_mean==True:
            output = EI_conj.mean()
        """
        if v==True:
            print_stats(EI_conj,name="EI conj")
        return [img_np,EI,output]
    def calibration_ei(self,df,image_index,file_name_column,square_poly_col,
                       conj_poly_col,img_points,reference_factor=200,v=False,epsilon=1e-6):
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
        rgb_c = self.rgb_poly_matrix(df,_index=image_index,file_name_column=file_name_column,polygon_column=conj_poly_col,
                     img_points=img_points)
        rgb_s = self.rgb_poly_matrix(df,_index=image_index,file_name_column=file_name_column,polygon_column=square_poly_col,
                     img_points=img_points)
        F = reference_factor/rgb_s.mean(0)
        if v==True:
            print(F)
        rgb_cF = rgb_c*F
        EI = self.erythema_index(input_matrix=rgb_cF,input_type='matrix',v=v,epsilon=epsilon)
        return EI.copy()
    def calc_EI(self,v=False):
        self.EI = self.calibration_ei(df=self.df,file_name_column=self.filename_col,
                             square_poly_col=self.square_poly_col,conj_poly_col=self.conj_poly_col,
                        img_points=self.img_points,image_index=self.image_index,
                       reference_factor=self.reference_factor,v=v,epsilon=self.epsilon)
        self.EI_mean=self.EI.mean()
        if v==True:
            print("EI mean:",self.EI_mean)
    def calc_EI_and_image(self,v=False):
        self.np_image,self.ei_image,self.EI = self.calibration_ei_image(df=self.df,file_name_column=self.filename_col,
                             square_poly_col=self.square_poly_col,conj_poly_col=self.conj_poly_col,
                        img_points=self.img_points,image_index=self.image_index,
                       reference_factor=self.reference_factor,v=v,epsilon=self.epsilon)
        self.EI_mean=self.EI.mean()
        if v==True:
            print("Image loaded into self.np_image. EI image loaded into self.ei_image")
    def show_np_image(self,figsize=(4,6)):
        if type(self.np_image)!=type(None):
            plt.figure(figsize=figsize)
            plt.imshow(self.np_image)
            plt.show()
        else:
            print("Image not set")
    def show_ei_image(self,figsize=(4,6),cmap="gray"):
        if type(self.ei_image)!=type(None):
            plt.figure(figsize=figsize)
            plt.imshow(self.ei_image,cmap=cmap)
            plt.show()
        else:
            print("Image not set")